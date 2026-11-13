"""Sequence compressors: Mamba v1 (padded batches) and Mamba v2 (packed, no padding).

Both implement the BaseCompressor interface — forward(compressor_inputs) and
compress_single — returning (soft_prompts [B, C, H_llm], compressed_mask [B, C]).
See docs/architecture/compressors.md on the `main` branch.
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint
from transformers import MambaForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel, RMSNorm, layer_norm_fn

from .utils.compression import mamba1_context_gather, mamba2_context_start, strided_gather

"""notation:
    - B: batch size
    - I: input sequence length
    - C: compressed sequence length
    - H_mamba: hidden dimension of Mamba
    - H_llm: hidden dimension of the LLM
    - S: stride (compression factor)
"""


class BaseCompressor(nn.Module):
    """Abstract base class for all sequence compressors.

    Subclasses must implement forward() and compress_single().
    Both methods return (soft_prompts [B, C, H_llm], compressed_mask [B, C]).
    """

    def __init__(self, llm_hidden_dim: int, device: torch.device, stride: int = 2):
        super().__init__()
        self.device = device
        self.stride = stride
        self.llm_hidden_dim = llm_hidden_dim

    def forward(self, compressor_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Run compression on a batch.

        Args:
            compressor_inputs: dict with compressor-specific keys.
                MambaCompressor  expects: input_ids [B, I], attention_mask [B, I]
                Mamba2Compressor expects: input_ids [1, T], seq_idx [1, T], lengths [B]
        Returns:
            soft_prompts:    [B, C, H_llm]
            compressed_mask: [B, C]  (1 = valid token, 0 = padding)
        """
        raise NotImplementedError

    @torch.no_grad()
    def compress_single(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress a single non-padded sequence. Convenience wrapper for inference.

        Args:
            input_ids: [1, L] or [L] tensor of token IDs.
        Returns:
            soft_prompts:    [1, C, H_llm]
            compressed_mask: [1, C]
        """
        raise NotImplementedError


class MambaCompressor(BaseCompressor):
    """Mamba (v1) compressor. Operates on left-padded batches.

    Input dict keys: input_ids [B, I], attention_mask [B, I].
    Strides are anchored to each sample's last real token so padding positions
    are never selected as compressed slots.
    """

    def __init__(self, model_name: str, llm_hidden_dim: int, device: torch.device,
                 stride: int = 2):
        super().__init__(llm_hidden_dim, device, stride)
        self.mamba = MambaForCausalLM.from_pretrained(model_name).to(device)
        self.mamba_hidden_size = self.mamba.config.hidden_size   # backend-agnostic K/V dim source
        self.projector = nn.Linear(self.mamba_hidden_size, llm_hidden_dim).to(device)

    def forward(self, compressor_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = compressor_inputs['input_ids']
        attention_mask = compressor_inputs.get('attention_mask')

        outputs = self.mamba(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1]  # [B, I, H_mamba]
        B, seq_len, _ = hidden.shape

        if attention_mask is not None:
            real_len = attention_mask.sum(dim=1).long()            # [B]
        else:
            raise ValueError("MambaCompressor requires attention_mask to determine real sequence lengths.")

        context_start = compressor_inputs.get('context_start')
        if context_start is not None:
            from_ = context_start.to(self.device).long()
            gather_idx_safe, valid_slot = mamba1_context_gather(
                seq_len, real_len, from_, self.stride
            )
            hidden_states = torch.gather(
                hidden, 1, gather_idx_safe.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
            )  # [B, C, H_mamba]
            compressed_mask = valid_slot.to(attention_mask.dtype)
        else:
            # B-qc / bare path: compress the whole real sequence.
            hidden_states, compressed_mask = strided_gather(hidden, attention_mask, self.stride)

        return self.projector(hidden_states), compressed_mask

    def gather_hidden(self, compressor_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw strided Mamba hidden [B, C, H_mamba] + mask, BEFORE the embedding projector.
        Used by the KV path. Mirrors `forward`'s gather: bare/B-qc compress the whole real
        sequence; B-ctx (context_start given) compresses only the context sub-range."""
        input_ids = compressor_inputs['input_ids']
        attention_mask = compressor_inputs['attention_mask']
        outputs = self.mamba(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]              # [B, I, H_mamba]

        context_start = compressor_inputs.get('context_start')
        if context_start is not None:
            seq_len = hidden.shape[1]
            real_len = attention_mask.sum(dim=1).long()
            from_ = context_start.to(self.device).long()
            gather_idx_safe, valid_slot = mamba1_context_gather(
                seq_len, real_len, from_, self.stride
            )
            hidden_states = torch.gather(
                hidden, 1, gather_idx_safe.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
            )
            return hidden_states, valid_slot.to(attention_mask.dtype)

        return strided_gather(hidden, attention_mask, self.stride)

    @torch.no_grad()
    def compress_single(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, L = input_ids.shape
        attention_mask = torch.ones((B, L), device=self.device, dtype=torch.long)
        return self.forward({'input_ids': input_ids, 'attention_mask': attention_mask})


class Mamba2Compressor(BaseCompressor):
    """Mamba2 compressor. Operates on variable-length packed batches (no padding).

    Input dict keys:
        input_ids [1, T]   — concatenated non-pad token IDs for the whole batch.
        seq_idx   [1, T]   — int32; seq_idx[0, t] = i where token t belongs to sample i.
        lengths   [B]      — int64; number of tokens per sample (no padding counted).

    After the backbone the flat hidden states are split per-sequence, strided to
    produce compressed soft prompts, projected to H_llm, and left-padded into a
    (B, C_max, H_llm) batch (mirroring the teacher's left-padded input).
    """

    def __init__(self, model_name: str, llm_hidden_dim: int, device: torch.device,
                 stride: int = 2, dtype: torch.dtype = torch.float32,
                 gradient_checkpointing: bool = True):
        super().__init__(llm_hidden_dim, device, stride)
        self.dtype = dtype
        self.mamba = MambaLMHeadModel.from_pretrained(model_name, device=device, dtype=dtype)
        self.mamba_hidden_size = self.mamba.config.d_model
        self.projector = nn.Linear(self.mamba_hidden_size, llm_hidden_dim).to(device=device, dtype=dtype)
        # Optional memory-profiling hook, wired up externally (see LMPretrainer /
        # StageMemoryProfiler in src/utils/memory_profile.py). No-op by default so this
        # costs nothing when profiling is off.
        self.mem_stage_fn = lambda tag: None
        # Trades ~2x backbone compute for O(1) instead of O(n_layer) peak activation
        # memory: each layer is checkpointed individually (NOT the backbone as a single
        # block, which would recompute the whole forward at once and save nothing), so
        # backward only holds one layer's activations at a time. Needed to fit long
        # packed [1, T] sequences on a single GPU.
        self.gradient_checkpointing = gradient_checkpointing

    def _backbone_forward(self, input_ids: torch.Tensor, seq_idx: torch.Tensor) -> torch.Tensor:
        """Re-implements MixerModel.forward (mamba_ssm.models.mixer_seq_simple), layer by
        layer, so each layer can be individually gradient-checkpointed. seq_idx is passed
        through to every layer's Mamba2 mixer to reset SSM state at packed-sequence
        boundaries, exactly as the stock backbone(...) call does."""
        backbone = self.mamba.backbone
        hidden_states = backbone.embedding(input_ids)
        residual = None
        for layer in backbone.layers:
            if self.gradient_checkpointing and self.training:
                # use_reentrant=False crashes inside mamba_ssm's fused Triton kernel
                # (mamba_split_conv1d_scan_combined raises a raw SystemError when
                # recomputed under non-reentrant checkpoint) -- use_reentrant=True
                # avoids it, but doesn't accept kwargs, so seq_idx is closed over via
                # a small positional-only wrapper instead of passed as a checkpoint kwarg.
                def _layer_fwd(hs, res, _layer=layer, _seq_idx=seq_idx):
                    return _layer(hs, res, seq_idx=_seq_idx)
                hidden_states, residual = torch.utils.checkpoint.checkpoint(
                    _layer_fwd, hidden_states, residual, use_reentrant=True,
                )
            else:
                hidden_states, residual = layer(hidden_states, residual, seq_idx=seq_idx)

        if not backbone.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = backbone.norm_f(residual.to(dtype=backbone.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states, backbone.norm_f.weight, backbone.norm_f.bias,
                eps=backbone.norm_f.eps, residual=residual, prenorm=False,
                residual_in_fp32=backbone.residual_in_fp32,
                is_rms_norm=isinstance(backbone.norm_f, RMSNorm),
            )
        return hidden_states

    def _gather_strided(self, compressor_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw strided, left-padded hidden [B, C, H_mamba] + mask [B, C], BEFORE the
        projector. Shared by `forward` (which then projects) and `gather_hidden` (KV path)."""
        packed_input_ids = compressor_inputs['input_ids']   # [1, T]
        seq_idx = compressor_inputs['seq_idx']              # [1, T] int32
        lengths = compressor_inputs['lengths']              # [B]

        # seq_idx propagates through MambaLMHeadModel → MixerModel → Block → Mamba2
        # via **mixer_kwargs, resetting the SSM state at each sequence boundary.
        self.mem_stage_fn("mamba2.pre_backbone")
        hidden_states = self._backbone_forward(packed_input_ids, seq_idx)
        self.mem_stage_fn("mamba2.post_backbone")
        hidden_states = hidden_states.squeeze(0)  # [T, H_mamba]

        # Split per-sequence and apply stride aligned to the last real token (or, when
        # context_start is provided, to the last token of the context sub-range).
        context_start = compressor_inputs.get('context_start')
        c0_list = context_start.tolist() if context_start is not None else None
        compressed_list = []
        offset = 0
        for i, L in enumerate(lengths.tolist()):
            seq_hidden = hidden_states[offset:offset + L]      # [L, H_mamba]
            if c0_list is not None:
                start = mamba2_context_start(L, int(c0_list[i]), self.stride)
            else:
                start = (L - 1) % self.stride
            compressed_list.append(seq_hidden[start::self.stride])
            offset += L

        compressed_lengths = [c.shape[0] for c in compressed_list]
        C_max = max(1, max(compressed_lengths))
        B = len(compressed_list)

        # Left-pad so the last valid soft prompt sits at C_max-1 for every sample,
        # mirroring the teacher's left-padded Qwen input at generation time.
        soft_prompts = torch.zeros(
            (B, C_max, self.mamba_hidden_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        compressed_mask = torch.zeros((B, C_max), device=hidden_states.device, dtype=torch.long)
        for i, (c, c_len) in enumerate(zip(compressed_list, compressed_lengths)):
            pad_len = C_max - c_len
            soft_prompts[i, pad_len:] = c
            compressed_mask[i, pad_len:] = 1

        self.mem_stage_fn("mamba2.post_stride_gather")
        return soft_prompts, compressed_mask

    def forward(self, compressor_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        soft_prompts, compressed_mask = self._gather_strided(compressor_inputs)
        soft_prompts = self.projector(soft_prompts)
        # Zero pad positions so downstream code that ignores the mask sees zeros,
        # not the projector bias at those slots.
        soft_prompts = soft_prompts * compressed_mask.unsqueeze(-1).to(soft_prompts.dtype)
        return soft_prompts, compressed_mask

    def gather_hidden(self, compressor_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw strided pre-projector hidden + mask for the KV path. Supports the packed
        bare/B-qc layout and B-ctx (context_start), mirroring `forward`'s gather."""
        return self._gather_strided(compressor_inputs)

    @torch.no_grad()
    def compress_single(self, input_ids: torch.Tensor,
                         context_start: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        L = input_ids.shape[1]
        seq_idx = torch.zeros((1, L), device=input_ids.device, dtype=torch.int32)
        lengths = torch.tensor([L], device=input_ids.device, dtype=torch.long)
        inputs = {'input_ids': input_ids, 'seq_idx': seq_idx, 'lengths': lengths}
        if context_start is not None:
            inputs['context_start'] = context_start
        return self.forward(inputs)
