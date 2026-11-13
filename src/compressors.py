"""Sequence compressors: Mamba v1 (padded batches) and Mamba v2 (packed, no padding).

Both implement the BaseCompressor interface — forward(compressor_inputs) and
compress_single — returning (soft_prompts [B, C, H_llm], compressed_mask [B, C]).
See docs/architecture/compressors.md on the `main` branch.
"""
import torch
import torch.nn as nn
from transformers import MambaForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from .utils.compression import mamba1_context_gather, mamba2_context_start

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

    def __init__(self, llm_hidden_dim: int, device: torch.device, compression_ratio: float = 0.5):
        super().__init__()
        self.device = device
        self.stride = int(1 / compression_ratio)
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
                 compression_ratio: float = 0.5):
        super().__init__(llm_hidden_dim, device, compression_ratio)
        self.mamba = MambaForCausalLM.from_pretrained(model_name).to(device)
        self.projector = nn.Linear(self.mamba.config.hidden_size, llm_hidden_dim).to(device)

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
            max_real = int(real_len.max().item()) if B > 0 else seq_len
        else:
            raise ValueError("MambaCompressor requires attention_mask to determine real sequence lengths.")

        context_start = compressor_inputs.get('context_start')
        if context_start is not None:
            from_ = context_start.to(self.device).long()
            gather_idx_safe, valid_slot = mamba1_context_gather(
                seq_len, real_len, from_, self.stride
            )
        else:
            # B-qc / bare path: compress the whole real sequence (unchanged).
            # C sized to the longest real sequence in the batch.
            C = max(1, (max_real + self.stride - 1) // self.stride)
            slot_idx = torch.arange(C, device=self.device)
            # Slot C-1 = last real token; slot 0 = earliest strided position.
            offsets = (C - 1 - slot_idx) * self.stride                # [C]

            # With left-padding the last real token is always at index seq_len-1.
            last_real_pos = seq_len - 1
            gather_idx = last_real_pos - offsets                       # [C]
            gather_idx = gather_idx.unsqueeze(0).expand(B, -1)         # [B, C]
            first_real = seq_len - real_len                            # [B]
            valid_slot = (gather_idx >= first_real.unsqueeze(1)).to(attention_mask.dtype)
            gather_idx_safe = gather_idx.clamp_min(0)

        hidden_states = torch.gather(
            hidden, 1, gather_idx_safe.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        )  # [B, C, H_mamba]

        compressed_mask = valid_slot.to(attention_mask.dtype)

        return self.projector(hidden_states), compressed_mask

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
                 compression_ratio: float = 0.5, dtype: torch.dtype = torch.float32):
        super().__init__(llm_hidden_dim, device, compression_ratio)
        self.dtype = dtype
        self.mamba = MambaLMHeadModel.from_pretrained(model_name, device=device, dtype=dtype)
        self.mamba_hidden_size = self.mamba.config.d_model
        self.projector = nn.Linear(self.mamba_hidden_size, llm_hidden_dim).to(device=device, dtype=dtype)

    def forward(self, compressor_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        packed_input_ids = compressor_inputs['input_ids']   # [1, T]
        seq_idx = compressor_inputs['seq_idx']              # [1, T] int32
        lengths = compressor_inputs['lengths']              # [B]

        # seq_idx propagates through MambaLMHeadModel → MixerModel → Block → Mamba2
        # via **mixer_kwargs, resetting the SSM state at each sequence boundary.
        hidden_states = self.mamba.backbone(packed_input_ids, seq_idx=seq_idx)
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

        soft_prompts = self.projector(soft_prompts)
        # Zero pad positions so downstream code that ignores the mask sees zeros,
        # not the projector bias at those slots.
        soft_prompts = soft_prompts * compressed_mask.unsqueeze(-1).to(soft_prompts.dtype)

        return soft_prompts, compressed_mask

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
