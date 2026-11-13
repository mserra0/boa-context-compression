"""Sequence compressors: Mamba v1 (padded batches) and Mamba v2 (packed, no padding)."""
import torch
import torch.nn as nn
from transformers import MambaForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from .utils.compression import mamba1_context_gather, mamba2_context_start, strided_gather

"""notation: B batch, I input len, C compressed len, H_mamba Mamba hidden, H_llm LLM hidden, S stride."""


class BaseCompressor(nn.Module):
    """Abstract base class for sequence compressors."""

    def __init__(self, llm_hidden_dim: int, device: torch.device, stride: int = 2):
        super().__init__()
        self.device = device
        self.stride = stride
        self.llm_hidden_dim = llm_hidden_dim

    def forward(self, compressor_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress a batch -> (soft_prompts [B, C, H_llm], compressed_mask [B, C])."""
        raise NotImplementedError

    @torch.no_grad()
    def compress_single(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress a single non-padded sequence -> (soft_prompts [1, C, H_llm], compressed_mask [1, C])."""
        raise NotImplementedError


class MambaCompressor(BaseCompressor):
    """Mamba (v1) compressor for left-padded batches."""

    def __init__(self, model_name: str, llm_hidden_dim: int, device: torch.device,
                 stride: int = 2):
        super().__init__(llm_hidden_dim, device, stride)
        self.mamba = MambaForCausalLM.from_pretrained(model_name).to(device)
        self.mamba_hidden_size = self.mamba.config.hidden_size
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
            real_len = attention_mask.sum(dim=1).long()  # [B]
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
            hidden_states, compressed_mask = strided_gather(hidden, attention_mask, self.stride)

        return self.projector(hidden_states), compressed_mask

    def gather_hidden(self, compressor_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw strided Mamba hidden [B, C, H_mamba] + mask BEFORE embedding projector (for KV path)."""
        input_ids = compressor_inputs['input_ids']
        attention_mask = compressor_inputs['attention_mask']
        outputs = self.mamba(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]  # [B, I, H_mamba]

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
    """Mamba2 compressor for variable-length packed batches (no padding)."""

    def __init__(self, model_name: str, llm_hidden_dim: int, device: torch.device,
                 stride: int = 2, dtype: torch.dtype = torch.float32):
        super().__init__(llm_hidden_dim, device, stride)
        self.dtype = dtype
        self.mamba = MambaLMHeadModel.from_pretrained(model_name, device=device, dtype=dtype)
        self.mamba_hidden_size = self.mamba.config.d_model
        self.projector = nn.Linear(self.mamba_hidden_size, llm_hidden_dim).to(device=device, dtype=dtype)

    def _gather_strided(self, compressor_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw strided, left-padded hidden [B, C, H_mamba] + mask [B, C] BEFORE projector."""
        packed_input_ids = compressor_inputs['input_ids']  # [1, T]
        seq_idx = compressor_inputs['seq_idx']  # [1, T] int32
        lengths = compressor_inputs['lengths']  # [B]

        hidden_states = self.mamba.backbone(packed_input_ids, seq_idx=seq_idx)
        hidden_states = hidden_states.squeeze(0)  # [T, H_mamba]

        context_start = compressor_inputs.get('context_start')
        c0_list = context_start.tolist() if context_start is not None else None
        compressed_list = []
        offset = 0
        for i, L in enumerate(lengths.tolist()):
            seq_hidden = hidden_states[offset:offset + L]  # [L, H_mamba]
            if c0_list is not None:
                start = mamba2_context_start(L, int(c0_list[i]), self.stride)
            else:
                start = (L - 1) % self.stride
            compressed_list.append(seq_hidden[start::self.stride])
            offset += L

        compressed_lengths = [c.shape[0] for c in compressed_list]
        C_max = max(1, max(compressed_lengths))
        B = len(compressed_list)

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

        return soft_prompts, compressed_mask

    def forward(self, compressor_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        soft_prompts, compressed_mask = self._gather_strided(compressor_inputs)
        soft_prompts = self.projector(soft_prompts)
        soft_prompts = soft_prompts * compressed_mask.unsqueeze(-1).to(soft_prompts.dtype)
        return soft_prompts, compressed_mask

    def gather_hidden(self, compressor_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Raw strided pre-projector hidden + mask for KV path."""
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

