"""Stride helpers for sequence compression."""
import torch


def strided_gather(hidden: torch.Tensor, attention_mask: torch.Tensor,
                   stride: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-anchored strided gather over left-padded batch.

    hidden [B, I, H], attention_mask [B, I] (1=real).
    Returns (gathered [B, C, H], compressed_mask [B, C]); slot C-1 is last real token.
    """
    B, seq_len, H = hidden.shape
    real_len = attention_mask.sum(dim=1).long()                 # [B]
    max_real = int(real_len.max().item()) if B > 0 else seq_len
    C = max(1, (max_real + stride - 1) // stride)
    slot_idx = torch.arange(C, device=hidden.device)
    offsets = (C - 1 - slot_idx) * stride                       # [C]
    last_real_pos = seq_len - 1                                 # left-padding invariant
    gather_idx = (last_real_pos - offsets).unsqueeze(0).expand(B, -1)   # [B, C]
    first_real = seq_len - real_len                             # [B]
    valid_slot = (gather_idx >= first_real.unsqueeze(1)).to(attention_mask.dtype)
    gather_idx_safe = gather_idx.clamp_min(0)
    gathered = torch.gather(
        hidden, 1, gather_idx_safe.unsqueeze(-1).expand(-1, -1, H)
    )
    return gathered, valid_slot


def mamba2_context_start(L: int, c0: int, stride: int) -> int:
    """Compute end-aligned stride start index in range [c0, L) for packed Mamba2 segment."""
    if c0 >= L:
        return L
    return c0 + ((L - 1 - c0) % stride)


def mamba1_context_gather(seq_len: int, real_len: torch.Tensor,
                          context_start: torch.Tensor, stride: int):
    """Compute end-aligned gather indices for left-padded Mamba1 batch context range.

    Returns (gather_idx [B, C], valid [B, C]). End anchor is seq_len - 1.
    """
    B = real_len.shape[0]
    device = real_len.device
    last_real_pos = seq_len - 1
    ctx_span = (last_real_pos - context_start + 1).clamp_min(0)     # [B]
    C = max(1, int(((ctx_span.max() + stride - 1) // stride).item()))
    slot_idx = torch.arange(C, device=device)
    offsets = (C - 1 - slot_idx) * stride                           # [C]
    gather_idx = last_real_pos - offsets                            # [C]
    gather_idx = gather_idx.unsqueeze(0).expand(B, -1)              # [B, C]
    valid = (gather_idx >= context_start.unsqueeze(1)) & (gather_idx <= last_real_pos)
    gather_idx_safe = gather_idx.clamp_min(0)
    return gather_idx_safe, valid.to(real_len.dtype)

