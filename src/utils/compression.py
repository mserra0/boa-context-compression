"""Stride helpers for context-only compression (Design B-ctx).

End-aligned striding that keeps the last real token and raises the lower bound to the
context start, so only the context sub-range is compressed into soft prompts. The
question hidden states still condition the SSM but are never gathered.
See docs/architecture/student-templates.md on the `main` branch.
"""
import torch


def mamba2_context_start(L: int, c0: int, stride: int) -> int:
    """End-aligned stride start index within a packed Mamba2 segment, compressing only
    the context sub-range [c0, L). Keeps the last token (L-1), walks back by `stride`,
    floors at c0. Returns a start s.t. `range(start, L, stride)` are the kept positions.
    If c0 >= L (no context) the returned start is >= L -> empty slice."""
    if c0 >= L:
        return L
    return c0 + ((L - 1 - c0) % stride)


def mamba1_context_gather(seq_len: int, real_len: torch.Tensor,
                          context_start: torch.Tensor, stride: int):
    """End-aligned gather indices for the left-padded Mamba1 batch, compressing only the
    context sub-range. End anchor = last real token (seq_len-1). Lower bound = per-row
    `context_start` (absolute index in the [B, seq_len] tensor). Returns:
      gather_idx [B, C] (clamped >= 0 for safe gather),
      valid      [B, C] (1 where the slot lands in [context_start, seq_len)).
    C is sized to the longest context span in the batch."""
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
