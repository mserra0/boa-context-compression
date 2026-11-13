"""Attention mask construction for student forward/generation paths."""
import torch

MASK_CACHE_CAPACITY = 64


def build_causal_mask(prefix_len: int, gen_len: int, device: torch.device) -> torch.Tensor:
    """Build fully-causal boolean mask over [prefix || gen], shape [1, 1, N, N] (N = prefix_len + gen_len)."""
    total_len = prefix_len + gen_len
    mask = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def build_additive_mask(causal_mask: torch.Tensor, full_padding_mask: torch.Tensor,
                        dtype: torch.dtype) -> torch.Tensor:
    """Combine causal mask with 2D padding mask into additive bias (0 keep, -inf block).

    Ensures diagonal is retained to prevent all-masked softmax NaNs.
    """
    final_mask = causal_mask & (full_padding_mask[:, None, None, :] > 0.5)
    final_mask.diagonal(dim1=-2, dim2=-1).fill_(True)
    attention_mask = torch.zeros_like(final_mask, dtype=dtype)
    attention_mask.masked_fill_(~final_mask, torch.finfo(dtype).min)
    return attention_mask

