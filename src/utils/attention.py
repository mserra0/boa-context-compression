"""Attention-mask construction for the student forward / generation paths.

The student is fully causal over [prefix | gen]; the 2D padding mask is intersected with
that causal geometry and converted to an additive bias. See docs/architecture/attention.md on the `main` branch.
"""
import torch

MASK_CACHE_CAPACITY = 64


def build_causal_mask(prefix_len: int, gen_len: int, device: torch.device) -> torch.Tensor:
    """Fully-causal boolean mask over [prefix || gen]. Shape [1, 1, N, N] where
    N = prefix_len + gen_len. True = attendable."""
    total_len = prefix_len + gen_len
    mask = torch.tril(torch.ones(total_len, total_len, dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def build_additive_mask(causal_mask: torch.Tensor, full_padding_mask: torch.Tensor,
                        dtype: torch.dtype) -> torch.Tensor:
    """Intersect the causal geometry with the 2D padding mask and convert to an additive
    bias (`0` keep, `-inf` block). Every position keeps its diagonal so a fully-padded
    row never produces an all-masked softmax (NaN guard)."""
    final_mask = causal_mask & (full_padding_mask[:, None, None, :] > 0.5)
    final_mask.diagonal(dim1=-2, dim2=-1).fill_(True)
    attention_mask = torch.zeros_like(final_mask, dtype=dtype)
    attention_mask.masked_fill_(~final_mask, torch.finfo(dtype).min)
    return attention_mask
