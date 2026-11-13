"""Pack projected per-layer K/V tensors into a transformers DynamicCache with RoPE."""
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


def build_kv_cache(per_layer_kv: torch.Tensor, position_ids: torch.Tensor, rotary_emb,
                   decoder_layers=None) -> DynamicCache:
    """Pack projected K/V tensors into a DynamicCache with RoPE applied to keys.

    Args:
        per_layer_kv: [L, 2, B, kv_heads, C, head_dim] (axis 1: 0=keys, 1=values).
        position_ids: [B, C] integer positions assigned to compressed slots.
        rotary_emb: LLM's rotary embedding module.
        decoder_layers: optional list of LLM decoder layers for applying key norm (k_norm).

    Returns:
        DynamicCache populated with L layers.
    """
    L = per_layer_kv.shape[0]
    ref = per_layer_kv[0, 0]                        # [B, kv_heads, C, head_dim]
    cos, sin = rotary_emb(ref, position_ids)         # [B, C, head_dim]

    cache = DynamicCache()
    for layer_idx in range(L):
        k = per_layer_kv[layer_idx, 0]              # [B, kv_heads, C, head_dim]
        v = per_layer_kv[layer_idx, 1]              # [B, kv_heads, C, head_dim]
        if decoder_layers is not None:
            k_norm = getattr(decoder_layers[layer_idx].self_attn, "k_norm", None)
            if k_norm is not None:
                k = k_norm(k.to(k_norm.weight.dtype)).to(per_layer_kv.dtype)
        _, k_rot = apply_rotary_pos_emb(k, k, cos, sin)
        cache.update(k_rot.contiguous(), v.contiguous(), layer_idx)
    return cache

