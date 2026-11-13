# src/utils/kv_cache.py
"""Pack projected per-layer K/V into a transformers DynamicCache, applying RoPE to keys.

Keys are rotated at their assigned compressed positions; values are never rotated.
transformers 5.9.0: build with DynamicCache().update(k, v, layer_idx) per layer.
"""
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


def build_kv_cache(per_layer_kv: torch.Tensor, position_ids: torch.Tensor, rotary_emb,
                   decoder_layers=None) -> DynamicCache:
    """Pack projected K/V tensors into a DynamicCache with RoPE applied to keys.

    Args:
        per_layer_kv: [L, 2, B, kv_heads, C, head_dim]
            Axis 1: 0 = keys, 1 = values.
        position_ids: [B, C] — integer positions assigned to each compressed slot.
        rotary_emb: the LLM's rotary embedding module (model.model.rotary_emb).
        decoder_layers: optional list of the LLM's decoder layers (model.model.layers).
            When a layer carries a per-head key norm (Qwen3's ``self_attn.k_norm``), the
            injected keys are passed through it BEFORE RoPE so they live in the same
            distribution as the model's real keys (real path: k_proj -> k_norm -> RoPE).
            Pass None (or layers without k_norm, e.g. Qwen2.5) to skip — behaviour is
            then identical to keys going straight to RoPE.

    Returns:
        DynamicCache with L layers populated; keys are (optionally k_norm'd then)
        RoPE-rotated, values are raw.
    """
    L = per_layer_kv.shape[0]
    # Use layer-0 keys as a reference for dtype/device only.
    ref = per_layer_kv[0, 0]                        # [B, kv_heads, C, head_dim]
    cos, sin = rotary_emb(ref, position_ids)         # [B, C, head_dim]

    cache = DynamicCache()
    for layer_idx in range(L):
        k = per_layer_kv[layer_idx, 0]              # [B, kv_heads, C, head_dim]
        v = per_layer_kv[layer_idx, 1]              # [B, kv_heads, C, head_dim]
        # Match the real attention path on architectures with a key norm (Qwen3):
        # k_norm is an RMSNorm over head_dim, so it applies directly to [.., C, head_dim].
        if decoder_layers is not None:
            k_norm = getattr(decoder_layers[layer_idx].self_attn, "k_norm", None)
            if k_norm is not None:
                k = k_norm(k.to(k_norm.weight.dtype)).to(per_layer_kv.dtype)
        # apply_rotary_pos_emb(q, k, cos, sin) -> (q_rot, k_rot).
        # We pass k as both arguments and use only the second output.
        _, k_rot = apply_rotary_pos_emb(k, k, cos, sin)
        cache.update(k_rot.contiguous(), v.contiguous(), layer_idx)
    return cache
