# src/kv_projector.py
"""Projects raw strided Mamba hidden states into per-layer K/V for the frozen LLM.

Output layout [L, 2, B, kv_heads, C, head_dim]:
  index [l, 0] = pre-RoPE keys of layer l,  [l, 1] = values of layer l.
RoPE is applied later in src/utils/kv_cache.build_kv_cache (keys only).

Training-stability design (see docs/architecture/kv-injection.md §10 on the `main` branch):
  - RMSNorm on the Mamba hidden before projection decouples K/V scale from the
    backbone's output magnitude.
  - Decoupled k_proj / v_proj: keys (drive attention weights) and values (carry
    content) have different statistics, so they get independent projections.
  - Small-init weights -> injected K/V start near zero -> attention starts
    near-uniform, a gentler optimisation start.
  - Learnable per-layer gain lets the optimiser attenuate or ramp each layer's
    injection independently.
"""
import torch
import torch.nn as nn


class KVProjector(nn.Module):
    """Projects Mamba hidden [B, C, H_m] -> per-layer K/V [L, 2, B, kv_heads, C, head_dim].

    [l, 0] = pre-RoPE keys of layer l, [l, 1] = values of layer l.

    Args:
        mamba_hidden: hidden dim of the Mamba backbone (H_m).
        num_layers:   number of frozen LLM layers (L).
        num_kv_heads: K/V head count of the frozen LLM.
        head_dim:     per-head dimension of the frozen LLM.
        init_std:     std of the small normal init for the projection weights.
    """

    def __init__(self, mamba_hidden: int, num_layers: int, num_kv_heads: int,
                 head_dim: int, init_std: float = 0.02):
        super().__init__()
        self.L = num_layers
        self.kv_heads = num_kv_heads
        self.head_dim = head_dim
        out_dim = num_layers * num_kv_heads * head_dim   # one of K or V, across all layers

        # Normalise the Mamba hidden before projecting -> output scale is independent
        # of the backbone's (drifting) activation magnitude.
        self.norm = nn.RMSNorm(mamba_hidden)
        # Decoupled key / value projections. Each maps to all layers' heads at once;
        # layer is the slower-varying axis within the output (kv_heads, head_dim fastest).
        self.k_proj = nn.Linear(mamba_hidden, out_dim)
        self.v_proj = nn.Linear(mamba_hidden, out_dim)
        # Small-init so injected K/V start near zero -> attention starts near-uniform.
        for lin in (self.k_proj, self.v_proj):
            nn.init.normal_(lin.weight, std=init_std)
            nn.init.zeros_(lin.bias)
        # Learnable per-layer gain. Init 1.0 (neutral) so the start is unchanged.
        self.gain = nn.Parameter(torch.ones(num_layers))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, C, H_m] -> [L, 2, B, kv_heads, C, head_dim]
        B, C, _ = hidden.shape
        h = self.norm(hidden)
        k = self.k_proj(h).view(B, C, self.L, self.kv_heads, self.head_dim)
        v = self.v_proj(h).view(B, C, self.L, self.kv_heads, self.head_dim)
        # Stack so (K, V) is the outermost channel axis: index 0 = K, 1 = V.
        x = torch.stack([k, v], dim=2)                       # [B, C, 2, L, kvh, hd]
        # permute: (B, C, 2, L, kvh, hd) -> (L, 2, B, kvh, C, hd)
        x = x.permute(3, 2, 0, 4, 1, 5).contiguous()
        # Per-layer gain broadcasts over (L, 1, 1, 1, 1, 1).
        return x * self.gain.view(self.L, 1, 1, 1, 1, 1)
