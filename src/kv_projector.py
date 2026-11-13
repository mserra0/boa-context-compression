"""Projects raw strided Mamba hidden states into per-layer K/V [L, 2, B, kv_heads, C, head_dim]."""
import torch
import torch.nn as nn


class KVProjector(nn.Module):
    """Projects Mamba hidden [B, C, H_m] to per-layer K/V [L, 2, B, kv_heads, C, head_dim]."""

    def __init__(self, mamba_hidden: int, num_layers: int, num_kv_heads: int,
                 head_dim: int, init_std: float = 0.02):
        super().__init__()
        self.L = num_layers
        self.kv_heads = num_kv_heads
        self.head_dim = head_dim
        out_dim = num_layers * num_kv_heads * head_dim

        self.norm = nn.RMSNorm(mamba_hidden)
        self.k_proj = nn.Linear(mamba_hidden, out_dim)
        self.v_proj = nn.Linear(mamba_hidden, out_dim)
        for lin in (self.k_proj, self.v_proj):
            nn.init.normal_(lin.weight, std=init_std)
            nn.init.zeros_(lin.bias)
        self.gain = nn.Parameter(torch.ones(num_layers))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [B, C, H_m] -> [L, 2, B, kv_heads, C, head_dim]
        B, C, _ = hidden.shape
        h = self.norm(hidden)
        k = self.k_proj(h).view(B, C, self.L, self.kv_heads, self.head_dim)
        v = self.v_proj(h).view(B, C, self.L, self.kv_heads, self.head_dim)
        x = torch.stack([k, v], dim=2)  # [B, C, 2, L, kvh, hd]
        x = x.permute(3, 2, 0, 4, 1, 5).contiguous()
        return x * self.gain.view(self.L, 1, 1, 1, 1, 1)

