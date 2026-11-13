# Based on the LoRA layer of SAC (https://github.com/lx-Meteors/SAC), modified. See THIRD_PARTY_NOTICES.md.
"""Minimal LoRA layer shared by the vendored SAC and ICAE baselines.

`LinearLoraLayer` follows SAC's `model/lora.py` (`baseline/sac` branch) with the same layer
math (the two baselines use the same configuration -- rank 128 on `q_proj`/`v_proj`, alpha
256, so `scale = alpha / r = 2`). Differences: an `enabled` flag, a `dtype` argument
(default bfloat16, as in the original), an out-of-place residual add, and `inject_lora`.

The flag exists for ICAE, which shares ONE base model between its LoRA-adapted
encoder and its frozen decoder. The reference implementation used PEFT's
`enable_adapter_layers()` / `disable_adapter_layers()` to flip between the two
roles; PEFT is not a dependency of this project, so the toggle lives here
instead. It is a plain Python attribute, not a buffer, so `state_dict()` is
unchanged and SAC's trained `adapter.pt` still loads.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class LinearLoraLayer(nn.Module):
    # No bias in the Qwen3 attention projections this targets.
    def __init__(self, in_features, out_features, r, weight=None, dtype=torch.bfloat16):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = 2  # The alpha value is usually twice the rank
        self.enabled = True
        self.lora_A = nn.Parameter(torch.zeros((in_features, r), device=self.weight.device, dtype=dtype),
                                   requires_grad=True)
        self.lora_B = nn.Parameter(torch.zeros((r, out_features), device=self.weight.device, dtype=dtype),
                                   requires_grad=True)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = F.linear(x, self.weight)
        if self.enabled:
            result = result + self.scale * (x @ self.lora_A @ self.lora_B)
        return result


def inject_lora(model, r, target_names=("q_proj", "v_proj")):
    """Replace the named `nn.Linear` submodules with LoRA-adapted equivalents.

    Same recursive walk as SAC's `add_compress_lora`, lifted here so ICAE can
    reuse it. Returns the number of layers replaced.
    """
    replaced = 0
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_names:
            if module.bias is not None:
                raise ValueError(
                    f"LinearLoraLayer drops the bias term, but {name} has one. "
                    "This baseline was written against a bias-free attention block."
                )
            setattr(model, name, LinearLoraLayer(
                module.in_features, module.out_features, r=r,
                weight=module.weight.data.clone(), dtype=module.weight.dtype,
            ))
            replaced += 1
        else:
            replaced += inject_lora(module, r, target_names)
    return replaced


def set_lora_enabled(model, enabled: bool):
    """Flip every injected LoRA layer on or off (ICAE's encoder/decoder switch)."""
    for module in model.modules():
        if isinstance(module, LinearLoraLayer):
            module.enabled = enabled
