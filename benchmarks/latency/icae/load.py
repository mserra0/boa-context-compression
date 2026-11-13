"""Build an ICAE model for the latency benchmark."""

import torch

from .modeling import ICAEModel


def load_icae_model(base_model_dir, checkpoint=None, lora_r=128, device="cuda", dtype="bf16"):
    """Build ICAE's shared-weight encoder/decoder pair.

    `checkpoint` is optional and off the benchmark's critical path. The trained
    tensors are the [MEM] slot embeddings and the LoRA factors; neither changes
    the shape of any matmul, so latency is identical to the randomly initialised
    model. A checkpoint is also *length-bound* -- it carries exactly the
    `n_mem = round(max_ctx * compression_ratio)` slots it was trained with (128
    for the released 8x runs), which cannot hold an 8x rate across a sweep that
    spans 512 to 256k tokens. Loading one therefore only ever restores the subset
    of banks whose size happens to match.
    """
    torch_dtype = torch.float32 if dtype == "fp32" else torch.bfloat16
    model = ICAEModel(base_model_dir, torch.device(device), lora_r=lora_r, dtype=torch_dtype)
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        state = state.get("model_state_dict", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"ICAE checkpoint loaded: {len(missing)} missing, {len(unexpected)} unexpected keys.")
    model.eval().requires_grad_(False)
    return model
