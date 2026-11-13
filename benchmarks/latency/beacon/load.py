"""Load an Activation Beacon checkpoint for the latency benchmark."""

import json
from pathlib import Path

import torch

from .qwen3_official import load_trainable_qwen3_official_beacon_model


def load_beacon_model(checkpoint_dir, base_model_dir=None, beacon_ratio=8,
                      beacon_window=None, device="cuda", dtype="bf16"):
    """Build the official-backend Beacon model from a trained adapter checkpoint.

    `base_model_dir` overrides where the frozen backbone weights come from.
    Activation Beacon never updates the backbone -- only the per-layer
    `beacon_{q,k,v}_proj` tensors and the beacon embedding train -- so any copy
    of the same base is equivalent. The override exists because a checkpoint's
    own `model.safetensors` may not be readable by the benchmark's user.

    `beacon_ratio` must be a single value. Checkpoints are trained with a list
    (e.g. [2, 4, 8, 16]) sampled per step, which would make both the compression
    rate and the latency nondeterministic across repeats.

    `beacon_window` overrides the compression window the checkpoint was trained
    at. This is legitimate *for latency* and not for anything else. The window is
    a pure runtime knob in the sliding-window loop -- not one trainable tensor's
    shape depends on it, since the beacon parameters are `nn.Embedding(1, hidden)`
    plus per-layer clones of the backbone's own q/k/v projections. So a checkpoint
    run at a window it was not trained at costs exactly what a checkpoint trained
    at that window would cost, while producing text that is worthless. That is a
    fair trade here and only here: the sweep feeds synthetic random token ids and
    forces full-length decode, so output quality is already not being measured.
    The same argument lets SAC and ICAE be swept with no checkpoint at all.
    """
    checkpoint_dir = Path(checkpoint_dir)
    metadata_path = checkpoint_dir / "activation_beacon_backend.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}

    settings = {
        "base_model_name_or_path": str(checkpoint_dir),
        "beacon_code_model": str(base_model_dir or checkpoint_dir),
        "dtype": dtype,
        "attn_implementation": "sdpa",
        "local_files_only": True,
        "beacon_ratio": [int(beacon_ratio)],
        "beacon_ratio_mix": "step-random",  # single-ratio lists short-circuit the mix
    }
    if beacon_window is not None:
        # Stride must move with the window: `beacon_pos='interleave'` (what every
        # checkpoint here uses) rejects window != stride, and the merge would
        # otherwise leave stride at the checkpoint's trained value. A settings key
        # only loses to the checkpoint metadata when it is None, so both win here.
        settings["beacon_window"] = int(beacon_window)
        settings["beacon_stride"] = int(beacon_window)

    model, tokenizer, _ = load_trainable_qwen3_official_beacon_model(
        {"model": settings}, backend_metadata=metadata
    )
    model.to(device).eval().requires_grad_(False)
    if dtype == "bf16":
        model.to(torch.bfloat16)
    return model, tokenizer
