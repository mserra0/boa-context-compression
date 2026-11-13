"""Load a SAC checkpoint and drive its decode loop for the latency benchmark."""

import json
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

from .modeling import get_model_for_compress, load_adapter

# Keys the benchmark reads out of a checkpoint's config.json, in preference
# order. The SFT stage is the one the eval harness measures, and its task config
# is what a deployed SAC would run.
_TASK_CONFIG_KEYS = ("sft_task_config", "pretrain_task_config")


def _resolve_checkpoint(checkpoint_dir):
    """Return `(config.json path, adapter path or None)` for a SAC run directory.

    Runs are laid out as `<run>/output/{config.json,adapter.pt,instruction_adapter.pt}`;
    the run root is accepted too and redirected into `output/` when present.
    `instruction_adapter.pt` is the SFT stage and wins over the pretrain `adapter.pt`.
    """
    root = Path(checkpoint_dir)
    if (root / "output" / "config.json").is_file():
        root = root / "output"
    config_path = root / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"No config.json under {checkpoint_dir} (looked in ./ and ./output/).")
    for name in ("instruction_adapter.pt", "adapter.pt"):
        if (root / name).is_file():
            return config_path, root / name
    return config_path, None


def load_sac_model(checkpoint_dir=None, base_model_dir=None, compress_ratio=8,
                   chunk_size=512, device="cuda", dtype="bf16", load_weights=True):
    """Build SAC's `CompressLLM` (LoRA encoder + frozen decoder) for timing.

    Without `checkpoint_dir` the architecture is built from the CLI arguments and
    left at its random initialisation. That is deliberate: the trained tensors are
    the [MEM] role embeddings, the two special tokens and the LoRA factors, none
    of which change the shape of any matmul. Latency is identical either way, and
    the benchmark only claims processing time.

    `compress_ratio` and `chunk_size` fix the compression rate: each `chunk_size`
    window contributes `chunk_size // compress_ratio` memory slots, so ratio 8 with
    the trained chunk size of 512 gives 64 slots per chunk at every sequence length.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise SystemExit("The sac adapter requires a CUDA device (its backend pins device_map).")
    rank = device.index if device.index is not None else 0

    task_config = None
    adapter_path = None
    model_id = base_model_dir
    if checkpoint_dir:
        config_path, adapter_path = _resolve_checkpoint(checkpoint_dir)
        config = json.loads(config_path.read_text())
        for key in _TASK_CONFIG_KEYS:
            if key in config:
                task_config = dict(config[key])
                break
        if task_config is None:
            raise SystemExit(f"{config_path} has none of {_TASK_CONFIG_KEYS}.")
        training = config.get("sft_training_config") or config.get("pretrain_training_config") or {}
        model_id = base_model_dir or training.get("model_id") or task_config.get("model_id")

    if task_config is None:
        task_config = {"task_type": "Compress", "use_pe": True,
                       "is_pretrain": False, "is_sft": True,
                       "use_ae_loss": False, "use_lm_loss": True}
    if not model_id:
        raise SystemExit("--sac-base is required when --sac-checkpoint is not given.")

    # CLI wins over the checkpoint so a single checkpoint can be swept at several
    # ratios, and so the requested ratio is always the one that gets measured.
    task_config["compress_ratio"] = int(compress_ratio)
    task_config["chunk_size"] = int(chunk_size)
    task_config["mem_size"] = int(chunk_size) // int(compress_ratio)
    task_config["task_type"] = task_config.get("task_type", "Compress")

    model = get_model_for_compress(model_id, task_config, rank)
    if adapter_path is not None and load_weights:
        load_adapter(model, str(adapter_path))
    # The reference `compress()` also builds `compress_token` and
    # `encoder_hidden_states` for the autoencoding objective. The QA/LM path timed
    # here never reads either -- `compress()` literally returns `None` in that slot
    # -- but producing them forces `output_hidden_states=True` on every chunk and
    # costs ~1 GB of peak memory at 32k, growing linearly. Off here so the
    # measurement reflects the method rather than a leftover of its training code;
    # the returned KV memory is bit-identical either way.
    model.collect_hidden_states = False
    model.eval().requires_grad_(False)
    if dtype == "fp32":
        model.to(torch.float32)
    else:
        model.to(torch.bfloat16)
    return model, task_config, adapter_path


def greedy_generate(model, memory, end_idx, max_new_tokens, question_ids=None):
    """SAC's decode loop, forced to run the full `max_new_tokens`.

    Mirrors `CompressLLM.lm_inference`, which is what SAC's own evaluator calls:
    the decoder prefills `[memory | LM | question]` and continues from there.
    `question_ids` is that prompt block -- SAC keeps the question *literal* on the
    decoder side rather than compressing it, so leaving it out would under-report
    its time-to-first-token (and `ssm_kv`, the other method with a literal
    question, already carries one).

    The one deliberate departure is the EOS check, dropped so random inputs cannot
    early-stop and corrupt the decode timing -- the same rigor the other adapters
    get from `min_new_tokens` / `eos_token_id=[-1]`.

    `memory` is the per-layer trimmed K/V from `compress()`. A fresh `Cache` is
    built per call, so repeated timed runs all start from the same memory.
    """
    decoder = model.decoder
    use_pe = model.task_config["use_pe"]
    lm_token = model.special_tokens[1:2].unsqueeze(0)          # [1, 1, E]
    bsz = memory[0][0].shape[0]
    next_embeds = lm_token.expand(bsz, 1, lm_token.shape[-1])
    if question_ids is not None:
        question_embeds = decoder.model.embed_tokens(question_ids).to(next_embeds.dtype)
        next_embeds = torch.cat([next_embeds, question_embeds.expand(bsz, -1, -1)], dim=1)

    past_key_values = model._decoder_cache(memory)
    # The [LM] token sits at the last *uncompressed* context position and the
    # question follows it; the answer continues from there. Memory slots keep the
    # strided position ids they were encoded with. Same arithmetic as
    # lm_inference's `arange(end_idx, end_idx + seq_len + 1)`.
    next_position_ids = torch.arange(
        end_idx, end_idx + next_embeds.shape[1], device=next_embeds.device, dtype=torch.long
    ).unsqueeze(0).expand(bsz, -1)

    generated = 0
    for _ in range(max_new_tokens):
        if use_pe:
            out = decoder(inputs_embeds=next_embeds, position_ids=next_position_ids,
                          past_key_values=past_key_values, use_cache=True)
        else:
            out = decoder(inputs_embeds=next_embeds,
                          past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        next_token_id = torch.argmax(out.logits[:, -1], dim=-1)
        next_embeds = decoder.model.embed_tokens(next_token_id).unsqueeze(1)
        next_position_ids = next_position_ids[:, -1:] + 1
        generated += 1
    return generated


__all__ = ["load_sac_model", "greedy_generate", "DynamicCache"]
