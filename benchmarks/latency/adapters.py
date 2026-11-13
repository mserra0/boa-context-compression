"""Method adapters for the latency benchmark.

Each adapter is three plain functions sharing one protocol:

    prepare(model, ctx, seq_len) -> state          # build inputs at exact length
    compress(model, state)       -> gen_state      # timed stage 1 (encoder)
    generate(model, gen_state, max_new_tokens) -> int   # timed stage 2, returns #new tokens

The runner times `compress` and `generate` separately so the paper figure can
show WHERE latency is spent. `full_ctx` has no real compression stage: its cost
lands entirely in the prefill inside `generate`, which is exactly the point.

Inputs are synthetic random token ids at the requested length. Token identity is
irrelevant to latency, so this keeps the sweep reproducible and dataset-free
while giving exact control over the sequence length axis.

Adding a new method = write three functions and register them in ADAPTERS. The
runner never changes.

Each entry also names a `loader`: methods that need a different model than ours
(Activation Beacon ships its own backbone) declare their own, and the runner
keeps only one loader's model resident at a time so peak-memory numbers are not
polluted by a competing method's weights.
"""

import math

import torch


def _rand_ids(vocab_size: int, length: int, device) -> torch.Tensor:
    return torch.randint(0, vocab_size, (1, length), device=device, dtype=torch.long)


# --------------------------------------------------------------------------- #
# Our method: linear-time SSM compressor -> soft prompts -> frozen LLM         #
# --------------------------------------------------------------------------- #
def ssm_prepare(model, ctx, seq_len):
    return {"input_ids": _rand_ids(ctx["compressor_vocab"], seq_len, ctx["device"])}


def ssm_compress(model, state):
    # Full-length input, attention_mask all ones -> no pad slots, so the
    # pad-soft-prompt replacement used at inference is a no-op and is skipped.
    soft_prompts, compressed_mask = model.compressor.compress_single(state["input_ids"])
    return {"soft_prompts": soft_prompts, "compressed_mask": compressed_mask}


def ssm_generate(model, gen_state, max_new_tokens):
    # eos_token_id=[-1] can never match a real argmax, so decoding always runs the
    # full max_new_tokens -- random inputs must not early-stop and corrupt timing.
    out = model.generate_compressed(
        gen_state["soft_prompts"],
        compressed_mask=gen_state["compressed_mask"],
        max_new_tokens=max_new_tokens,
        eos_token_id=[-1],
    )
    return int(out.shape[1])


# --------------------------------------------------------------------------- #
# Our method, KV path: SSM -> per-layer K/V -> injected KV cache -> frozen LLM  #
# (prompt_mode='kv' checkpoints). generate_compressed_kv fuses the SSM forward, #
# the KV-cache build and decoding, so — like full_ctx — there is no separable   #
# compress stage: the SSM + cache-build cost lands in the first-token (ttft)    #
# timing, which is exactly the time-to-first-token.                          #
# --------------------------------------------------------------------------- #
def ssm_kv_prepare(model, ctx, seq_len):
    device = ctx["device"]
    compressor_inputs = {
        "input_ids": _rand_ids(ctx["compressor_vocab"], seq_len, device),
        "seq_idx": torch.zeros((1, seq_len), device=device, dtype=torch.int32),
        "lengths": torch.tensor([seq_len], device=device, dtype=torch.long),
        "context_start": torch.tensor([0], device=device, dtype=torch.long),
        # B-ctx (compress_question=false): the question is a literal decode-side
        # scaffold token block, not compressed. Length is irrelevant to latency.
        "question_ids": _rand_ids(ctx["llm_vocab"], ctx.get("question_len", 32), device),
    }
    return {"compressor_inputs": compressor_inputs, "stride": model.compressor.stride}


def ssm_kv_compress(model, state):
    return state  # SSM forward + KV-cache build are folded into generate() / ttft


def ssm_kv_generate(model, gen_state, max_new_tokens):
    # eos_token_id=[-1] forces full-length decode (see ssm_generate).
    out = model.generate_compressed_kv(
        gen_state["compressor_inputs"], max_new_tokens=max_new_tokens, eos_token_id=[-1]
    )
    return int(out.shape[1])


# --------------------------------------------------------------------------- #
# Baseline: full context fed straight to the frozen LLM (no compression)       #
# --------------------------------------------------------------------------- #
def full_prepare(model, ctx, seq_len):
    ids = _rand_ids(ctx["llm_vocab"], seq_len, ctx["device"])
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "pad_token_id": ctx["pad_token_id"],
    }


def full_compress(model, state):
    return state  # no-op; the quadratic prefill is paid inside generate()


def full_generate(model, gen_state, max_new_tokens):
    # min_new_tokens == max_new_tokens suppresses EOS so decoding always runs the
    # full length -- random inputs must not early-stop and corrupt timing.
    out = model.llm.generate(
        input_ids=gen_state["input_ids"],
        attention_mask=gen_state["attention_mask"],
        max_new_tokens=max_new_tokens,
        min_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=gen_state["pad_token_id"],
    )
    return int(out.shape[1] - gen_state["input_ids"].shape[1])


# --------------------------------------------------------------------------- #
# SOTA baseline: Activation Beacon (official recurrent backend, Qwen3).        #
# Beacon is homogeneous -- the same Transformer both compresses and generates. #
# Its `generate` runs the windowed compression inside the first forward pass,  #
# so like ssm_kv there is no separable compress stage and the compression cost #
# lands in ttft. That is precisely the cost Beacon's own papers omit.          #
# --------------------------------------------------------------------------- #
def beacon_prepare(model, ctx, seq_len):
    # Beacon tokens occupy ids >= vocab_size, so sampled ids must stay below it.
    ids = _rand_ids(ctx["beacon_vocab"], seq_len, ctx["device"])
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "seq_len": seq_len,
        "window": int(model.config.beacon_window),
        "ratio": int(model.config.beacon_ratio[0]),
    }


def beacon_compress(model, state):
    return state  # windowed compression is folded into generate() / ttft


def beacon_generate(model, gen_state, max_new_tokens):
    # eos_token_id=[-1] forces full-length decode (see ssm_generate). Beacon's
    # generate is a hand-written greedy loop with no min_new_tokens.
    out = model.generate(
        input_ids=gen_state["input_ids"],
        attention_mask=gen_state["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=[-1],
    )
    return int(out.shape[1] - gen_state["seq_len"])


def _beacon_tokens(gen_state, seq_len):
    """Each full window of `window` tokens becomes `window // ratio` beacons; a
    trailing partial window stays raw."""
    window, ratio = gen_state["window"], gen_state["ratio"]
    return (seq_len // window) * (window // ratio) + seq_len % window


# --------------------------------------------------------------------------- #
# SOTA baseline: SAC. Homogeneous like Beacon -- one Transformer compresses and #
# another (frozen copy of the same weights) generates -- but chunked rather     #
# than recurrent: each `chunk_size` window is encoded independently under full  #
# bidirectional attention, and the K/V at `chunk_size // ratio` strided         #
# positions is kept as that chunk's memory. Compression IS separable here, so   #
# unlike ssm_kv and beacon this method reports a real `t_compress`.             #
# --------------------------------------------------------------------------- #
def sac_prepare(model, ctx, seq_len):
    device = ctx["device"]
    return {
        "inputs": {"input_ids": _rand_ids(ctx["sac_vocab"], seq_len, device)},
        # SAC keeps the question literal on the decoder side (it is the `lm_targets`
        # prompt its own evaluator passes to lm_inference), so it prefills a real
        # block -- same accounting as ssm_kv, and the same length.
        "question_ids": _rand_ids(ctx["sac_vocab"], ctx.get("question_len", 32), device),
    }


def sac_compress(model, state):
    _ids, _tok, end_idx, _hidden, memory, mem_size = model.compress(state["inputs"])
    return {"memory": memory, "end_idx": end_idx, "mem_size": mem_size,
            "question_ids": state["question_ids"]}


def sac_generate(model, gen_state, max_new_tokens):
    # Forced full-length greedy decode from the compressed memory; see
    # sac.load.greedy_generate for why the reference EOS check is dropped.
    from .sac import greedy_generate

    return greedy_generate(model, gen_state["memory"], gen_state["end_idx"], max_new_tokens,
                           question_ids=gen_state["question_ids"])


# --------------------------------------------------------------------------- #
# SOTA baseline: ICAE. A LoRA-adapted copy of the LLM reads [context || MEM]    #
# in ONE pass and hands the hidden states at the MEM positions to the frozen    #
# decoder. No chunking and no recurrence, so the encoder pays full quadratic    #
# attention over `seq_len + n_mem` -- the cost ICAE's own evaluation avoids by  #
# capping the context. `n_mem` is set per cell to hold the rate at `--icae-ratio`#
# (the reference fixes it at `max_ctx * ratio`, which would drift over a sweep).#
# Compression is separable, so ICAE also reports a real `t_compress`.           #
# --------------------------------------------------------------------------- #
def icae_prepare(model, ctx, seq_len):
    from .icae import n_mem_for

    return {
        "input_ids": _rand_ids(ctx["icae_vocab"], seq_len, ctx["device"]),
        "n_mem": n_mem_for(seq_len, ctx["icae_ratio"]),
    }


def icae_compress(model, state):
    soft_prompts, compressed_mask = model.compressor.compress(state["input_ids"], state["n_mem"])
    return {"soft_prompts": soft_prompts, "compressed_mask": compressed_mask}


def icae_generate(model, gen_state, max_new_tokens):
    # eos_token_id=-1 forces full-length decode (see ssm_generate).
    return model.generate_compressed(
        gen_state["soft_prompts"],
        compressed_mask=gen_state["compressed_mask"],
        max_new_tokens=max_new_tokens,
        eos_token_id=-1,
    )


ADAPTERS = {
    "ssm": {
        "loader": "boa",
        "prepare": ssm_prepare,
        "compress": ssm_compress,
        "generate": ssm_generate,
        "compressed_tokens": lambda gen_state, seq_len: int(gen_state["soft_prompts"].shape[1]),
    },
    "ssm_kv": {
        "loader": "boa",
        "prepare": ssm_kv_prepare,
        "compress": ssm_kv_compress,
        "generate": ssm_kv_generate,
        "compressed_tokens": lambda gen_state, seq_len: math.ceil(seq_len / gen_state["stride"]),
    },
    "full_ctx": {
        "loader": "boa",
        "prepare": full_prepare,
        "compress": full_compress,
        "generate": full_generate,
        "compressed_tokens": lambda gen_state, seq_len: seq_len,
    },
    "beacon": {
        "loader": "beacon",
        "prepare": beacon_prepare,
        "compress": beacon_compress,
        "generate": beacon_generate,
        "compressed_tokens": _beacon_tokens,
    },
    "sac": {
        "loader": "sac",
        "prepare": sac_prepare,
        "compress": sac_compress,
        "generate": sac_generate,
        # Reported by compress() itself: the encoder counts the strided positions
        # it actually kept, rather than the benchmark re-deriving the formula.
        "compressed_tokens": lambda gen_state, seq_len: int(gen_state["mem_size"]),
    },
    "icae": {
        "loader": "icae",
        "prepare": icae_prepare,
        "compress": icae_compress,
        "generate": icae_generate,
        "compressed_tokens": lambda gen_state, seq_len: int(gen_state["soft_prompts"].shape[1]),
    },
}
