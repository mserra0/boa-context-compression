"""ICAE backend for the latency benchmark.

The encoder and decoder paths are lifted from `ICAECompressor.forward` and
`ICAEDestiller.generate_compressed` on the `baseline/icae` branch, trimmed to the
inference path. They are copied rather than imported because that branch's
`src/model.py` also carries the Mamba compressors, the KD losses and the
training wrappers, none of which the benchmark needs and all of which would
have to be kept in sync with this branch's own `src/`.

Two deviations from the branch, both required here:

  * **LoRA comes from `benchmarks/latency/lora.py`, not PEFT.** PEFT is not a
    dependency of this project. The replacement is the same rank-128 adapter on
    `q_proj`/`v_proj` with `alpha/r = 2`, and the `enabled` flag reproduces
    `enable_adapter_layers()` / `disable_adapter_layers()` -- ICAE shares one
    base model between the LoRA-adapted encoder and the frozen decoder.

  * **`n_mem` is derived per sequence length instead of being fixed.** The
    branch computes `n_mem = round(max_ctx * compression_ratio)` once at
    startup, so a checkpoint trained at `max_ctx=1024` always emits 128 slots
    whatever the input length -- the compression rate would drift across the
    sweep. Here `n_mem = ceil(seq_len / ratio)`, which holds the rate at exactly
    8x in every cell. Slot embeddings are learnable parameters of that size, so
    the count is the only thing latency depends on.
"""

import math

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from ..lora import inject_lora, set_lora_enabled


class ICAECompressor(nn.Module):
    """Appends `n_mem` learnable embeddings to the context, runs the concatenation
    through the LoRA-adapted LM, and returns the hidden states at those positions.

    Because the [MEM] tokens sit at the end, standard causal attention already
    gives each MEM_i full visibility into the context and the preceding MEM
    tokens -- no special masking on the encoder side.
    """

    def __init__(self, base_lm, llm_hidden_dim: int, device: torch.device):
        super().__init__()
        object.__setattr__(self, "base_lm", base_lm)  # shared with the decoder; register once
        self.llm_hidden_dim = llm_hidden_dim
        self.device = device
        # One parameter bank per distinct n_mem in the sweep. Built on demand and
        # kept, so a re-timed cell reuses the same tensor and no allocation noise
        # lands inside the measurement window.
        self._mem_banks: dict[int, nn.Parameter] = {}

    def mem_embeddings(self, n_mem: int) -> nn.Parameter:
        if n_mem not in self._mem_banks:
            dtype = self.base_lm.dtype
            self._mem_banks[n_mem] = nn.Parameter(
                torch.randn(n_mem, self.llm_hidden_dim, device=self.device, dtype=dtype) * 0.02,
                requires_grad=False,
            )
        return self._mem_banks[n_mem]

    @torch.no_grad()
    def compress(self, input_ids: torch.Tensor, n_mem: int):
        """`input_ids` [B, I] -> `(soft_prompts [B, n_mem, H], mask [B, n_mem])`."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, I = input_ids.shape

        context_embeds = self.base_lm.get_input_embeddings()(input_ids)      # [B, I, H]
        mem_embeds = self.mem_embeddings(n_mem).unsqueeze(0).expand(B, -1, -1)
        full_embeds = torch.cat([context_embeds, mem_embeds], dim=1)         # [B, I+n_mem, H]
        full_mask = torch.ones((B, I + n_mem), dtype=torch.long, device=self.device)

        # The encoder is the ONLY place LoRA is active: enable it for this call,
        # then restore the disabled default so the shared model behaves as the
        # plain frozen decoder everywhere else.
        set_lora_enabled(self.base_lm, True)
        try:
            # Backbone-only call: this path never uses the vocab projection, and
            # skipping lm_head over I+n_mem positions is what the branch does too.
            backbone_out = self.base_lm.model(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                return_dict=True,
            )
        finally:
            set_lora_enabled(self.base_lm, False)

        soft_prompts = backbone_out.last_hidden_state[:, I:, :]              # [B, n_mem, H]
        compressed_mask = torch.ones(B, n_mem, dtype=torch.long, device=self.device)
        return soft_prompts, compressed_mask


class ICAEModel(nn.Module):
    """LoRA-adapted encoder + frozen decoder, sharing one set of base weights."""

    def __init__(self, model_id: str, device: torch.device, lora_r: int = 128,
                 dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.device = device
        base_lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        if base_lm.config.pad_token_id is None:
            base_lm.config.pad_token_id = base_lm.config.eos_token_id
        # Encoder and decoder are the SAME underlying weights (LoRA-adapted vs.
        # not) rather than two full copies, which halves the VRAM footprint.
        inject_lora(base_lm, r=lora_r)
        set_lora_enabled(base_lm, False)   # decoder role is the ambient default
        self.llm = base_lm
        self.compressor = ICAECompressor(base_lm, base_lm.config.hidden_size, device)

    @torch.no_grad()
    def generate_compressed(self, soft_prompts: torch.Tensor, compressed_mask: torch.Tensor = None,
                            max_new_tokens: int = 64, eos_token_id: int = None) -> int:
        """Greedy decode from the memory slots. Returns the number of new tokens.

        `eos_token_id=-1` can never match an argmax, so decoding always runs the
        full length -- random benchmark inputs must not early-stop and corrupt the
        decode timing (same rule as every other adapter).
        """
        B, C, _ = soft_prompts.shape
        device = soft_prompts.device
        if compressed_mask is None:
            compressed_mask = torch.ones((B, C), dtype=torch.long, device=device)

        # Memory slots attend to each other bidirectionally; padded slots are masked.
        prefix_allowed = compressed_mask[:, None, None, :] > 0.5
        prefix_allowed = prefix_allowed.expand(B, 1, C, C)
        mask_dtype = soft_prompts.dtype
        prefix_mask_4d = torch.zeros((B, 1, C, C), dtype=mask_dtype, device=device)
        prefix_mask_4d.masked_fill_(~prefix_allowed, torch.finfo(mask_dtype).min)

        position_ids = compressed_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(compressed_mask == 0, 1)

        prefix_out = self.llm.model(
            inputs_embeds=soft_prompts,
            attention_mask=prefix_mask_4d,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        past_kv = prefix_out.past_key_values
        first_logits = self.llm.lm_head(prefix_out.last_hidden_state[:, -1:, :])
        cur_id = first_logits.argmax(dim=-1)
        generated = 1
        real_prefix_len = compressed_mask.sum(dim=1).long()

        for k in range(max_new_tokens - 1):
            pos_ids = (real_prefix_len + k).unsqueeze(1)
            out = self.llm.model(
                input_ids=cur_id,
                past_key_values=past_kv,
                position_ids=pos_ids,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            cur_id = self.llm.lm_head(out.last_hidden_state[:, -1:, :]).argmax(dim=-1)
            generated += 1
            if B == 1 and eos_token_id is not None and cur_id.item() == eos_token_id:
                break

        return generated


def n_mem_for(seq_len: int, ratio: int) -> int:
    """Slot count that holds the compression rate at `ratio` for this length."""
    return max(1, math.ceil(seq_len / ratio))
