import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .losses import lm_ce_loss
from .compressors import MambaCompressor, Mamba2Compressor
from .kv_projector import KVProjector
from .utils.attention import build_causal_mask, build_additive_mask
from .utils.kv_cache import build_kv_cache
from .utils.memory_profile import StageMemoryProfiler

"""notation: B batch, I input len, C compressed len, G continuation len,
   H_llm LLM hidden, V vocab, S stride."""


class LMPretrainer(nn.Module):
    """Continuation-LM pretrainer: compress prefix X → frozen decoder predicts suffix Y.

    Trains only the compressor injection params. The decoder never sees X in plaintext,
    so the compressed representation must carry the information. No teacher pass, no KD.

    Two injection modes, selected by `prompt_mode` (mirrors PromptDestiller):
      - 'embed': compressed hidden -> soft-prompt embeds prepended to Y in the sequence.
      - 'kv':    compressed hidden -> per-layer K/V seeded into the decoder's cache; a
                 single learned trigger token then queries that cache to start the
                 continuation (SAC-style: a trainable [1,1,H] vector, warm-started from
                 the BOS embedding).
    Both score the same G continuation tokens, so the two modes stay comparable.

    Wrapped chat templates are intentionally NOT supported: continuation has no question
    and Y is raw document text, so chat framing would be a distribution mismatch. The KV
    trigger is a free learned vector, not a chat anchor seeded from assistant-turn tokens.
    """

    def __init__(self, mamba_model_name: str, llm_model_name: str, device: torch.device,
                 stride: int = 2, compressor_type: str = "mamba2",
                 prompt_mode: str = "embed", memory_profile: bool = False,
                 compressor_gradient_checkpointing: bool = True,
                 llm_dtype: torch.dtype = torch.bfloat16,
                 attn_implementation: str = "sdpa"):
        super().__init__()
        self.device = device
        self.compressor_type = compressor_type
        self.prompt_mode = prompt_mode
        # Opt-in CUDA memory stage profiler (see src/utils/memory_profile.py). No-op
        # unless memory_profile=True; `pretrain.py` calls `self.mem.reset()` once per
        # training step, stage() calls below print incremental deltas.
        self.mem = StageMemoryProfiler(enabled=memory_profile, device=device)

        # bf16 halves weight/activation memory for the (much larger) frozen LLM and the
        # Mamba2 backbone alike -- both are read-heavy, gradient-free-on-the-LLM-side
        # workloads that tolerate bf16 well. attn_implementation is set explicitly
        # (rather than left to whatever the environment defaults to) so we don't
        # silently fall back to the memory-hungry "eager" full-attention-matrix path on
        # a cluster that may lack a compiled flash-attention wheel for its GPU/CUDA combo.
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, dtype=llm_dtype, attn_implementation=attn_implementation,
        ).to(device)
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()
        if self.llm.config.pad_token_id is None:
            self.llm.config.pad_token_id = self.llm.config.eos_token_id

        if compressor_type == "mamba2":
            self.compressor = Mamba2Compressor(
                mamba_model_name, self.llm.config.hidden_size, device,
                stride, dtype=next(self.llm.parameters()).dtype,
                gradient_checkpointing=compressor_gradient_checkpointing,
            )
            self.compressor.mem_stage_fn = self.mem.stage
        elif compressor_type == "mamba":
            self.compressor = MambaCompressor(
                mamba_model_name, self.llm.config.hidden_size, device, stride,
            )
            self.compressor.mamba.config.padding_side = "left"
        else:
            raise ValueError(f"Unknown compressor_type {compressor_type!r}. Use 'mamba' or 'mamba2'.")

        self.compressor.compile()

        self.kv_projector = None
        if prompt_mode == "kv":
            c = self.llm.config
            head_dim = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
            self.kv_projector = KVProjector(
                self.compressor.mamba_hidden_size,   # backend-agnostic (mamba1 hidden / mamba2 d_model)
                c.num_hidden_layers, c.num_key_value_heads, head_dim,
            ).to(device=device, dtype=next(self.llm.parameters()).dtype)
            # Continuation decodes from a learned trigger token (SAC-style), NOT a frozen
            # BOS: the decoder learns the optimal "begin continuation" query into the KV
            # cache. Warm-started from the BOS (<|endoftext|>, 151643) embedding so it
            # begins identical to frozen-BOS behaviour, then trains.
            bos_id = c.bos_token_id if c.bos_token_id is not None else c.eos_token_id
            with torch.no_grad():
                bos_embed = self.llm.get_input_embeddings()(
                    torch.tensor([[bos_id]], device=device)).float()  # [1,1,H]
            self.kv_trigger = nn.Parameter(bos_embed.squeeze(0))       # [1,H], trainable
            # KV mode taps the raw hidden via gather_hidden, so the embedding-space
            # projector is unused; freeze it so DDP doesn't flag an unused parameter.
            self.compressor.projector.requires_grad_(False)
        elif prompt_mode != "embed":
            raise ValueError(f"Unknown prompt_mode {prompt_mode!r}. Use 'embed' or 'kv'.")

    def train(self, mode: bool = True):
        # The frozen LLM must stay in eval mode regardless of the module's train
        # state: nn.Module.train() propagates to all children, so a plain
        # model.train() after evaluation would silently re-enable dropout on the
        # LLM. Pin it back to eval after delegating to the default behaviour.
        super().train(mode)
        self.llm.eval()
        return self

    def forward(self, target_inputs: dict, compressor_inputs: dict, return_logits: bool = False):
        if self.prompt_mode == "kv":
            return self._kv_forward(target_inputs, compressor_inputs, return_logits)
        return self._embed_forward(target_inputs, compressor_inputs, return_logits)

    def _embed_forward(self, target_inputs: dict, compressor_inputs: dict, return_logits: bool):
        target_ids = target_inputs["target_ids"]     # [B, G]
        target_mask = target_inputs["target_mask"]   # [B, G]

        soft_prompts, compressed_mask = self.compressor(compressor_inputs)  # [B,C,H], [B,C]
        self.mem.stage("embed.post_compressor")

        with torch.no_grad():
            y_embeds = self.llm.get_input_embeddings()(target_ids)          # [B, G, H]
        soft_prompts = soft_prompts.to(y_embeds.dtype)

        combined = torch.cat([soft_prompts, y_embeds], dim=1)               # [B, C+G, H]
        full_padding_mask = torch.cat([compressed_mask, target_mask], dim=1)  # [B, C+G]
        prefix_len = soft_prompts.shape[1]   # C
        gen_len = target_ids.shape[1]        # G

        # Cumsum position_ids handle left-padded soft prompts AND right-padded Y; pad
        # positions are parked at 1 (they are masked out of attention anyway).
        position_ids = full_padding_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(full_padding_mask == 0, 1)

        geometry = build_causal_mask(prefix_len, gen_len, self.device)
        attention_mask = build_additive_mask(geometry, full_padding_mask, combined.dtype)

        out = self.llm.model(
            inputs_embeds=combined, attention_mask=attention_mask,
            position_ids=position_ids, return_dict=True,
        )
        self.mem.stage("embed.post_llm_forward")
        # Shift by −1: last soft prompt (index C−1) predicts Y_0; Y_{i−1} predicts Y_i.
        student_logits = self.llm.lm_head(out.last_hidden_state)[:, prefix_len - 1:-1, :]  # [B,G,V]

        loss = lm_ce_loss(student_logits, target_ids, target_mask)
        if return_logits:
            return loss, student_logits
        return loss

    @torch.no_grad()
    def generate_compressed_kv(self, compressor_inputs: dict, max_new_tokens: int = 128,
                               eos_token_id=None) -> torch.Tensor:
        """Autoregressive generation for prompt_mode='kv' (mirrors
        PromptDestiller.generate_compressed_kv's bare-template path, but decodes from
        this model's own single-token `kv_trigger` instead of a multi-token chat
        anchor -- there is no chat template or question/context split here, matching
        the raw-continuation training distribution). Seeds the KV cache from the
        compressed context, then greedily decodes until eos or max_new_tokens."""
        if self.prompt_mode != "kv":
            raise RuntimeError("generate_compressed_kv requires prompt_mode='kv'.")
        if eos_token_id is None:
            cfg_eos = self.llm.config.eos_token_id
            eos_ids = list(cfg_eos) if isinstance(cfg_eos, (list, tuple)) else (
                [cfg_eos] if cfg_eos is not None else [])
        else:
            eos_ids = list(eos_token_id) if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]

        hidden, compressed_mask = self.compressor.gather_hidden(compressor_inputs)
        B, C, _ = hidden.shape
        device = hidden.device
        ref_dtype = next(self.llm.parameters()).dtype
        hidden = hidden.to(self.kv_projector.k_proj.weight.dtype)
        per_layer_kv = self.kv_projector(hidden).to(ref_dtype)
        comp_pos = (compressed_mask.long().cumsum(-1) - 1).clamp_min(0)
        cache = build_kv_cache(per_layer_kv, comp_pos, self.llm.model.rotary_emb,
                               decoder_layers=self.llm.model.layers)

        C_real = compressed_mask.sum(dim=1).long()              # [B]
        pad_id = self.llm.config.pad_token_id
        eos_tensor = torch.tensor(eos_ids, device=device) if eos_ids else None

        cur_emb = self.kv_trigger.to(ref_dtype).unsqueeze(0).expand(B, 1, -1)   # [B,1,H]
        cur_mask = compressed_mask.new_ones((B, 1))
        seen_mask = compressed_mask.new_zeros((B, 0))
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generated = []
        for _ in range(max_new_tokens):
            n = cur_emb.shape[1]                                    # 1 every step (single trigger token)
            full_dec_mask = torch.cat([seen_mask, cur_mask], dim=1)
            rel = (full_dec_mask.cumsum(-1) - 1).clamp_min(0)
            pos = (C_real.unsqueeze(1) + rel)[:, -n:]               # [B, n]
            attn = torch.cat([compressed_mask, seen_mask, cur_mask], dim=1)
            out = self.llm.model(inputs_embeds=cur_emb, past_key_values=cache,
                                 attention_mask=attn, position_ids=pos,
                                 use_cache=True, return_dict=True)
            cache = out.past_key_values
            seen_mask = full_dec_mask
            logits = self.llm.lm_head(out.last_hidden_state[:, -1:, :])
            cur_id = logits.argmax(-1)                              # [B, 1]
            if eos_tensor is not None:
                cur_id = torch.where(finished.unsqueeze(1),
                                     torch.full_like(cur_id, pad_id), cur_id)
                finished |= (cur_id == eos_tensor).any(-1)
            generated.append(cur_id)
            if eos_tensor is not None and bool(finished.all()):
                break
            cur_emb = self.llm.get_input_embeddings()(cur_id).to(ref_dtype)
            cur_mask = compressed_mask.new_ones((B, 1))
        return torch.cat(generated, dim=1)

    def _kv_forward(self, target_inputs: dict, compressor_inputs: dict, return_logits: bool):
        target_ids = target_inputs["target_ids"]     # [B, G]
        target_mask = target_inputs["target_mask"]   # [B, G]

        # 1. Raw strided Mamba hidden -> per-layer pre-RoPE K/V seeded into the cache.
        hidden, compressed_mask = self.compressor.gather_hidden(compressor_inputs)
        self.mem.stage("kv.post_gather_hidden")
        B, C, _ = hidden.shape
        hidden = hidden.to(self.kv_projector.k_proj.weight.dtype)
        per_layer_kv = self.kv_projector(hidden)                 # [L,2,B,kvh,C,hd]
        ref_dtype = next(self.llm.parameters()).dtype
        per_layer_kv = per_layer_kv.to(ref_dtype)
        self.mem.stage("kv.post_kv_projector")

        # 2. Compressed positions: left-anchored 0..C_real-1 (pad slots -> 0).
        comp_pos = (compressed_mask.long().cumsum(-1) - 1).clamp_min(0)
        cache = build_kv_cache(per_layer_kv, comp_pos, self.llm.model.rotary_emb,
                               decoder_layers=self.llm.model.layers)
        self.mem.stage("kv.post_build_cache")

        # 3. Decoder input = [trigger | continuation embeds]. The learned trigger is the
        #    single query that reads the compressed cache to emit Y_0 (the KV analogue of
        #    the last soft prompt in embed mode).
        trigger = self.kv_trigger.to(ref_dtype).unsqueeze(0).expand(B, 1, -1)  # [B,1,H]
        with torch.no_grad():
            y_embeds = self.llm.get_input_embeddings()(target_ids).to(ref_dtype)      # [B,G,H]
        G = target_ids.shape[1]
        dec_input = torch.cat([trigger, y_embeds], dim=1)        # [B, 1+G, H]

        # 4. Positions continue per sample from each sample's real compressed length;
        #    right-padded Y stays pad-aware via cumsum.
        C_real = compressed_mask.sum(dim=1).long()              # [B]
        dec_mask = torch.cat([target_mask.new_ones((B, 1)), target_mask], dim=1)  # [B, 1+G]
        dec_pos = C_real.unsqueeze(1) + (dec_mask.cumsum(-1) - 1).clamp_min(0)

        # 5. Attention spans [compressed (pad-aware) | trigger | continuation (pad-aware)].
        attn_mask = torch.cat([compressed_mask, dec_mask], dim=1)

        out = self.llm.model(
            inputs_embeds=dec_input, past_key_values=cache,
            attention_mask=attn_mask, position_ids=dec_pos,
            use_cache=True, return_dict=True,
        )
        self.mem.stage("kv.post_llm_forward")
        # trigger (idx 0) predicts Y_0 ... idx G-1 predicts Y_{G-1}; the final position drops.
        student_logits = self.llm.lm_head(out.last_hidden_state[:, 0:G, :])   # [B,G,V]

        loss = lm_ce_loss(student_logits, target_ids, target_mask)
        if return_logits:
            return loss, student_logits
        return loss
