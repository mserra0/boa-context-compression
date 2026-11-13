import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .losses import lm_ce_loss, kd_lm_loss
from .compressors import MambaCompressor, Mamba2Compressor
from .kv_projector import KVProjector
from .utils.attention import build_causal_mask, build_additive_mask
from .utils.kv_cache import build_kv_cache

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
                 compression_ratio: float = 0.5, compressor_type: str = "mamba2",
                 prompt_mode: str = "embed"):
        super().__init__()
        self.device = device
        self.compressor_type = compressor_type
        self.prompt_mode = prompt_mode

        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()
        if self.llm.config.pad_token_id is None:
            self.llm.config.pad_token_id = self.llm.config.eos_token_id

        if compressor_type == "mamba2":
            self.compressor = Mamba2Compressor(
                mamba_model_name, self.llm.config.hidden_size, device,
                compression_ratio, dtype=next(self.llm.parameters()).dtype,
            )
        elif compressor_type == "mamba":
            self.compressor = MambaCompressor(
                mamba_model_name, self.llm.config.hidden_size, device, compression_ratio,
            )
            self.compressor.mamba.config.padding_side = "left"
        else:
            raise ValueError(f"Unknown compressor_type {compressor_type!r}. Use 'mamba' or 'mamba2'.")

        self.kv_projector = None
        if prompt_mode == "kv":
            c = self.llm.config
            head_dim = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
            self.kv_projector = KVProjector(
                self.compressor.mamba_hidden_size,   # backend-agnostic (mamba1 hidden / mamba2 d_model)
                c.num_hidden_layers, c.num_key_value_heads, head_dim,
            ).to(device)
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
        target_ids = target_inputs["target_ids"]     # [B, G]
        target_mask = target_inputs["target_mask"]   # [B, G]
        student_logits = self._student_logits(target_inputs, compressor_inputs)  # [B,G,V]
        loss = lm_ce_loss(student_logits, target_ids, target_mask)
        if return_logits:
            return loss, student_logits
        return loss

    def _student_logits(self, target_inputs: dict, compressor_inputs: dict) -> torch.Tensor:
        """Compressed-prefix student pass -> logits [B,G,V] aligned with the Y targets.

        The objective-agnostic seam: lm_ce_loss (this class) and kd_lm_loss
        (KDPretrainer) both consume these logits, so the two objectives stay
        numerically comparable on the same Y positions. Dispatches by prompt_mode.
        """
        if self.prompt_mode == "kv":
            return self._kv_student_logits(target_inputs, compressor_inputs)
        return self._embed_student_logits(target_inputs, compressor_inputs)

    def ce_readout(self, target_inputs: dict, compressor_inputs: dict) -> torch.Tensor:
        """Student cross-entropy on Y, independent of the training objective.

        Used at eval time so a KD-pretrain run reports the SAME continuation-CE /
        perplexity metric as an lm_ce_loss run (the KD training loss is a KL and is
        not comparable across runs).
        """
        student_logits = self._student_logits(target_inputs, compressor_inputs)
        return lm_ce_loss(student_logits, target_inputs["target_ids"], target_inputs["target_mask"])

    def _embed_student_logits(self, target_inputs: dict, compressor_inputs: dict) -> torch.Tensor:
        target_ids = target_inputs["target_ids"]     # [B, G]
        target_mask = target_inputs["target_mask"]   # [B, G]

        soft_prompts, compressed_mask = self.compressor(compressor_inputs)  # [B,C,H], [B,C]

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
        # Shift by −1: last soft prompt (index C−1) predicts Y_0; Y_{i−1} predicts Y_i.
        return self.llm.lm_head(out.last_hidden_state)[:, prefix_len - 1:-1, :]  # [B,G,V]

    def _kv_student_logits(self, target_inputs: dict, compressor_inputs: dict) -> torch.Tensor:
        target_ids = target_inputs["target_ids"]     # [B, G]
        target_mask = target_inputs["target_mask"]   # [B, G]

        # 1. Raw strided Mamba hidden -> per-layer pre-RoPE K/V seeded into the cache.
        hidden, compressed_mask = self.compressor.gather_hidden(compressor_inputs)
        B, C, _ = hidden.shape
        hidden = hidden.to(self.kv_projector.k_proj.weight.dtype)
        per_layer_kv = self.kv_projector(hidden)                 # [L,2,B,kvh,C,hd]
        ref_dtype = next(self.llm.parameters()).dtype
        per_layer_kv = per_layer_kv.to(ref_dtype)

        # 2. Compressed positions: left-anchored 0..C_real-1 (pad slots -> 0).
        comp_pos = (compressed_mask.long().cumsum(-1) - 1).clamp_min(0)
        cache = build_kv_cache(per_layer_kv, comp_pos, self.llm.model.rotary_emb,
                               decoder_layers=self.llm.model.layers)

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
        # trigger (idx 0) predicts Y_0 ... idx G-1 predicts Y_{G-1}; the final position drops.
        return self.llm.lm_head(out.last_hidden_state[:, 0:G, :])   # [B,G,V]


class KDPretrainer(LMPretrainer):
    """KD continuation-pretrainer: same compressed-prefix student as LMPretrainer, but the
    target is the frozen teacher's soft distribution instead of the hard corpus token.

    Teacher = the SAME frozen decoder conditioned on the PLAINTEXT prefix X (Qwen tokens,
    left-padded), teacher-forced over the corpus continuation Y in one forward pass — no
    autoregressive generation. The student sees only the COMPRESSED X; the plaintext
    advantage the teacher has is exactly what the compressed representation must recover.
    Both score the same right-padded Y positions, so a KD run is directly comparable to an
    lm_ce_loss run (same X, same corpus Y, soft vs hard labels).

    The teacher pass is compression-agnostic, so this works unchanged for embed/kv modes
    and mamba1/mamba2 — only the student seam (_student_logits, inherited) differs.
    """

    def __init__(self, *args, temperature: float = 2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def forward(self, target_inputs: dict, compressor_inputs: dict,
                teacher_inputs: dict, return_logits: bool = False):
        teacher_logits = self._teacher_logits(teacher_inputs, target_inputs)   # [B,G,V]
        student_logits = self._student_logits(target_inputs, compressor_inputs)  # [B,G,V]
        loss = kd_lm_loss(student_logits, teacher_logits,
                          target_inputs["target_mask"], self.temperature)
        if return_logits:
            return loss, teacher_logits, student_logits
        return loss

    @torch.no_grad()
    def _teacher_logits(self, teacher_inputs: dict, target_inputs: dict) -> torch.Tensor:
        """Frozen teacher, plaintext prefix X + teacher-forced Y -> logits [B,G,V].

        X is LEFT-padded ([B, Lx]) so every row's real prefix ends exactly at column
        Lx-1; Y is right-padded ([B, G]) directly after. The logit at column Lx-1 (last
        real X token) predicts Y_0 and column Lx-1+i predicts Y_i, so the slice
        [Lx-1 : Lx-1+G] lands on the same Y positions the student is scored on. The 2D
        padding mask is expanded by the decoder to its native causal geometry.
        """
        x_ids = teacher_inputs["input_ids"]          # [B, Lx] left-padded
        x_mask = teacher_inputs["attention_mask"]    # [B, Lx]
        y_ids = target_inputs["target_ids"]          # [B, G] right-padded
        y_mask = target_inputs["target_mask"]        # [B, G]
        Lx = x_ids.shape[1]
        G = y_ids.shape[1]

        full_ids = torch.cat([x_ids, y_ids], dim=1)          # [B, Lx+G]
        full_mask = torch.cat([x_mask, y_mask], dim=1)       # [B, Lx+G]
        embeds = self.llm.get_input_embeddings()(full_ids)   # [B, Lx+G, H]

        position_ids = full_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(full_mask == 0, 1)

        out = self.llm.model(
            inputs_embeds=embeds, attention_mask=full_mask,
            position_ids=position_ids, return_dict=True,
        )
        return self.llm.lm_head(out.last_hidden_state)[:, Lx - 1:Lx - 1 + G, :]  # [B,G,V]
