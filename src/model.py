import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .losses import kl_loss, confidence_weighted_kl_loss, reconstruction_loss
from .compressors import BaseCompressor, MambaCompressor, Mamba2Compressor
from .utils.student_layout import (
    split_template_around_doc,
    wrapped_geometry,
    wrapped_student_geometry,
    build_question_block,
    as_id_list,
    first_id,
)
from .utils.compression import mamba1_context_gather, mamba2_context_start
from .utils.attention import MASK_CACHE_CAPACITY, build_causal_mask, build_additive_mask
from .dataset import SYSTEM_PROMPT  # single source of truth for the system turn
from .kv_projector import KVProjector
from .utils.kv_cache import build_kv_cache

# Re-exported for backward compatibility: The implementations now live in `src/compressors.py` and `src/utils/`.
__all__ = [
    "BaseCompressor", "MambaCompressor", "Mamba2Compressor", "PromptDestiller",
    "split_template_around_doc", "wrapped_geometry", "wrapped_student_geometry",
    "mamba1_context_gather", "mamba2_context_start",
]

"""notation:
    - B: batch size
    - I: input sequence length
    - C: compressed sequence length
    - G: generated sequence length
    - H_mamba: hidden dimension of Mamba
    - H_llm: hidden dimension of the LLM
    - V: vocabulary size of the LLM
    - S: stride (compression factor)
"""

class PromptDestiller(nn.Module):
    def __init__(self, mamba_model_name: str, llm_model_name: str, device: torch.device,
                 compression_ratio: float = 0.5, temperature: float = 2.0,
                 max_new_tokens: int = 50, compressor_type: str = "mamba",
                 loss_type: str = "kl", confidence_mode: str = "max_prob",
                 compile_llm: bool = False, student_template: str = "bare",
                 compress_question: bool = True, prompt_mode: str = "embed",
                 anchor_init_token_ids: list = None):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.compressor_type = compressor_type
        self.loss_type = loss_type
        self.confidence_mode = confidence_mode
        # "bare"    → Design A: student sees [soft prompts | answer], no scaffolding.
        # "wrapped" → Design B-qc: student sees [emb(tpl_prefix) | soft | emb(tpl_suffix) | answer], so the frozen LLM generates from its native chat format.
        self.student_template = student_template
        self.compress_question = compress_question
        self.prompt_mode = prompt_mode

        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)
        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm.eval()

        self.llm.generation_config.padding_side = 'left'

        # Qwen2.5 quirk: config.json lacks pad_token_id and has a scalar eos, while
        # generation_config has pad_token_id + a LIST eos. Reconcile both so the frozen
        # LLM pads and stops correctly.
        gen_cfg = self.llm.generation_config

        # pad: prefer config, then generation_config, then fall back to eos.
        if self.llm.config.pad_token_id is None:
            self.llm.config.pad_token_id = gen_cfg.pad_token_id
        if self.llm.config.pad_token_id is None:
            self.llm.config.pad_token_id = first_id(self.llm.config.eos_token_id)
        gen_cfg.pad_token_id = self.llm.config.pad_token_id

        # eos: union of generation_config (may be a list) and config, so manual decoding
        # stops on every end-of-sequence token.
        self._eos_token_ids = sorted(set(
            as_id_list(gen_cfg.eos_token_id) + as_id_list(self.llm.config.eos_token_id)
        ))

        # Wrapped student (Design B-qc): precompute the chat-template prefix/suffix embeds
        # ONCE — identical every batch, so the forward only broadcasts them.
        if student_template == "wrapped":
            tok = AutoTokenizer.from_pretrained(llm_model_name)
            prefix_ids, suffix_ids = split_template_around_doc(tok, SYSTEM_PROMPT)
            embed = self.llm.get_input_embeddings()
            with torch.no_grad():
                prefix_emb = embed(torch.tensor(prefix_ids, device=device)).detach()
                suffix_emb = embed(torch.tensor(suffix_ids, device=device)).detach()
            self.register_buffer('_tpl_prefix_emb', prefix_emb)  # [P, H]
            self.register_buffer('_tpl_suffix_emb', suffix_emb)  # [S, H]
            self._tpl_prefix_len = len(prefix_ids)
            self._tpl_suffix_len = len(suffix_ids)

            sep_ids = tok("\n", add_special_tokens=False)["input_ids"]
            assert len(sep_ids) >= 1, "separator '\\n' tokenized to nothing"
            with torch.no_grad():
                sep_emb = embed(torch.tensor(sep_ids, device=device)).detach()
            self.register_buffer('_tpl_sep_emb', sep_emb)   # [sep_len, H]
            self._tpl_sep_len = len(sep_ids)

        if compressor_type == "mamba":
            _pid = self.llm.config.pad_token_id or self.llm.config.eos_token_id
            with torch.no_grad():
                _pad_embed = self.llm.get_input_embeddings()(
                    torch.tensor([_pid], device=device)
                ).squeeze(0)
            self.register_buffer('_pad_embed', _pad_embed)

        # Compile the inner transformer IN PLACE (not the wrapper — that delegates to the
        # uncompiled module, so `torch.compile(self.llm)` is a no-op). lm_head stays eager.
        # dynamic=True is required: per-batch shape changes (B, C+G) would otherwise force a
        if compile_llm:
            self.llm.model.compile(dynamic=True)

        if compressor_type == "mamba2":
            self.compressor = Mamba2Compressor(
                mamba_model_name,
                self.llm.config.hidden_size,
                device,
                compression_ratio,
                dtype=next(self.llm.parameters()).dtype,
            )
        elif compressor_type == "mamba":
            self.compressor = MambaCompressor(
                mamba_model_name,
                self.llm.config.hidden_size,
                device,
                compression_ratio,
            )
            self.compressor.mamba.config.padding_side = 'left'
        else:
            raise ValueError(
                f"Unknown compressor_type '{compressor_type}'. Valid options: 'mamba', 'mamba2'."
            )

        self._mask_cache: dict = {}

        self.kv_projector = None
        self.anchor_embed = None
        if prompt_mode == "kv":
            if compressor_type not in ("mamba", "mamba2"):
                raise ValueError("prompt_mode='kv' supports compressor_type in {'mamba', 'mamba2'}.")
            if student_template not in ("bare", "wrapped"):
                raise ValueError("prompt_mode='kv' supports student_template in {'bare', 'wrapped'}.")
            if not compress_question and student_template != "wrapped":
                raise ValueError("compress_question=False (B-ctx) requires student_template='wrapped'.")
            c = self.llm.config
            head_dim = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
            self.kv_projector = KVProjector(
                self.compressor.mamba_hidden_size,   # backend-agnostic (mamba1: hidden_size, mamba2: d_model)
                c.num_hidden_layers, c.num_key_value_heads, head_dim,
            ).to(device)
            # Bare KV decodes from a trainable anchor; wrapped KV decodes from the frozen
            # chat-template scaffold (prefix/[question]/suffix), so no trainable anchor.
            if student_template == "bare":
                # Seed the anchor from the embeddings of Qwen2.5's assistant-turn scaffold
                # (<|im_start|>assistant\n = [151644, 77091, 198] by default) so decoding
                # begins from the exact prefix the frozen LLM was trained to generate after.
                # Qwen2.5 has no BOS token, so the chat-turn prompt is the natural seed.
                # The anchor stays trainable; this only sets where it starts.
                if anchor_init_token_ids is None:
                    anchor_init_token_ids = [151644, 77091, 198]
                self.num_anchors = len(anchor_init_token_ids)
                with torch.no_grad():
                    seed = self.llm.get_input_embeddings()(
                        torch.tensor(anchor_init_token_ids, device=device)
                    ).float()                               # [num_anchors, hidden_size]
                self.anchor_embed = nn.Parameter(seed.clone())
            # KV mode never uses the compressor's embedding-space projector (raw hidden is
            # tapped via gather_hidden instead). Freeze it so DDP doesn't flag it as an
            # unused parameter every backward step.
            self.compressor.projector.requires_grad_(False)

    @torch.no_grad()
    def generate_teacher(self, teacher_inputs: dict):
        teacher_outputs = self.llm.generate(
            **teacher_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.llm.config.pad_token_id if self.llm.config.pad_token_id is not None else self.llm.config.eos_token_id
        )
        teacher_logits = torch.stack(teacher_outputs.scores, dim=1)  # [B, G, V]
        context_len = teacher_inputs['input_ids'].shape[1]  # I
        # sequences are [B, I+G]; drop the context to keep the generated tail.
        generated_ids = teacher_outputs.sequences[:, context_len:]  # [B, G]

        return generated_ids, teacher_logits

    def _replace_pad_soft_prompts(self, soft_prompts, compressed_mask, pad_embed):
        # Overwrite left-padded positions with pad_embed; order is preserved so position
        # C-1 stays the last real soft prompt (anchor for predicting T_0).
        # soft_prompts [B, C, H], compressed_mask [B, C] (1=real, 0=pad, left-aligned).
        pad = pad_embed.view(1, 1, -1)
        soft_prompts = torch.where(compressed_mask.unsqueeze(-1).bool(), soft_prompts, pad)
        return soft_prompts, compressed_mask

    def _build_kv_scaffold(self, B: int, dtype: torch.dtype, compressor_inputs: dict):
        """Decode-side scaffold for prompt_mode='kv': the literal tokens that sit AFTER
        the injected context cache and BEFORE the answer (context-first ordering).

          bare           -> [anchor]                      (trainable)
          wrapped (B-qc) -> [prefix | suffix]             (frozen template)
          wrapped (B-ctx)-> [prefix | question | sep | suffix]  (question left-padded)

        Returns (embeds [B, A, H], mask [B, A]); mask 0 only over left-padded question
        slots. The last scaffold token is always the real suffix anchor that predicts T_0.
        """
        if self.student_template == "bare":
            anchor = self.anchor_embed.to(dtype).unsqueeze(0).expand(B, -1, -1)  # [B, A, H]
            return anchor, anchor.new_ones((B, self.num_anchors), dtype=torch.long)

        prefix = self._tpl_prefix_emb.to(dtype).unsqueeze(0).expand(B, -1, -1)
        suffix = self._tpl_suffix_emb.to(dtype).unsqueeze(0).expand(B, -1, -1)
        P, S = self._tpl_prefix_len, self._tpl_suffix_len
        if self.compress_question:
            embeds = torch.cat([prefix, suffix], dim=1)
            return embeds, embeds.new_ones((B, P + S), dtype=torch.long)

        # B-ctx: the literal question sits between prefix and suffix.
        q_block, q_mask = self._build_question_block(compressor_inputs["question_ids"], dtype)
        sep = self._tpl_sep_emb.to(dtype).unsqueeze(0).expand(B, -1, -1)
        embeds = torch.cat([prefix, q_block, sep, suffix], dim=1)
        mask = torch.cat(
            [q_mask.new_ones((B, P)), q_mask,
             q_mask.new_ones((B, self._tpl_sep_len)), q_mask.new_ones((B, S))],
            dim=1,
        )
        return embeds, mask

    def _kv_forward(self, compressor_inputs, generated_ids, teacher_logits, return_logits):
        # 1. Raw strided Mamba hidden -> per-layer pre-RoPE K/V.
        hidden, compressed_mask = self.compressor.gather_hidden(compressor_inputs)
        B, C, _ = hidden.shape
        # Mamba2 hidden may be bf16 while the trainable projector stays fp32; align first.
        hidden = hidden.to(self.kv_projector.k_proj.weight.dtype)
        per_layer_kv = self.kv_projector(hidden)              # [L,2,B,kvh,C,hd]
        ref_dtype = next(self.llm.parameters()).dtype
        per_layer_kv = per_layer_kv.to(ref_dtype)

        # 2. Positions for compressed tokens: left-anchored 0..C_real-1 (pad slots -> 0).
        comp_pos = compressed_mask.long().cumsum(-1) - 1
        comp_pos = comp_pos.clamp_min(0)
        cache = build_kv_cache(per_layer_kv, comp_pos, self.llm.model.rotary_emb,
                               decoder_layers=self.llm.model.layers)

        # 3. Decoder input = [scaffold (A tokens) | teacher answer embeds].
        with torch.no_grad():
            answer_embeds = self.llm.get_input_embeddings()(generated_ids).to(ref_dtype)
        G = answer_embeds.shape[1]
        scaffold, scaffold_mask = self._build_kv_scaffold(B, ref_dtype, compressor_inputs)
        A = scaffold.shape[1]
        dec_input = torch.cat([scaffold, answer_embeds], dim=1)   # [B, A+G, H]

        # 4. Positions continue per sample from each sample's real compressed length,
        #    pad-aware via cumsum (bare/B-qc have no scaffold pad -> contiguous arange).
        C_real = compressed_mask.sum(dim=1).long()              # [B]
        dec_mask = torch.cat(
            [scaffold_mask, scaffold_mask.new_ones((B, G))], dim=1)   # [B, A+G]
        dec_pos = C_real.unsqueeze(1) + (dec_mask.cumsum(-1) - 1).clamp_min(0)

        # 5. Attention mask spans [compressed (pad-aware) | scaffold (pad-aware) | answer].
        attn_mask = torch.cat([compressed_mask, dec_mask], dim=1)

        out = self.llm.model(
            inputs_embeds=dec_input, past_key_values=cache,
            attention_mask=attn_mask, position_ids=dec_pos,
            use_cache=True, return_dict=True,
        )
        # Last anchor (idx A-1) predicts T_0 ... idx A-1+G-1 predicts T_{G-1}; the final
        # position is dropped. Slice the hidden BEFORE lm_head so the dropped position
        # skips a V-sized matmul.
        student_logits = self.llm.lm_head(out.last_hidden_state[:, A - 1:A - 1 + G, :])

        eos_ids = self._eos_token_ids
        if self.loss_type == "confidence_kl":
            loss = confidence_weighted_kl_loss(
                student_logits, teacher_logits, generated_ids,
                self.temperature, eos_ids, mode=self.confidence_mode)
        else:
            loss = kl_loss(student_logits, teacher_logits, generated_ids,
                           self.temperature, eos_ids)
        if return_logits:
            return loss, teacher_logits, student_logits
        return loss

    def _build_question_block(self, question_ids_list, dtype):
        """Embed per-sample literal question ids (Qwen) and LEFT-pad to the batch max.
        Returns (q_block [B, Qmax, H], q_mask [B, Qmax]) with q_mask 1 over real tokens."""
        return build_question_block(
            question_ids_list, dtype, self.llm.get_input_embeddings(),
            self.llm.config.hidden_size, self.device,
        )

    def generate_student(self, combined_embeds: torch.Tensor, attention_mask: torch.Tensor,
                          compressed_len: int, position_ids: torch.Tensor = None):
        student_outputs = self.llm.model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=True,
        )
        # Shift by -1: prediction for T_0 comes from SoftPrompt[-1] (index C-1),
        # prediction for T_i comes from T_{i-1} (index C+i-1).
        student_logits = self.llm.lm_head(student_outputs.last_hidden_state)[:, compressed_len - 1:-1, :]
        return student_logits

    @torch.no_grad()
    def generate_compressed(self, soft_prompts: torch.Tensor,
                             compressed_mask: torch.Tensor = None,
                             max_new_tokens: int = None,
                             eos_token_id=None,
                             question_ids=None) -> torch.Tensor:
        """Autoregressive generation from compressed soft prompts.

        Reproduces the training attention geometry for mamba2:
        1. Prefix forward with causal mask + explicit position_ids → populates KV cache.
        2. Autoregressive generation against the cached prefix (causal for gen tokens).

        Args:
            soft_prompts:    [B, C, H_llm] compressed prefix embeddings.
            compressed_mask: [B, C] 1=valid, 0=pad. If None, assumes no padding.
            max_new_tokens:  overrides self.max_new_tokens.
            eos_token_id:    int or list of ids that stop a sequence. Defaults to the
                             model's full eos set (Qwen2.5 = [<|im_end|>, <|endoftext|>]).
                             Stopping is per-row, so it works for any batch size.
            question_ids:    list of per-sample literal question token ids (Design B-ctx).
                             Only used when student_template == "wrapped" and
                             self.compress_question is False; otherwise ignored.
        Returns:
            generated_ids: [B, G] generated token IDs.
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        if eos_token_id is None:
            eos_ids = list(self._eos_token_ids)
        else:
            eos_ids = as_id_list(eos_token_id)

        B, C, _ = soft_prompts.shape
        device = soft_prompts.device
        pad_id = self.llm.config.pad_token_id
        eos_tensor = torch.tensor(eos_ids, device=device) if eos_ids else None

        if compressed_mask is None:
            compressed_mask = torch.ones((B, C), dtype=torch.long, device=device)

        # Seed the KV cache to match the training anchor (last suffix token predicts T_0):
        #   wrapped → [prefix | soft | suffix]
        #   B-ctx (compress_question=False) → [prefix | question | sep | soft | suffix]
        #   bare → soft only
        if self.student_template == "wrapped":
            P, S = self._tpl_prefix_len, self._tpl_suffix_len
            prefix = self._tpl_prefix_emb.to(soft_prompts.dtype).unsqueeze(0).expand(B, -1, -1)
            suffix = self._tpl_suffix_emb.to(soft_prompts.dtype).unsqueeze(0).expand(B, -1, -1)
            if not self.compress_question:
                # A B-ctx model MUST be given the literal question at generation time, or
                # the seed geometry would silently differ from the trained forward.
                assert question_ids is not None, (
                    "compress_question=False (B-ctx) requires question_ids in generate_compressed"
                )
                q_block, q_mask = self._build_question_block(question_ids, soft_prompts.dtype)
                sep = self._tpl_sep_emb.to(soft_prompts.dtype).unsqueeze(0).expand(B, -1, -1)
                seed_embeds = torch.cat([prefix, q_block, sep, soft_prompts, suffix], dim=1)
                seed_mask = torch.cat(
                    [compressed_mask.new_ones((B, P)), q_mask.to(compressed_mask.dtype),
                     compressed_mask.new_ones((B, self._tpl_sep_len)),
                     compressed_mask, compressed_mask.new_ones((B, S))], dim=1
                )
            else:
                seed_embeds = torch.cat([prefix, soft_prompts, suffix], dim=1)
                seed_mask = torch.cat(
                    [compressed_mask.new_ones((B, P)), compressed_mask,
                     compressed_mask.new_ones((B, S))], dim=1)
        else:
            seed_embeds = soft_prompts
            seed_mask = compressed_mask
        Lp = seed_embeds.shape[1]   # P + C + S (wrapped) or C (bare)

        # 4D additive mask: 0 where allowed, -inf where blocked.
        # Prefix is fully causal; pad columns are masked via seed_mask.
        causal_mask = torch.tril(torch.ones((Lp, Lp), dtype=torch.bool, device=device))
        prefix_allowed = (seed_mask[:, None, None, :] > 0.5) & causal_mask.unsqueeze(0).unsqueeze(0)
        mask_dtype = seed_embeds.dtype
        prefix_mask_4d = torch.zeros((B, 1, Lp, Lp), dtype=mask_dtype, device=device)
        prefix_mask_4d.masked_fill_(~prefix_allowed, torch.finfo(mask_dtype).min)

        position_ids = seed_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(seed_mask == 0, 1)

        prefix_out = self.llm.model(
            inputs_embeds=seed_embeds,
            attention_mask=prefix_mask_4d,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        past_kv = prefix_out.past_key_values

        # First generated token: predicted from the last prefix position's hidden state.
        first_logits = self.llm.lm_head(prefix_out.last_hidden_state[:, -1:, :])
        cur_id = first_logits.argmax(dim=-1)   # [B, 1]
        generated_ids = [cur_id]

        # Per-row early stop: a row is finished once it emits any eos id. Finished rows
        # emit pad thereafter so decoded output is clean past the first eos.
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        if eos_tensor is not None:
            finished |= (cur_id == eos_tensor).any(dim=-1)   # [B]

        real_prefix_len = seed_mask.sum(dim=1).long()  # [B]  (P + C_real + S when wrapped)

        # Additive key-mask: -inf at padded prefix positions so generated tokens never
        # attend to pad KVs (matches training; without it, batched uneven-length decode
        # leaks onto pad KVs and degrades output).
        mask_dtype = seed_embeds.dtype
        prefix_key_mask = torch.zeros((B, Lp), dtype=mask_dtype, device=device)
        prefix_key_mask.masked_fill_(seed_mask == 0, torch.finfo(mask_dtype).min)

        for k in range(max_new_tokens - 1):
            pos_ids = (real_prefix_len + k).unsqueeze(1)   # [B, 1]
            # Key mask spans [prefix (C) | already-generated (k+1)]; gen positions always visible.
            gen_key_mask = torch.zeros((B, k + 1), dtype=mask_dtype, device=device)
            decode_mask = torch.cat([prefix_key_mask, gen_key_mask], dim=1)[:, None, None, :]
            out = self.llm.model(
                input_ids=cur_id,
                past_key_values=past_kv,
                attention_mask=decode_mask,
                position_ids=pos_ids,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            next_logits = self.llm.lm_head(out.last_hidden_state[:, -1:, :])
            cur_id = next_logits.argmax(dim=-1)
            if eos_tensor is not None:
                # Rows already finished emit pad; rows finishing now still keep their eos.
                cur_id = torch.where(finished.unsqueeze(1),
                                     torch.full_like(cur_id, pad_id), cur_id)
                finished |= (cur_id == eos_tensor).any(dim=-1)
            generated_ids.append(cur_id)
            if eos_tensor is not None and bool(finished.all()):
                break

        return torch.cat(generated_ids, dim=1)  # [B, G]

    @torch.no_grad()
    def generate_compressed_kv(self, compressor_inputs: dict, max_new_tokens: int = None,
                               eos_token_id=None) -> torch.Tensor:
        """Autoregressive generation for prompt_mode='kv', bare template.
        Seeds the KV cache from the compressed context, then decodes from the anchor."""
        if self.prompt_mode != "kv":
            raise RuntimeError("generate_compressed_kv requires prompt_mode='kv'.")
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        eos_ids = as_id_list(eos_token_id) if eos_token_id is not None else list(self._eos_token_ids)

        hidden, compressed_mask = self.compressor.gather_hidden(compressor_inputs)
        B, C, _ = hidden.shape
        device = hidden.device
        ref_dtype = next(self.llm.parameters()).dtype
        hidden = hidden.to(self.kv_projector.k_proj.weight.dtype)
        per_layer_kv = self.kv_projector(hidden).to(ref_dtype)
        comp_pos = (compressed_mask.long().cumsum(-1) - 1).clamp_min(0)
        cache = build_kv_cache(per_layer_kv, comp_pos, self.llm.model.rotary_emb,
                               decoder_layers=self.llm.model.layers)

        C_real = compressed_mask.sum(dim=1).long()                  # [B]
        pad_id = self.llm.config.pad_token_id
        eos_tensor = torch.tensor(eos_ids, device=device) if eos_ids else None

        # First step feeds the whole scaffold (bare anchor or wrapped template); later
        # steps feed 1 token. cur_mask carries the scaffold's pad-awareness (B-ctx
        # left-pads the literal question); generated tokens are always real.
        cur_emb, cur_mask = self._build_kv_scaffold(B, ref_dtype, compressor_inputs)
        seen_mask = compressed_mask.new_zeros((B, 0))   # decode-stream keys written so far
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generated = []
        for step in range(max_new_tokens):
            n = cur_emb.shape[1]                                    # A on first step, else 1
            # Positions continue the decode stream from C_real, pad-aware via cumsum.
            full_dec_mask = torch.cat([seen_mask, cur_mask], dim=1)             # [B, off+n]
            rel = (full_dec_mask.cumsum(-1) - 1).clamp_min(0)
            pos = (C_real.unsqueeze(1) + rel)[:, -n:]               # [B, n]
            attn = torch.cat([compressed_mask, seen_mask, cur_mask], dim=1)     # [B, C+off+n]
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
            cur_mask = compressed_mask.new_ones((B, 1))   # generated tokens are real
        return torch.cat(generated, dim=1)

    def forward(self, teacher_inputs: dict = None, compressor_inputs: dict = None,
                return_logits: bool = False):

        generated_ids, teacher_logits = self.generate_teacher(teacher_inputs)
        # generated_ids: [B, G],  teacher_logits: [B, G, V]

        # KV mode taps the raw Mamba hidden inside _kv_forward (via gather_hidden), so the
        # embedding-space compressor call is skipped entirely — running it here would do a
        # redundant second Mamba forward and discard its projected soft prompts.
        if self.prompt_mode == "kv":
            return self._kv_forward(compressor_inputs, generated_ids,
                                    teacher_logits, return_logits)

        soft_prompts, compressed_mask = self.compressor(compressor_inputs)
        # soft_prompts: [B, C, H_llm],  compressed_mask: [B, C]

        if self.compressor_type == "mamba":
            soft_prompts, compressed_mask = self._replace_pad_soft_prompts(
                soft_prompts, compressed_mask, self._pad_embed
            )

        gen_mask = torch.ones(generated_ids.shape, dtype=compressed_mask.dtype, device=self.device)

        with torch.no_grad():
            teacher_token_embeds = self.llm.get_input_embeddings()(generated_ids)
        # teacher_token_embeds: [B, G, H_llm]

        # Dtype alignment: Mamba2 projector may produce bf16; LLM embeddings may differ.
        soft_prompts = soft_prompts.to(teacher_token_embeds.dtype)

        gen_len = teacher_token_embeds.shape[1]  # G

        if self.student_template == "wrapped":
            # Precomputed frozen embeds, broadcast across the batch.
            B = soft_prompts.shape[0]
            P, S = self._tpl_prefix_len, self._tpl_suffix_len
            prefix = self._tpl_prefix_emb.to(teacher_token_embeds.dtype).unsqueeze(0).expand(B, -1, -1)
            suffix = self._tpl_suffix_emb.to(teacher_token_embeds.dtype).unsqueeze(0).expand(B, -1, -1)

            if not self.compress_question:
                # Design B-ctx: [prefix | question (left-pad to Qmax) | sep | soft | suffix
                # | answer]. Only the question and soft block carry padding.
                q_block, q_mask = self._build_question_block(
                    compressor_inputs["question_ids"], teacher_token_embeds.dtype
                )                                   # q_block [B, Qmax, H], q_mask [B, Qmax]
                sep = self._tpl_sep_emb.to(teacher_token_embeds.dtype).unsqueeze(0).expand(B, -1, -1)
                sep_len = self._tpl_sep_len
                combined_embeds = torch.cat(
                    [prefix, q_block, sep, soft_prompts, suffix, teacher_token_embeds], dim=1
                )
                # combined_embeds: [B, P + Qmax + sep_len + C + S + G, H_llm]
                prefix_region_mask = torch.cat(
                    [compressed_mask.new_ones((B, P)), q_mask.to(compressed_mask.dtype),
                     compressed_mask.new_ones((B, sep_len))], dim=1
                )
                full_padding_mask, position_ids, prefix_len = wrapped_geometry(
                    prefix_region_mask, compressed_mask, S, gen_mask
                )
            else:
                # Design B-qc: [ emb(tpl_prefix) | soft prompts | emb(tpl_suffix) | answer ]
                combined_embeds = torch.cat([prefix, soft_prompts, suffix, teacher_token_embeds], dim=1)
                # combined_embeds: [B, P + C + S + G, H_llm]
                full_padding_mask, position_ids, prefix_len = wrapped_student_geometry(
                    compressed_mask, gen_mask, P, S
                )
            # prefix_len = anchor offset; the last suffix token predicts T_0.
        else:
            combined_embeds = torch.cat([soft_prompts, teacher_token_embeds], dim=1)
            # combined_embeds: [B, C + G, H_llm]
            full_padding_mask = torch.cat([compressed_mask, gen_mask], dim=1)
            prefix_len = soft_prompts.shape[1]   # C
            position_ids = None
            if self.compressor_type == "mamba2":
                # Cumsum-based position_ids match the teacher's generation positions
                # and correctly handle left-padded soft prompts.
                position_ids = full_padding_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(full_padding_mask == 0, 1)

        geometry_mask = self._create_causal_mask(prefix_len, gen_len)
        attention_mask = build_additive_mask(geometry_mask, full_padding_mask, combined_embeds.dtype)

        student_logits = self.generate_student(
            combined_embeds, attention_mask, prefix_len, position_ids=position_ids
        )
        # student_logits: [B, G, V]

        # eos-aware loss mask: supervise up to and including the first eos, drop the
        # post-eos pad tail. Passes the full eos set so a turn ending with <|endoftext|>
        # (which is also the pad id) is still supervised as a real stop token.
        eos_ids = self._eos_token_ids
        if self.loss_type == "confidence_kl":
            loss = confidence_weighted_kl_loss(
                student_logits, teacher_logits, generated_ids,
                self.temperature, eos_ids, mode=self.confidence_mode,
            )
        elif self.loss_type == "kl":
            loss = kl_loss(student_logits, teacher_logits, generated_ids, self.temperature, eos_ids)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type!r}. Choose 'kl' or 'confidence_kl'.")

        if return_logits:
            return loss, teacher_logits, student_logits
        return loss

    def _create_causal_mask(self, prefix_len: int, gen_len: int) -> torch.Tensor:
        cache_key = (prefix_len, gen_len)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        mask = build_causal_mask(prefix_len, gen_len, self.device)
        if len(self._mask_cache) >= MASK_CACHE_CAPACITY:
            self._mask_cache.pop(next(iter(self._mask_cache)))
        self._mask_cache[cache_key] = mask
        return mask


class ICAECompressor(BaseCompressor):
    """ICAE encoder: LoRA-adapted LLM + learnable [MEM] tokens.

    Appends n_mem learnable embeddings to the context, runs the concatenation
    through a LoRA-adapted causal LM, and returns the hidden states at the
    [MEM] positions as compressed memory slots.

    Because [MEM] tokens sit at the end, standard causal attention gives each
    MEM_i full visibility into the real context and preceding MEM tokens —
    no special masking required on the encoder side.

    Shares its base model with ICAEDestiller's frozen decoder (see that class's
    __init__): base_lm is the same LoRA-injected causal LM, registered as a
    submodule exactly once (via ICAEDestiller.llm) so its parameters aren't
    double-counted. Adapters are OFF by ambient default (so ICAEDestiller.llm
    keeps behaving like a plain frozen model for every external caller that
    already expects that — ModelRunner, inference.py) — this class temporarily
    flips them on around its own forward pass only, via lora_tuner.

    Input dict keys:
        input_ids      [B, I]  — left-padded context token IDs
        attention_mask [B, I]  — 1 at real tokens, 0 at padding

    Returns:
        soft_prompts:    [B, n_mem, H_llm]
        compressed_mask: [B, n_mem]  — all ones (MEM slots are never padded)
    """

    def __init__(self, base_lm, lora_tuner, llm_hidden_dim: int, device: torch.device,
                 n_mem: int, compression_ratio: float = 0.1):
        super().__init__(llm_hidden_dim, device, compression_ratio)
        self.n_mem = n_mem
        # Neither is registered as an nn.Module attribute (would double-count
        # base_lm's parameters alongside ICAEDestiller.llm, since lora_tuner also
        # wraps the same object) -- object.__setattr__ bypasses nn.Module's
        # auto-registration while keeping both fully usable for calls below.
        object.__setattr__(self, "base_lm", base_lm)
        object.__setattr__(self, "lora_tuner", lora_tuner)

        # from_pretrained doesn't pin a dtype, so the model may load in bf16
        # depending on the checkpoint's default — match it explicitly (mem_embeddings
        # must share dtype with the model's own embeddings/hidden states).
        encoder_dtype = base_lm.dtype

        # Learnable [MEM] embeddings in the shared model's hidden space
        self.mem_embeddings = nn.Parameter(
            torch.randn(n_mem, llm_hidden_dim, device=device, dtype=encoder_dtype) * 0.02
        )

    def forward(self, compressor_inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        input_ids = compressor_inputs['input_ids']                    # [B, I]
        attention_mask = compressor_inputs.get('attention_mask')      # [B, I]
        B, I = input_ids.shape

        # Embed context tokens through the shared model's own embedding layer
        context_embeds = self.base_lm.get_input_embeddings()(input_ids)  # [B, I, H_llm]

        # Tile learnable [MEM] embeddings across the batch
        mem_embeds = self.mem_embeddings.unsqueeze(0).expand(B, -1, -1)  # [B, n_mem, H_llm]

        # [context || MEM] — MEM tokens see full context via causal attention
        full_embeds = torch.cat([context_embeds, mem_embeds], dim=1)  # [B, I+n_mem, H_llm]

        # Extend the padding mask: MEM positions are always valid
        mem_mask = torch.ones(B, self.n_mem, dtype=attention_mask.dtype, device=self.device)
        full_mask = torch.cat([attention_mask, mem_mask], dim=1)       # [B, I+n_mem]

        # This is the ONLY place LoRA is meant to be active -- enable for this call,
        # then restore the disabled default. Backbone-only call (mirrors
        # ICAEDestiller._decode's self.llm.model(...)) skips computing the full
        # lm_head vocab projection, which this path never uses.
        self.lora_tuner.enable_adapter_layers()
        try:
            backbone_out = self.base_lm.model(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                return_dict=True,
            )
        finally:
            self.lora_tuner.disable_adapter_layers()
            # disable_adapter_layers() also sets requires_grad=False on the LoRA
            # params (documented PEFT behavior, intended for pure-inference use where
            # nothing calls backward() afterward). Here, .backward() for this training
            # step's loss runs later in the caller (src/pretrain.py), well after this
            # forward pass returns -- if left False, the LoRA gradients would be
            # silently dropped instead of accumulated, so force it back on.
            for n, p in self.base_lm.named_parameters():
                if ".lora_" in n:
                    p.requires_grad_(True)
        hidden = backbone_out.last_hidden_state  # [B, I+n_mem, H_llm]

        # Hidden states at the last n_mem positions = memory slots
        soft_prompts = hidden[:, I:, :]  # [B, n_mem, H_llm]

        compressed_mask = torch.ones(B, self.n_mem, dtype=torch.long, device=self.device)
        return soft_prompts, compressed_mask

    @torch.no_grad()
    def compress_single(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        B, L = input_ids.shape
        attention_mask = torch.ones((B, L), device=self.device, dtype=torch.long)
        return self.forward({'input_ids': input_ids, 'attention_mask': attention_mask})


class ICAEDestiller(nn.Module):
    """ICAE training wrapper.

    Encoder: LoRA-adapted LLM — compresses context into n_mem memory slots.
    Decoder: Frozen LLM       — reconstructs context from memory slots (CE loss).

    forward() signature matches PromptDestiller so train_epoch / validate work
    without modification.
    """

    def __init__(self, encoder_model_name: str, llm_model_name: str, device: torch.device,
                 n_mem: int, compression_ratio: float = 0.1, temperature: float = 2.0,
                 max_new_tokens: int = 50, lora_r: int = 8, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, objective: str = "ae",
                 loss_type: str = "kl", confidence_mode: str = "max_prob",
                 pretrain_ae_weight: float = 0.5, compress_question: bool = True):
        super().__init__()
        self.device = device
        self.n_mem = n_mem
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.objective = objective
        self.loss_type = loss_type
        # When False (objective='gt' only): the compressor's input never contains the
        # question at all (see CompressionDataset._build_compressor_text) -- the
        # question reaches the decoder only as literal embeddings inserted between the
        # memory slots and the answer (_teacher_forced_target_loss / generate_compressed).
        # Genuinely question-blind, unlike PromptDestiller's B-ctx (which still lets the
        # question condition the compressor's raw input; see docs/architecture/student-templates.md on the `main` branch).
        self.compress_question = compress_question
        self.confidence_mode = confidence_mode
        # lambda in the paper's L_pretrain = lambda*L_AE + (1-lambda)*L_LM (objective='ae_lm' only).
        self.pretrain_ae_weight = pretrain_ae_weight
        # Populated by forward() only for objective='ae_lm', so train.py can log the
        # two components separately; None for every other objective.
        self.last_loss_components = None

        # Encoder and decoder are the SAME underlying weights (LoRA-adapted vs. not)
        # instead of two separate full copies of the base model (halves its VRAM
        # footprint). Assumes encoder_model_name == llm_model_name.
        assert encoder_model_name == llm_model_name, (
            "ICAEDestiller shares one base model between encoder and decoder — "
            f"encoder_model_name ({encoder_model_name!r}) must equal llm_model_name "
            f"({llm_model_name!r})."
        )
        from peft import get_peft_model, LoraConfig, TaskType

        base_lm = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)
        if base_lm.config.pad_token_id is None:
            base_lm.config.pad_token_id = base_lm.config.eos_token_id
        base_lm.generation_config.padding_side = 'left'
        # eos: union of generation_config (may be a list) and config, matching
        # PromptDestiller — needed because the pad id and the real stop token can differ.
        self._eos_token_ids = sorted(set(
            as_id_list(base_lm.generation_config.eos_token_id) + as_id_list(base_lm.config.eos_token_id)
        ))

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"],
        )
        # get_peft_model() injects LoRA layers into base_lm IN PLACE (freezing every
        # other param automatically) and returns a PeftModel wrapper around it.
        # lora_tuner (the wrapper's .base_model) owns enable/disable_adapter_layers(),
        # the two methods that actually flip the injected layers' forward behavior.
        lora_tuner = get_peft_model(base_lm, lora_config).base_model
        # Adapters OFF by default: every external caller that already treats
        # `<destiller>.llm` as a plain frozen causal LM (ModelRunner.run_teacher,
        # inference.py's chat_loop) keeps working unmodified. ICAECompressor is the
        # only place that flips this on, and only around its own forward call.
        lora_tuner.disable_adapter_layers()
        self.llm = base_lm
        # Not registered as an nn.Module attribute (lora_tuner wraps the same base_lm
        # object already registered via self.llm above; state_dict() doesn't dedupe
        # repeated submodule references the way parameters() does, so a plain
        # `self._lora_tuner = ...` here would silently double the checkpoint size).
        object.__setattr__(self, "_lora_tuner", lora_tuner)

        # ICAE encoder (learnable [MEM] tokens; LoRA lives on the shared base_lm)
        self.compressor = ICAECompressor(
            base_lm=base_lm,
            lora_tuner=lora_tuner,
            llm_hidden_dim=base_lm.config.hidden_size,
            device=device,
            n_mem=n_mem,
            compression_ratio=compression_ratio,
        )

        self._mask_cache: dict = {}

    def _create_prefix_causal_mask(self, prefix_len: int, gen_len: int) -> torch.Tensor:
        # Memory slots = fully bidirectional prefix; context tokens = causal.
        cache_key = (prefix_len, gen_len)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        total_len = prefix_len + gen_len
        mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=self.device)
        mask[:prefix_len, :prefix_len] = True
        if gen_len > 0:
            mask[prefix_len:, :prefix_len] = True
            mask[prefix_len:, prefix_len:] = torch.tril(
                torch.ones(gen_len, gen_len, dtype=torch.bool, device=self.device)
            )

        mask = mask.unsqueeze(0).unsqueeze(0)
        if len(self._mask_cache) >= MASK_CACHE_CAPACITY:
            self._mask_cache.pop(next(iter(self._mask_cache)))
        self._mask_cache[cache_key] = mask
        return mask

    def _apply_prefix_mask(self, full_mask: torch.Tensor, geometry_mask: torch.Tensor,
                            dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        final_mask = geometry_mask & (full_mask[:, None, None, :] > 0.5)
        final_mask.diagonal(dim1=-2, dim2=-1).fill_(True)

        attention_mask_4d = torch.zeros_like(final_mask, dtype=dtype)
        attention_mask_4d.masked_fill_(~final_mask, torch.finfo(dtype).min)

        position_ids = full_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(full_mask == 0, 1)
        return attention_mask_4d, position_ids

    def _decode(self, combined_embeds: torch.Tensor, full_mask: torch.Tensor,
                geometry_mask: torch.Tensor) -> torch.Tensor:
        attention_mask_4d, position_ids = self._apply_prefix_mask(
            full_mask, geometry_mask, combined_embeds.dtype
        )
        # Frozen-decoder role: adapters are off by default (see __init__), so this
        # sees the plain base model.
        decoder_out = self.llm.model(
            inputs_embeds=combined_embeds,
            attention_mask=attention_mask_4d,
            position_ids=position_ids,
            return_dict=True,
        )
        return self.llm.lm_head(decoder_out.last_hidden_state)

    def _teacher_forced_target_loss(self, memory_slots: torch.Tensor, compressed_mask: torch.Tensor,
                                     target_ids: torch.Tensor, target_mask: torch.Tensor,
                                     question_ids: list = None,
                                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared by objectives 'gt' and 'lm': decode [MEM | target[:-1]] -> target
        (or, when `question_ids` is given, [MEM | question | target[:-1]] -> target)
        via plain CE, with no teacher generation involved. The only difference between
        'gt' and 'lm' is what `target_ids` actually contains (gold MRQA answer vs. a
        withheld document continuation) -- `question_ids` is only ever passed for 'gt'
        with compress_question=False; 'lm' has no question concept.

        The question span (when present) is literal LLM embeddings, left-padded to the
        batch max like the target span already is, and is causal but NOT part of the
        loss -- only target positions are scored. It sits between the bidirectional
        memory prefix and the causal target span, so `_create_prefix_causal_mask` needs
        no changes: memory stays the bidirectional prefix, [question | target[:-1]] is
        just a wider causal "gen" region than before.
        """
        with torch.no_grad():
            target_embeds = self.llm.get_input_embeddings()(target_ids)  # [B, A, H_llm]
        memory_slots = memory_slots.to(target_embeds.dtype)

        decoder_tgt_embeds = target_embeds[:, :-1, :]
        decoder_tgt_mask = target_mask[:, :-1]

        if question_ids is not None:
            q_block, q_mask = build_question_block(
                question_ids, target_embeds.dtype, self.llm.get_input_embeddings(),
                self.llm.config.hidden_size, self.device,
            )
            q_len = q_block.shape[1]
            combined_embeds = torch.cat([memory_slots, q_block, decoder_tgt_embeds], dim=1)
            full_mask = torch.cat([compressed_mask, q_mask, decoder_tgt_mask], dim=1)
            geometry_mask = self._create_prefix_causal_mask(
                self.n_mem, q_len + target_ids.shape[1] - 1
            )
            anchor = self.n_mem + q_len - 1
        else:
            combined_embeds = torch.cat([memory_slots, decoder_tgt_embeds], dim=1)
            full_mask = torch.cat([compressed_mask, decoder_tgt_mask], dim=1)
            geometry_mask = self._create_prefix_causal_mask(self.n_mem, target_ids.shape[1] - 1)
            anchor = self.n_mem - 1

        logits = self._decode(combined_embeds, full_mask, geometry_mask)
        recon_logits = logits[:, anchor:, :]  # [B, A, V]
        loss = reconstruction_loss(recon_logits, target_ids, target_mask)
        return loss, recon_logits

    @torch.no_grad()
    def generate_teacher(self, teacher_inputs: dict):
        # Teacher = frozen, un-adapted decoder generating its own reference output.
        # Adapters are off by default (see __init__), so this is already the case.
        teacher_outputs = self.llm.generate(
            **teacher_inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=self.llm.config.pad_token_id if self.llm.config.pad_token_id is not None else self.llm.config.eos_token_id
        )
        teacher_logits = torch.stack(teacher_outputs.scores, dim=1)
        context_len = teacher_inputs['input_ids'].shape[1]
        generated_ids = teacher_outputs.sequences[:, context_len:]
        return generated_ids, teacher_logits

    @torch.no_grad()
    def generate_compressed(self, soft_prompts: torch.Tensor,
                             compressed_mask: torch.Tensor = None,
                             max_new_tokens: int = None,
                             eos_token_id=None,
                             question_ids: list = None) -> torch.Tensor:
        """Autoregressive generation from compressed memory slots (inference only).

        Stopping is per-row (matches PromptDestiller.generate_compressed), so it
        works for any batch size. Rows that hit EOS stop contributing text, so batched
        eval (B > 1) does not run finished rows on to max_new_tokens, which would
        produce repeated/looping text and distort EM/F1.

        `question_ids`: literal per-sample question token id lists (LLM tokenizer), for
        the question-blind path (compress_question=False, objective='gt'). When given,
        seeds the KV cache with [memory | question] before decoding the answer,
        reproducing the training-time geometry in `_teacher_forced_target_loss` exactly
        — same `_create_prefix_causal_mask`/`_apply_prefix_mask` calls, bidirectional
        memory + causal question, NOT a plain causal mask over everything (that would
        mismatch how the memory was actually trained to attend to itself).
        """
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens
        eos_ids = as_id_list(eos_token_id) if eos_token_id is not None else list(self._eos_token_ids)
        eos_tensor = torch.tensor(eos_ids, device=soft_prompts.device) if eos_ids else None
        pad_id = self.llm.config.pad_token_id

        B, C, _ = soft_prompts.shape
        device = soft_prompts.device
        mask_dtype = soft_prompts.dtype
        if compressed_mask is None:
            compressed_mask = torch.ones((B, C), dtype=torch.long, device=device)

        if question_ids is not None:
            q_block, q_mask = build_question_block(
                question_ids, soft_prompts.dtype, self.llm.get_input_embeddings(),
                self.llm.config.hidden_size, device,
            )
            q_len = q_block.shape[1]
            seed_embeds = torch.cat([soft_prompts, q_block], dim=1)
            seed_mask = torch.cat([compressed_mask, q_mask], dim=1)
        else:
            q_len = 0
            seed_embeds = soft_prompts
            seed_mask = compressed_mask

        geometry_mask = self._create_prefix_causal_mask(self.n_mem, q_len)
        prefix_mask_4d, position_ids = self._apply_prefix_mask(seed_mask, geometry_mask, mask_dtype)

        # Decoder role throughout this method: adapters are off by default (see
        # __init__), same as _decode/generate_teacher.
        prefix_out = self.llm.model(
            inputs_embeds=seed_embeds,
            attention_mask=prefix_mask_4d,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
        past_kv = prefix_out.past_key_values
        first_logits = self.llm.lm_head(prefix_out.last_hidden_state[:, -1:, :])
        cur_id = first_logits.argmax(dim=-1)
        generated_ids = [cur_id]
        real_prefix_len = seed_mask.sum(dim=1).long()

        # Additive key-mask over the prefix (memory + question): -inf at padded
        # question slots so generated tokens never attend to pad KVs. Memory is never
        # padded (compressed_mask is always all-ones for ICAE), so this is a no-op
        # in the question_ids=None case (matches the prior behavior exactly).
        prefix_key_mask = torch.zeros_like(seed_mask, dtype=mask_dtype)
        prefix_key_mask.masked_fill_(seed_mask == 0, torch.finfo(mask_dtype).min)

        # Per-row early stop: a row is finished once it emits any eos id. Finished
        # rows emit pad thereafter so decoded output is clean past the first eos.
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        if eos_tensor is not None:
            finished |= (cur_id == eos_tensor).any(dim=-1)

        for k in range(max_new_tokens - 1):
            if eos_tensor is not None and bool(finished.all()):
                break
            pos_ids = (real_prefix_len + k).unsqueeze(1)
            # Key mask spans [prefix (memory+question) | already-generated (k+1)].
            gen_key_mask = torch.zeros((B, k + 1), dtype=mask_dtype, device=device)
            decode_mask = torch.cat([prefix_key_mask, gen_key_mask], dim=1)[:, None, None, :]
            out = self.llm.model(
                input_ids=cur_id,
                past_key_values=past_kv,
                attention_mask=decode_mask,
                position_ids=pos_ids,
                use_cache=True,
                return_dict=True,
            )
            past_kv = out.past_key_values
            next_logits = self.llm.lm_head(out.last_hidden_state[:, -1:, :])
            cur_id = next_logits.argmax(dim=-1)
            if eos_tensor is not None:
                cur_id = torch.where(finished.unsqueeze(1),
                                     torch.full_like(cur_id, pad_id), cur_id)
                finished |= (cur_id == eos_tensor).any(dim=-1)
            generated_ids.append(cur_id)

        return torch.cat(generated_ids, dim=1)

    def forward(self, llm_inputs: dict = None, compressor_inputs: dict = None,
                answer_inputs: dict = None, continuation_inputs: dict = None,
                return_logits: bool = False):
        # --- Encode context into memory slots ---
        memory_slots, compressed_mask = self.compressor(compressor_inputs)
        # memory_slots: [B, n_mem, H_llm],  compressed_mask: [B, n_mem] (all ones)

        self.last_loss_components = None

        if self.objective == "ae":
            # Autoencoding pretraining: decoder reconstructs the compressed context
            # itself from [MEM | context[:-1]]. No question/answer signal involved.
            # Structurally identical to 'gt'/'lm' below — the target here is simply
            # the same span that was compressed.
            loss, recon_logits = self._teacher_forced_target_loss(
                memory_slots, compressed_mask,
                compressor_inputs['input_ids'], compressor_inputs['attention_mask'],
            )
            teacher_logits_for_artifact = torch.zeros_like(recon_logits)
            student_logits = recon_logits

        elif self.objective == "ae_lm":
            # Paper's actual pretraining recipe (arXiv:2307.06945 Appendix A):
            # L_pretrain = lambda*L_AE + (1-lambda)*L_LM, lambda=0.5 by default.
            # Both losses share the SAME memory_slots (one compressor forward over
            # span A), but get two independent decoder forward passes: one
            # reconstructing A itself, one predicting the withheld continuation B.
            if continuation_inputs is None:
                raise ValueError("objective='ae_lm' requires continuation_inputs (LM continuation target).")

            ae_loss, ae_recon_logits = self._teacher_forced_target_loss(
                memory_slots, compressed_mask,
                compressor_inputs['input_ids'], compressor_inputs['attention_mask'],
            )
            lm_loss, lm_recon_logits = self._teacher_forced_target_loss(
                memory_slots, compressed_mask,
                continuation_inputs['input_ids'], continuation_inputs['attention_mask'],
            )
            lam = self.pretrain_ae_weight
            loss = lam * ae_loss + (1 - lam) * lm_loss
            self.last_loss_components = {"ae_loss": ae_loss.detach(), "lm_loss": lm_loss.detach()}
            recon_logits = lm_recon_logits  # arbitrary pick for artifact-shape compat
            teacher_logits_for_artifact = torch.zeros_like(recon_logits)
            student_logits = recon_logits

        elif self.objective in ("gt", "lm"):
            # 'gt': fine-tuning against gold MRQA answers — decoder predicts the
            #       ground-truth answer from [MEM | answer[:-1]] (question-blind memory
            #       compresses only the context; when compress_question=False the
            #       question is inserted as literal tokens: [MEM | question | answer[:-1]]).
            #       With compress_question=True (default), the question is already
            #       encoded into memory by the compressor instead.
            # 'lm': pretraining continuation objective — decoder predicts a withheld
            #       span of the SAME document that follows what was compressed,
            #       from [MEM | continuation[:-1]], with no direct access to it.
            #       Reproduces the half of ICAE's pretraining recipe that AE alone
            #       doesn't cover (memory must support generation, not just recall).
            #       No question concept here regardless of compress_question.
            target_inputs = answer_inputs if self.objective == "gt" else continuation_inputs
            if target_inputs is None:
                kind = "answer_inputs" if self.objective == "gt" else "continuation_inputs"
                raise ValueError(f"objective={self.objective!r} requires {kind}.")

            question_ids = None
            if self.objective == "gt" and not self.compress_question:
                question_ids = compressor_inputs.get("question_ids")

            loss, recon_logits = self._teacher_forced_target_loss(
                memory_slots, compressed_mask,
                target_inputs['input_ids'], target_inputs['attention_mask'],
                question_ids=question_ids,
            )
            teacher_logits_for_artifact = torch.zeros_like(recon_logits)
            student_logits = recon_logits

        elif self.objective == "distill":
            # KL distillation against the frozen LLM's own teacher-generated answer,
            # mirroring PromptDestiller's recipe exactly (same data, same loss) so the
            # only difference vs. the Mamba baseline is the compressor architecture.
            if llm_inputs is None:
                raise ValueError("objective='distill' requires llm_inputs (teacher prompt batch).")

            generated_ids, teacher_logits = self.generate_teacher(llm_inputs)
            # generated_ids: [B, G],  teacher_logits: [B, G, V]
            gen_mask = torch.ones(generated_ids.shape, dtype=compressed_mask.dtype, device=self.device)

            with torch.no_grad():
                teacher_token_embeds = self.llm.get_input_embeddings()(generated_ids)
            memory_slots = memory_slots.to(teacher_token_embeds.dtype)

            combined_embeds = torch.cat([memory_slots, teacher_token_embeds], dim=1)
            full_mask = torch.cat([compressed_mask, gen_mask], dim=1)
            geometry_mask = self._create_prefix_causal_mask(self.n_mem, generated_ids.shape[1])

            logits = self._decode(combined_embeds, full_mask, geometry_mask)
            # Shift by -1, same convention as PromptDestiller.generate_student.
            student_logits = logits[:, self.n_mem - 1:-1, :]  # [B, G, V]

            eos_ids = self._eos_token_ids
            if self.loss_type == "confidence_kl":
                loss = confidence_weighted_kl_loss(
                    student_logits, teacher_logits, generated_ids,
                    self.temperature, eos_ids, mode=self.confidence_mode,
                )
            elif self.loss_type == "kl":
                loss = kl_loss(student_logits, teacher_logits, generated_ids, self.temperature, eos_ids)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type!r}. Choose 'kl' or 'confidence_kl'.")
            teacher_logits_for_artifact = teacher_logits

        else:
            raise ValueError(
                f"Unknown objective: {self.objective!r}. Choose 'ae', 'ae_lm', 'gt', 'lm', or 'distill'."
            )

        if return_logits:
            return loss, teacher_logits_for_artifact, student_logits
        return loss
