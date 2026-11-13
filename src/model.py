import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .losses import kl_loss, confidence_weighted_kl_loss, lm_ce_loss
from .compressors import BaseCompressor, MambaCompressor, Mamba2Compressor
from .utils.student_layout import (
    split_template_around_doc,
    wrapped_geometry,
    wrapped_student_geometry,
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
    "MRQASupervisedTrainer",
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
                 stride: int = 2, temperature: float = 2.0,
                 max_new_tokens: int = 50, compressor_type: str = "mamba",
                 loss_type: str = "kl", confidence_mode: str = "max_prob",
                 compile_llm: bool = False, student_template: str = "bare",
                 compress_question: bool = True, prompt_mode: str = "embed",
                 anchor_init_token_ids: list = None,
                 compressor_gradient_checkpointing: bool = True,
                 llm_dtype: torch.dtype = torch.bfloat16,
                 attn_implementation: str = "sdpa"):
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

        # bf16 halves weight/activation memory for the frozen LLM and (via the derived
        # compressor dtype below) the Mamba2 backbone. attn_implementation is set
        # explicitly rather than left to the environment default, so a cluster lacking a
        # compiled flash-attention wheel doesn't silently fall back to the memory-hungry
        # "eager" full-attention-matrix path.
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, dtype=llm_dtype, attn_implementation=attn_implementation,
        ).to(device)
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
                stride,
                dtype=next(self.llm.parameters()).dtype,
                gradient_checkpointing=compressor_gradient_checkpointing,
            )
        elif compressor_type == "mamba":
            self.compressor = MambaCompressor(
                mamba_model_name,
                self.llm.config.hidden_size,
                device,
                stride,
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
            ).to(device=device, dtype=next(self.llm.parameters()).dtype)
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
          wrapped (B-ctx)-> [prefix | sep | question | suffix]  (question left-padded)

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

        # B-ctx: sep ('\n') precedes the literal question, so the decode reads
        # [context KV] ... user\n '\n' {question} <|im_end|>. The '\n' mirrors the
        # teacher's context->question separator ({context}\n{question}), and the
        # question sits DIRECTLY before the suffix (<|im_end|>), exactly as the FullFT
        # teacher saw it.
        q_block, q_mask = self._build_question_block(compressor_inputs["question_ids"], dtype)
        sep = self._tpl_sep_emb.to(dtype).unsqueeze(0).expand(B, -1, -1)
        embeds = torch.cat([prefix, sep, q_block, suffix], dim=1)
        mask = torch.cat(
            [q_mask.new_ones((B, P)), q_mask.new_ones((B, self._tpl_sep_len)),
             q_mask, q_mask.new_ones((B, S))],
            dim=1,
        )
        return embeds, mask

    def _kv_student_logits(self, compressor_inputs, answer_ids):
        """Shared KV-injection student forward: compress -> per-layer KV cache ->
        decode [scaffold | answer] and return the per-answer-token logits [B, G, V].

        `answer_ids` [B, G] are the tokens sitting at the G decode positions: the
        teacher's generated_ids (KD) or the ground-truth answer (supervised ablation).
        The geometry is identical either way; the loss (KL vs CE) is the caller's
        concern. Extracted so PromptDestiller (KD) and MRQASupervisedTrainer (SFT)
        share one provably-identical B-ctx KV forward, keeping the ablation clean.
        """
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

        # 3. Decoder input = [scaffold (A tokens) | answer embeds].
        with torch.no_grad():
            answer_embeds = self.llm.get_input_embeddings()(answer_ids).to(ref_dtype)
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
        return self.llm.lm_head(out.last_hidden_state[:, A - 1:A - 1 + G, :])

    def _kv_forward(self, compressor_inputs, generated_ids, teacher_logits, return_logits):
        student_logits = self._kv_student_logits(compressor_inputs, generated_ids)

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
        B = len(question_ids_list)
        embed = self.llm.get_input_embeddings()
        lens = [len(q) for q in question_ids_list]
        Qmax = max(1, max(lens))
        H = self.llm.config.hidden_size
        q_block = torch.zeros((B, Qmax, H), device=self.device, dtype=dtype)
        q_mask = torch.zeros((B, Qmax), device=self.device, dtype=torch.long)
        for i, q in enumerate(question_ids_list):
            n = len(q)
            if n == 0:
                continue
            ids = torch.tensor(q, device=self.device)
            with torch.no_grad():
                q_block[i, Qmax - n:] = embed(ids).to(dtype)
            q_mask[i, Qmax - n:] = 1
        return q_block, q_mask

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


class MRQASupervisedTrainer(PromptDestiller):
    """MRQA supervised-finetuning ablation — no teacher, no KD.

    Same student geometry as the KD B-ctx KV student (wrapped chat template, compressed
    context injected as a per-layer KV cache, literal-question scaffold), but the decoder
    is trained to predict the GROUND-TRUTH MRQA answer with token-level cross-entropy
    instead of matching a teacher's distribution via KL. This isolates the effect of the
    supervision signal: the ONLY difference from the KD run is generated_ids+kl_loss ->
    answer_ids+lm_ce_loss. Everything else (compressor, kv_projector, scaffold, positions,
    masks) is shared verbatim through `_kv_student_logits`, so the comparison is clean.

    Reuses PromptDestiller.__init__ unchanged; `temperature` / `max_new_tokens` /
    `loss_type` are ignored for the sft objective (no teacher pass is run).

    Scope: prompt_mode='kv' only (the current MRQA-KV frontier: wrapped + compress_question
    False). embed-mode supervision would need an analogous `_embed_student_logits` seam.
    """

    def forward(self, target_inputs: dict, compressor_inputs: dict, return_logits: bool = False):
        if self.prompt_mode != "kv":
            raise NotImplementedError(
                "MRQASupervisedTrainer supports prompt_mode='kv' only. Extract an "
                "`_embed_student_logits` seam from PromptDestiller.forward to add embed mode."
            )
        answer_ids = target_inputs["answer_ids"]     # [B, G] GT answer + <|im_end|>, right-padded
        answer_mask = target_inputs["answer_mask"]   # [B, G] 1 = real answer/eos token, 0 = pad

        student_logits = self._kv_student_logits(compressor_inputs, answer_ids)  # [B, G, V]
        loss = lm_ce_loss(student_logits, answer_ids, answer_mask)
        if return_logits:
            return loss, student_logits
        return loss
