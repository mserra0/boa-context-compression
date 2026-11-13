import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .losses import kl_loss, confidence_weighted_kl_loss
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
from src.utils.offline_kd import reconstruct_teacher_logits

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
                 compress_question: bool = True):
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
        #   B-ctx (compress_question=False) → [prefix | soft | sep | question | suffix]
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
                seed_embeds = torch.cat([prefix, soft_prompts, sep, q_block, suffix], dim=1)
                seed_mask = torch.cat(
                    [compressed_mask.new_ones((B, P)), compressed_mask,
                     compressed_mask.new_ones((B, self._tpl_sep_len)),
                     q_mask.to(compressed_mask.dtype),
                     compressed_mask.new_ones((B, S))], dim=1
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

    def forward(self, teacher_inputs: dict = None, compressor_inputs: dict = None,
                return_logits: bool = False, teacher_data: dict = None):

        if teacher_data is not None:
            # Offline KD: teacher ids come from the dump; dense logits reconstructed
            # after the student forward (needs the student vocab size / device).
            generated_ids = teacher_data["generated_ids"]
            teacher_logits = None
        else:
            generated_ids, teacher_logits = self.generate_teacher(teacher_inputs)
        # generated_ids: [B, G],  teacher_logits: [B, G, V] (or None until reconstructed)

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
                # Design B-ctx: mirror the teacher's [context | question | answer] order,
                # with the compressed context replacing the literal context:
                #   [prefix | soft | sep | question (left-pad to Qmax) | suffix | answer].
                # Only the soft block and the question block carry padding. Built inline
                # (not via wrapped_geometry) because the question + sep sit AFTER the
                # soft block, between it and the all-ones tpl_suffix.
                q_block, q_mask = self._build_question_block(
                    compressor_inputs["question_ids"], teacher_token_embeds.dtype
                )                                   # q_block [B, Qmax, H], q_mask [B, Qmax]
                sep = self._tpl_sep_emb.to(teacher_token_embeds.dtype).unsqueeze(0).expand(B, -1, -1)
                sep_len = self._tpl_sep_len
                combined_embeds = torch.cat(
                    [prefix, soft_prompts, sep, q_block, suffix, teacher_token_embeds], dim=1
                )
                # combined_embeds: [B, P + C + sep_len + Qmax + S + G, H_llm]
                C = compressed_mask.shape[1]
                Qmax = q_mask.shape[1]
                full_padding_mask = torch.cat(
                    [compressed_mask.new_ones((B, P)), compressed_mask,
                     compressed_mask.new_ones((B, sep_len)), q_mask.to(compressed_mask.dtype),
                     compressed_mask.new_ones((B, S)), gen_mask], dim=1
                )
                # HF cumsum(mask)-1 position recipe; pad slots consume no position (-> 1).
                position_ids = full_padding_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(full_padding_mask == 0, 1)
                # Anchor = last suffix token; T_0 predicted from index prefix_len-1.
                prefix_len = P + C + sep_len + Qmax + S
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

        if teacher_logits is None:
            # Offline: scatter dumped top-K into a dense target matching the student vocab.
            teacher_logits = reconstruct_teacher_logits(
                teacher_data["topk_values"], teacher_data["topk_indices"],
                vocab_size=student_logits.shape[-1], device=student_logits.device,
            )

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
