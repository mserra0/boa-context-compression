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
from .dataset import SYSTEM_PROMPT
from .kv_projector import KVProjector
from .utils.kv_cache import build_kv_cache

__all__ = [
    "BaseCompressor", "MambaCompressor", "Mamba2Compressor", "PromptDestiller",
    "split_template_around_doc", "wrapped_geometry", "wrapped_student_geometry",
    "mamba1_context_gather", "mamba2_context_start",
]

"""notation: B batch, I input len, C compressed len, G generated len, H_mamba Mamba hidden, H_llm LLM hidden, V vocab, S stride."""


class PromptDestiller(nn.Module):
    def __init__(self, mamba_model_name: str, llm_model_name: str, device: torch.device,
                 stride: int = 2, temperature: float = 2.0,
                 max_new_tokens: int = 50, compressor_type: str = "mamba",
                 loss_type: str = "kl", confidence_mode: str = "max_prob",
                 compile_llm: bool = False, student_template: str = "bare",
                 compress_question: bool = True, prompt_mode: str = "embed",
                 llm_dtype: str = None, amp: bool = False,
                 anchor_init_token_ids: list = None):
        super().__init__()
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.compressor_type = compressor_type
        self.loss_type = loss_type
        self.confidence_mode = confidence_mode
        self.amp = amp
        self.student_template = student_template  # bare (Design A) or wrapped (Design B-qc/B-ctx)
        self.compress_question = compress_question
        self.prompt_mode = prompt_mode

        _llm_dtype = {None: None, "float32": torch.float32, "fp32": torch.float32,
                      "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                      "float16": torch.float16, "fp16": torch.float16}[llm_dtype]
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, dtype=_llm_dtype).to(device)
        for param in self.llm.parameters():
            param.requires_grad = False
        self.llm.eval()
        self.llm.generation_config.padding_side = 'left'

        # Reconcile pad and eos tokens
        gen_cfg = self.llm.generation_config
        if self.llm.config.pad_token_id is None:
            self.llm.config.pad_token_id = gen_cfg.pad_token_id
        if self.llm.config.pad_token_id is None:
            self.llm.config.pad_token_id = first_id(self.llm.config.eos_token_id)
        gen_cfg.pad_token_id = self.llm.config.pad_token_id

        self._eos_token_ids = sorted(set(
            as_id_list(gen_cfg.eos_token_id) + as_id_list(self.llm.config.eos_token_id)
        ))

        # Precompute wrapped template prefix/suffix embeddings
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
            self.register_buffer('_tpl_sep_emb', sep_emb)  # [sep_len, H]
            self._tpl_sep_len = len(sep_ids)

        if compressor_type == "mamba":
            _pid = self.llm.config.pad_token_id or self.llm.config.eos_token_id
            with torch.no_grad():
                _pad_embed = self.llm.get_input_embeddings()(
                    torch.tensor([_pid], device=device)
                ).squeeze(0)
            self.register_buffer('_pad_embed', _pad_embed)

        if compile_llm:
            self.llm.model.compile(dynamic=True)

        if compressor_type == "mamba2":
            _comp_dtype = torch.float32 if self.amp else next(self.llm.parameters()).dtype
            self.compressor = Mamba2Compressor(
                mamba_model_name,
                self.llm.config.hidden_size,
                device,
                stride,
                dtype=_comp_dtype,
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
                self.compressor.mamba_hidden_size,
                c.num_hidden_layers, c.num_key_value_heads, head_dim,
            ).to(device)
            if student_template == "bare":
                if anchor_init_token_ids is None:
                    anchor_init_token_ids = [151644, 77091, 198]
                self.num_anchors = len(anchor_init_token_ids)
                with torch.no_grad():
                    seed = self.llm.get_input_embeddings()(
                        torch.tensor(anchor_init_token_ids, device=device)
                    ).float()  # [num_anchors, hidden_size]
                self.anchor_embed = nn.Parameter(seed.clone())
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
        generated_ids = teacher_outputs.sequences[:, context_len:]  # [B, G]
        return generated_ids, teacher_logits

    def _replace_pad_soft_prompts(self, soft_prompts, compressed_mask, pad_embed):
        # soft_prompts: [B, C, H], compressed_mask: [B, C]
        pad = pad_embed.view(1, 1, -1)
        soft_prompts = torch.where(compressed_mask.unsqueeze(-1).bool(), soft_prompts, pad)
        return soft_prompts, compressed_mask

    def _build_kv_scaffold(self, B: int, dtype: torch.dtype, compressor_inputs: dict):
        """Decode-side scaffold for prompt_mode='kv' -> (embeds [B, A, H], mask [B, A])."""
        if self.student_template == "bare":
            anchor = self.anchor_embed.to(dtype).unsqueeze(0).expand(B, -1, -1)  # [B, A, H]
            return anchor, anchor.new_ones((B, self.num_anchors), dtype=torch.long)

        prefix = self._tpl_prefix_emb.to(dtype).unsqueeze(0).expand(B, -1, -1)
        suffix = self._tpl_suffix_emb.to(dtype).unsqueeze(0).expand(B, -1, -1)
        P, S = self._tpl_prefix_len, self._tpl_suffix_len
        if self.compress_question:
            embeds = torch.cat([prefix, suffix], dim=1)
            return embeds, embeds.new_ones((B, P + S), dtype=torch.long)

        q_block, q_mask = self._build_question_block(compressor_inputs["question_ids"], dtype)
        sep = self._tpl_sep_emb.to(dtype).unsqueeze(0).expand(B, -1, -1)
        embeds = torch.cat([prefix, sep, q_block, suffix], dim=1)
        mask = torch.cat(
            [q_mask.new_ones((B, P)), q_mask.new_ones((B, self._tpl_sep_len)),
             q_mask, q_mask.new_ones((B, S))],
            dim=1,
        )
        return embeds, mask

    def _kv_forward(self, compressor_inputs, generated_ids, teacher_logits, return_logits):
        hidden, compressed_mask = self.compressor.gather_hidden(compressor_inputs)
        B, C, _ = hidden.shape
        hidden = hidden.to(self.kv_projector.k_proj.weight.dtype)
        per_layer_kv = self.kv_projector(hidden)  # [L, 2, B, kvh, C, hd]
        ref_dtype = next(self.llm.parameters()).dtype
        per_layer_kv = per_layer_kv.to(ref_dtype)

        comp_pos = compressed_mask.long().cumsum(-1) - 1
        comp_pos = comp_pos.clamp_min(0)
        cache = build_kv_cache(per_layer_kv, comp_pos, self.llm.model.rotary_emb,
                               decoder_layers=self.llm.model.layers)

        with torch.no_grad():
            answer_embeds = self.llm.get_input_embeddings()(generated_ids).to(ref_dtype)
        G = answer_embeds.shape[1]
        scaffold, scaffold_mask = self._build_kv_scaffold(B, ref_dtype, compressor_inputs)
        A = scaffold.shape[1]
        dec_input = torch.cat([scaffold, answer_embeds], dim=1)  # [B, A+G, H]

        C_real = compressed_mask.sum(dim=1).long()  # [B]
        dec_mask = torch.cat(
            [scaffold_mask, scaffold_mask.new_ones((B, G))], dim=1)  # [B, A+G]
        dec_pos = C_real.unsqueeze(1) + (dec_mask.cumsum(-1) - 1).clamp_min(0)

        attn_mask = torch.cat([compressed_mask, dec_mask], dim=1)

        out = self.llm.model(
            inputs_embeds=dec_input, past_key_values=cache,
            attention_mask=attn_mask, position_ids=dec_pos,
            use_cache=True, return_dict=True,
        )
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
        """Embed per-sample question ids left-padded to batch max -> (q_block [B, Qmax, H], q_mask [B, Qmax])."""
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
        student_logits = self.llm.lm_head(student_outputs.last_hidden_state)[:, compressed_len - 1:-1, :]
        return student_logits

    @torch.no_grad()
    def generate_compressed(self, soft_prompts: torch.Tensor,
                             compressed_mask: torch.Tensor = None,
                             max_new_tokens: int = None,
                             eos_token_id=None,
                             question_ids=None) -> torch.Tensor:
        """Autoregressive generation from compressed soft prompts -> [B, G]."""
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

        if self.student_template == "wrapped":
            P, S = self._tpl_prefix_len, self._tpl_suffix_len
            prefix = self._tpl_prefix_emb.to(soft_prompts.dtype).unsqueeze(0).expand(B, -1, -1)
            suffix = self._tpl_suffix_emb.to(soft_prompts.dtype).unsqueeze(0).expand(B, -1, -1)
            if not self.compress_question:
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
        Lp = seed_embeds.shape[1]

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

        first_logits = self.llm.lm_head(prefix_out.last_hidden_state[:, -1:, :])
        cur_id = first_logits.argmax(dim=-1)  # [B, 1]
        generated_ids = [cur_id]

        finished = torch.zeros(B, dtype=torch.bool, device=device)
        if eos_tensor is not None:
            finished |= (cur_id == eos_tensor).any(dim=-1)  # [B]

        real_prefix_len = seed_mask.sum(dim=1).long()  # [B]

        mask_dtype = seed_embeds.dtype
        prefix_key_mask = torch.zeros((B, Lp), dtype=mask_dtype, device=device)
        prefix_key_mask.masked_fill_(seed_mask == 0, torch.finfo(mask_dtype).min)

        for k in range(max_new_tokens - 1):
            pos_ids = (real_prefix_len + k).unsqueeze(1)  # [B, 1]
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
            if eos_tensor is not None and bool(finished.all()):
                break

        return torch.cat(generated_ids, dim=1)  # [B, G]

    @torch.no_grad()
    def generate_compressed_kv(self, compressor_inputs: dict, max_new_tokens: int = None,
                               eos_token_id=None) -> torch.Tensor:
        """Autoregressive generation for prompt_mode='kv' -> [B, G]."""
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

        C_real = compressed_mask.sum(dim=1).long()  # [B]
        pad_id = self.llm.config.pad_token_id
        eos_tensor = torch.tensor(eos_ids, device=device) if eos_ids else None

        cur_emb, cur_mask = self._build_kv_scaffold(B, ref_dtype, compressor_inputs)
        seen_mask = compressed_mask.new_zeros((B, 0))
        finished = torch.zeros(B, dtype=torch.bool, device=device)
        generated = []
        for step in range(max_new_tokens):
            n = cur_emb.shape[1]
            full_dec_mask = torch.cat([seen_mask, cur_mask], dim=1)
            rel = (full_dec_mask.cumsum(-1) - 1).clamp_min(0)
            pos = (C_real.unsqueeze(1) + rel)[:, -n:]  # [B, n]
            attn = torch.cat([compressed_mask, seen_mask, cur_mask], dim=1)
            out = self.llm.model(inputs_embeds=cur_emb, past_key_values=cache,
                                 attention_mask=attn, position_ids=pos,
                                 use_cache=True, return_dict=True)
            cache = out.past_key_values
            seen_mask = full_dec_mask
            logits = self.llm.lm_head(out.last_hidden_state[:, -1:, :])
            cur_id = logits.argmax(-1)  # [B, 1]
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

    def forward(self, teacher_inputs: dict = None, compressor_inputs: dict = None,
                return_logits: bool = False):

        generated_ids, teacher_logits = self.generate_teacher(teacher_inputs)

        if self.prompt_mode == "kv":
            return self._kv_forward(compressor_inputs, generated_ids,
                                    teacher_logits, return_logits)

        soft_prompts, compressed_mask = self.compressor(compressor_inputs)  # [B, C, H_llm], [B, C]

        if self.compressor_type == "mamba":
            soft_prompts, compressed_mask = self._replace_pad_soft_prompts(
                soft_prompts, compressed_mask, self._pad_embed
            )

        gen_mask = torch.ones(generated_ids.shape, dtype=compressed_mask.dtype, device=self.device)

        with torch.no_grad():
            teacher_token_embeds = self.llm.get_input_embeddings()(generated_ids)  # [B, G, H_llm]

        soft_prompts = soft_prompts.to(teacher_token_embeds.dtype)
        gen_len = teacher_token_embeds.shape[1]  # G

        if self.student_template == "wrapped":
            B = soft_prompts.shape[0]
            P, S = self._tpl_prefix_len, self._tpl_suffix_len
            prefix = self._tpl_prefix_emb.to(teacher_token_embeds.dtype).unsqueeze(0).expand(B, -1, -1)
            suffix = self._tpl_suffix_emb.to(teacher_token_embeds.dtype).unsqueeze(0).expand(B, -1, -1)

            if not self.compress_question:
                q_block, q_mask = self._build_question_block(
                    compressor_inputs["question_ids"], teacher_token_embeds.dtype
                )
                sep = self._tpl_sep_emb.to(teacher_token_embeds.dtype).unsqueeze(0).expand(B, -1, -1)
                sep_len = self._tpl_sep_len
                combined_embeds = torch.cat(
                    [prefix, q_block, sep, soft_prompts, suffix, teacher_token_embeds], dim=1
                )
                prefix_region_mask = torch.cat(
                    [compressed_mask.new_ones((B, P)), q_mask.to(compressed_mask.dtype),
                     compressed_mask.new_ones((B, sep_len))], dim=1
                )
                full_padding_mask, position_ids, prefix_len = wrapped_geometry(
                    prefix_region_mask, compressed_mask, S, gen_mask
                )
            else:
                combined_embeds = torch.cat([prefix, soft_prompts, suffix, teacher_token_embeds], dim=1)
                full_padding_mask, position_ids, prefix_len = wrapped_student_geometry(
                    compressed_mask, gen_mask, P, S
                )
        else:
            combined_embeds = torch.cat([soft_prompts, teacher_token_embeds], dim=1)
            full_padding_mask = torch.cat([compressed_mask, gen_mask], dim=1)
            prefix_len = soft_prompts.shape[1]  # C
            position_ids = None
            if self.compressor_type == "mamba2":
                position_ids = full_padding_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(full_padding_mask == 0, 1)

        geometry_mask = self._create_causal_mask(prefix_len, gen_len)
        attention_mask = build_additive_mask(geometry_mask, full_padding_mask, combined_embeds.dtype)

        student_logits = self.generate_student(
            combined_embeds, attention_mask, prefix_len, position_ids=position_ids
        )  # [B, G, V]

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

