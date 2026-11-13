import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from .losses import lm_ce_loss
from .compressors import MambaCompressor, Mamba2Compressor
from .kv_projector import KVProjector
from .utils.attention import build_causal_mask, build_additive_mask
from .utils.kv_cache import build_kv_cache

"""notation: B batch, I input len, C compressed len, G continuation len, H_llm LLM hidden, V vocab, S stride."""


class LMPretrainer(nn.Module):
    """Continuation-LM pretrainer: compress prefix X -> frozen decoder predicts suffix Y.

    Supports prompt_mode='embed' (soft prompts prepended) and prompt_mode='kv' (K/V cache seeding).
    """

    def __init__(self, mamba_model_name: str, llm_model_name: str, device: torch.device,
                 stride: int = 2, compressor_type: str = "mamba2",
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
                stride, dtype=next(self.llm.parameters()).dtype,
            )
        elif compressor_type == "mamba":
            self.compressor = MambaCompressor(
                mamba_model_name, self.llm.config.hidden_size, device, stride,
            )
            self.compressor.mamba.config.padding_side = "left"
        else:
            raise ValueError(f"Unknown compressor_type {compressor_type!r}. Use 'mamba' or 'mamba2'.")

        self.kv_projector = None
        if prompt_mode == "kv":
            c = self.llm.config
            head_dim = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
            self.kv_projector = KVProjector(
                self.compressor.mamba_hidden_size,
                c.num_hidden_layers, c.num_key_value_heads, head_dim,
            ).to(device)
            # Trainable trigger token initialized from BOS embedding
            bos_id = c.bos_token_id if c.bos_token_id is not None else c.eos_token_id
            with torch.no_grad():
                bos_embed = self.llm.get_input_embeddings()(
                    torch.tensor([[bos_id]], device=device)).float()  # [1, 1, H]
            self.kv_trigger = nn.Parameter(bos_embed.squeeze(0))  # [1, H]
            self.compressor.projector.requires_grad_(False)
        elif prompt_mode != "embed":
            raise ValueError(f"Unknown prompt_mode {prompt_mode!r}. Use 'embed' or 'kv'.")

    def train(self, mode: bool = True):
        # Keep frozen LLM in eval mode regardless of module train mode
        super().train(mode)
        self.llm.eval()
        return self

    def forward(self, target_inputs: dict, compressor_inputs: dict, return_logits: bool = False):
        if self.prompt_mode == "kv":
            return self._kv_forward(target_inputs, compressor_inputs, return_logits)
        return self._embed_forward(target_inputs, compressor_inputs, return_logits)

    def _embed_forward(self, target_inputs: dict, compressor_inputs: dict, return_logits: bool):
        target_ids = target_inputs["target_ids"]  # [B, G]
        target_mask = target_inputs["target_mask"]  # [B, G]

        soft_prompts, compressed_mask = self.compressor(compressor_inputs)  # [B, C, H], [B, C]

        with torch.no_grad():
            y_embeds = self.llm.get_input_embeddings()(target_ids)  # [B, G, H]
        soft_prompts = soft_prompts.to(y_embeds.dtype)

        combined = torch.cat([soft_prompts, y_embeds], dim=1)  # [B, C+G, H]
        full_padding_mask = torch.cat([compressed_mask, target_mask], dim=1)  # [B, C+G]
        prefix_len = soft_prompts.shape[1]  # C
        gen_len = target_ids.shape[1]  # G

        position_ids = full_padding_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(full_padding_mask == 0, 1)

        geometry = build_causal_mask(prefix_len, gen_len, self.device)
        attention_mask = build_additive_mask(geometry, full_padding_mask, combined.dtype)

        out = self.llm.model(
            inputs_embeds=combined, attention_mask=attention_mask,
            position_ids=position_ids, return_dict=True,
        )
        student_logits = self.llm.lm_head(out.last_hidden_state[:, prefix_len - 1:-1, :])  # [B, G, V]

        loss = lm_ce_loss(student_logits, target_ids, target_mask)
        if return_logits:
            return loss, student_logits
        return loss

    def _kv_forward(self, target_inputs: dict, compressor_inputs: dict, return_logits: bool):
        target_ids = target_inputs["target_ids"]  # [B, G]
        target_mask = target_inputs["target_mask"]  # [B, G]

        hidden, compressed_mask = self.compressor.gather_hidden(compressor_inputs)
        B, C, _ = hidden.shape
        hidden = hidden.to(self.kv_projector.k_proj.weight.dtype)
        per_layer_kv = self.kv_projector(hidden)  # [L, 2, B, kvh, C, hd]
        ref_dtype = next(self.llm.parameters()).dtype
        per_layer_kv = per_layer_kv.to(ref_dtype)

        comp_pos = (compressed_mask.long().cumsum(-1) - 1).clamp_min(0)
        cache = build_kv_cache(per_layer_kv, comp_pos, self.llm.model.rotary_emb,
                               decoder_layers=self.llm.model.layers)

        trigger = self.kv_trigger.to(ref_dtype).unsqueeze(0).expand(B, 1, -1)  # [B, 1, H]
        with torch.no_grad():
            y_embeds = self.llm.get_input_embeddings()(target_ids).to(ref_dtype)  # [B, G, H]
        G = target_ids.shape[1]
        dec_input = torch.cat([trigger, y_embeds], dim=1)  # [B, 1+G, H]

        C_real = compressed_mask.sum(dim=1).long()  # [B]
        dec_mask = torch.cat([target_mask.new_ones((B, 1)), target_mask], dim=1)  # [B, 1+G]
        dec_pos = C_real.unsqueeze(1) + (dec_mask.cumsum(-1) - 1).clamp_min(0)

        attn_mask = torch.cat([compressed_mask, dec_mask], dim=1)

        out = self.llm.model(
            inputs_embeds=dec_input, past_key_values=cache,
            attention_mask=attn_mask, position_ids=dec_pos,
            use_cache=True, return_dict=True,
        )
        student_logits = self.llm.lm_head(out.last_hidden_state[:, 0:G, :])  # [B, G, V]

        loss = lm_ce_loss(student_logits, target_ids, target_mask)
        if return_logits:
            return loss, student_logits
        return loss

