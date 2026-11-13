import torch
import torch.nn.functional as F
from pathlib import Path

from .inference_utils import _build_aligned_batch, load_resources
from ..dataset import SYSTEM_PROMPT

# Safety buffer margin when trimming context for teacher prompt
_TEACHER_TRUNCATION_MARGIN = 4

class ModelRunner:
    """Evaluation runner managing model loading, batch inference, and generation."""

    def __init__(self, experiment_dir: str, device: str = "cuda"):
        exp_path = Path(experiment_dir)
        ckpt_path = exp_path / "best_model.pt"
        config_path = exp_path / "config.yaml"

        if not ckpt_path.exists():
            raise FileNotFoundError(f"best_model.pt not found in {exp_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found in {exp_path}")

        self.device = torch.device(device)
        self.run_name = exp_path.resolve().name

        self.model, self.llm_tokenizer, self.compressor_tokenizer, self.config, self.compressor_type = (
            load_resources(str(config_path), str(ckpt_path), device)
        )
        self.llm_name_safe = self.config["llm_model_name"].split("/")[-1]
        self._llm_eval = getattr(self.model.llm, "_orig_mod", self.model.llm)
        self._llm_dtype = next(self._llm_eval.parameters()).dtype
        self.amp_eval = bool(self.config.get("amp_eval", False))

    def _autocast(self):
        """Autocast context matching the training run's validation pass."""
        return torch.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp_eval
        )

    def _run_config(self) -> tuple[int, int, bool, int]:
        """Return (max_seq_len, max_new_tokens, use_context, pad_token_id)."""
        max_seq_len = self.config.get("max_seq_len", 512)
        max_new_tokens = self.config["max_new_tokens"]
        use_context = self.config.get("use_context", True)
        _pid = self._llm_eval.config.pad_token_id
        pad_token_id = _pid if _pid is not None else self._llm_eval.config.eos_token_id
        return max_seq_len, max_new_tokens, use_context, pad_token_id

    def _teacher_prompt_ids(self, context: str, question: str, max_seq_len: int):
        """Build aligned teacher prompt IDs tensor [1, L]."""
        def render(ctx: str) -> str:
            user_content = f"{ctx}\n{question}" if ctx else question
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
            return self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

        ids = self.llm_tokenizer(render(context), add_special_tokens=False)["input_ids"]
        if context and len(ids) > max_seq_len:
            excess = len(ids) - max_seq_len + _TEACHER_TRUNCATION_MARGIN
            ctx_ids = self.llm_tokenizer(context, add_special_tokens=False)["input_ids"]
            keep = len(ctx_ids) - excess
            context = self.llm_tokenizer.decode(
                ctx_ids[:keep], skip_special_tokens=True
            ) if keep > 0 else ""
            ids = self.llm_tokenizer(render(context), add_special_tokens=False)["input_ids"]
        ids = ids[:max_seq_len]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    @torch.no_grad()
    def run_teacher(self, samples: list, use_context: bool = True,
                    batch_size: int = 8, progress=None, task_id=None) -> list:
        """Run teacher LLM generation on samples."""
        answers = []
        max_seq_len, max_new_tokens, _, pad_token_id = self._run_config()

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            all_ids, all_masks = [], []

            for sample in batch:
                context = sample.get("context", "") if use_context else ""
                ids = self._teacher_prompt_ids(context, sample["question"], max_seq_len)
                all_ids.append(ids)
                all_masks.append(torch.ones_like(ids))

            batch_max = max(x.shape[1] for x in all_ids)
            input_ids = torch.cat(
                [F.pad(x, (batch_max - x.shape[1], 0), value=pad_token_id) for x in all_ids], dim=0
            ).to(self.device)
            attention_mask = torch.cat(
                [F.pad(x, (batch_max - x.shape[1], 0), value=0) for x in all_masks], dim=0
            ).to(self.device)

            with self._autocast():
                out = self._llm_eval.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )  # [B, batch_max + gen_len]

            for row in out:
                answers.append(
                    self.llm_tokenizer.decode(row[batch_max:], skip_special_tokens=True)
                )

            if progress is not None and task_id is not None:
                progress.update(task_id, advance=len(batch))

        return answers

    @torch.no_grad()
    def run_student(self, samples: list, batch_size: int = 8,
                    progress=None, task_id=None) -> list:
        """Run student compressor + LLM generation on samples."""
        with self._autocast():
            if self.compressor_type == "mamba2":
                return self._run_student_mamba2(samples, batch_size, progress, task_id)
            return self._run_student_mamba1(samples, batch_size, progress, task_id)

    @torch.no_grad()
    def _run_student_mamba1(self, samples: list, batch_size: int,
                             progress=None, task_id=None) -> list:
        """Batched Mamba1 student inference."""
        answers = []
        max_seq_len, max_new_tokens, use_context, pad_token_id = self._run_config()
        _cpid = self.compressor_tokenizer.pad_token_id
        comp_pad_id = _cpid if _cpid is not None else self.compressor_tokenizer.eos_token_id
        pad_embed = self._llm_eval.get_input_embeddings()(
            torch.tensor(pad_token_id, device=self.device)
        )

        compress_question = getattr(self.model, "compress_question", True)

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            all_comp_ids, all_comp_masks = [], []
            all_context_starts, all_question_ids = [], []

            for sample in batch:
                context = sample.get("context", "") if use_context else ""
                b = _build_aligned_batch(
                    context_text=context,
                    question_text=sample["question"],
                    llm_tokenizer=self.llm_tokenizer,
                    compressor_tokenizer=self.compressor_tokenizer,
                    max_seq_len=max_seq_len,
                    use_context=use_context and bool(context),
                    compressor_type=self.compressor_type,
                    student_template=getattr(self.model, "student_template", "bare"),
                    compress_question=compress_question,
                )
                comp_in = {
                    k: (v.to(self.device) if torch.is_tensor(v) else v)
                    for k, v in b["compressor_inputs"].items()
                }
                all_comp_ids.append(comp_in["input_ids"])
                all_comp_masks.append(comp_in["attention_mask"])
                if "context_start" in comp_in:
                    all_context_starts.append(comp_in["context_start"])
                if "question_ids" in comp_in:
                    all_question_ids.extend(comp_in["question_ids"])

            batch_max = max(x.shape[1] for x in all_comp_ids)
            comp_ids = torch.cat(
                [F.pad(x, (batch_max - x.shape[1], 0), value=comp_pad_id) for x in all_comp_ids], dim=0
            ).to(self.device)
            comp_masks = torch.cat(
                [F.pad(x, (batch_max - x.shape[1], 0), value=0) for x in all_comp_masks], dim=0
            ).to(self.device)

            compressor_inputs = {
                'input_ids': comp_ids,
                'attention_mask': comp_masks,
            }
            if all_context_starts:
                shifted = []
                for cs, ids in zip(all_context_starts, all_comp_ids):
                    shift = batch_max - ids.shape[1]
                    shifted.append(cs + shift)
                compressor_inputs['context_start'] = torch.cat(shifted, dim=0).to(self.device)
            if all_question_ids and not compress_question:
                compressor_inputs['question_ids'] = all_question_ids

            if getattr(self.model, "prompt_mode", "embed") == "kv":
                out = self.model.generate_compressed_kv(
                    compressor_inputs, max_new_tokens=max_new_tokens,
                )  # [B, G]
                for row in out:
                    answers.append(
                        self.llm_tokenizer.decode(row, skip_special_tokens=True)
                    )
                if progress is not None and task_id is not None:
                    progress.update(task_id, advance=len(batch))
                continue

            soft_prompts, compressed_mask = self.model.compressor(compressor_inputs)
            soft_prompts, compressed_mask = self.model._replace_pad_soft_prompts(
                soft_prompts, compressed_mask, pad_embed
            )
            soft_prompts = soft_prompts.to(self._llm_dtype)

            if getattr(self.model, "student_template", "bare") == "wrapped":
                qids = all_question_ids if (not compress_question and all_question_ids) else None
                out = self.model.generate_compressed(
                    soft_prompts,
                    compressed_mask=compressed_mask,
                    max_new_tokens=max_new_tokens,
                    question_ids=qids,
                )  # [B, G]
            else:
                out = self._llm_eval.generate(
                    inputs_embeds=soft_prompts,
                    attention_mask=compressed_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_token_id,
                )  # [B, gen_len]

            for row in out:
                answers.append(
                    self.llm_tokenizer.decode(row, skip_special_tokens=True)
                )

            if progress is not None and task_id is not None:
                progress.update(task_id, advance=len(batch))

        return answers

    @torch.no_grad()
    def _run_student_mamba2(self, samples: list, batch_size: int,
                             progress=None, task_id=None) -> list:
        """Batched Mamba2 student inference."""
        answers = []
        max_seq_len, max_new_tokens, use_context, pad_token_id = self._run_config()
        compress_question = getattr(self.model, "compress_question", True)

        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            real_ids_per_sample = []
            all_context_starts, all_question_ids = [], []

            for sample in batch:
                context = sample.get("context", "") if use_context else ""
                b = _build_aligned_batch(
                    context_text=context,
                    question_text=sample["question"],
                    llm_tokenizer=self.llm_tokenizer,
                    compressor_tokenizer=self.compressor_tokenizer,
                    max_seq_len=max_seq_len,
                    use_context=use_context and bool(context),
                    compressor_type=self.compressor_type,
                    student_template=getattr(self.model, "student_template", "bare"),
                    compress_question=compress_question,
                )
                comp_in = {
                    k: (v.to(self.device) if torch.is_tensor(v) else v)
                    for k, v in b["compressor_inputs"].items()
                }
                comp_ids = comp_in["input_ids"][0]
                real_ids_per_sample.append(comp_ids.tolist())
                if "context_start" in comp_in:
                    all_context_starts.append(comp_in["context_start"])
                if "question_ids" in comp_in:
                    all_question_ids.extend(comp_in["question_ids"])

            packed, seq_idx_list, lengths = [], [], []
            for j, ids in enumerate(real_ids_per_sample):
                packed.extend(ids)
                seq_idx_list.extend([j] * len(ids))
                lengths.append(len(ids))

            packed_ids = torch.tensor(packed, dtype=torch.long).unsqueeze(0).to(self.device)
            seq_idx = torch.tensor(seq_idx_list, dtype=torch.int32).unsqueeze(0).to(self.device)
            lengths_t = torch.tensor(lengths, dtype=torch.long).to(self.device)

            compressor_inputs = {
                'input_ids': packed_ids,
                'seq_idx': seq_idx,
                'lengths': lengths_t,
            }
            if all_context_starts:
                compressor_inputs['context_start'] = torch.cat(all_context_starts, dim=0).to(self.device)
            if all_question_ids and not compress_question:
                compressor_inputs['question_ids'] = all_question_ids

            if getattr(self.model, "prompt_mode", "embed") == "kv":
                out = self.model.generate_compressed_kv(
                    compressor_inputs, max_new_tokens=max_new_tokens,
                )  # [B, G]
                for row in out:
                    answers.append(
                        self.llm_tokenizer.decode(row, skip_special_tokens=True)
                    )
                if progress is not None and task_id is not None:
                    progress.update(task_id, advance=len(batch))
                continue

            soft_prompts, compressed_mask = self.model.compressor(compressor_inputs)
            soft_prompts = soft_prompts.to(self._llm_dtype)

            qids = all_question_ids if (not compress_question and all_question_ids) else None
            out = self.model.generate_compressed(
                soft_prompts,
                compressed_mask=compressed_mask,
                max_new_tokens=max_new_tokens,
                question_ids=qids,
            )  # [B, G]

            for row in out:
                answers.append(
                    self.llm_tokenizer.decode(row, skip_special_tokens=True)
                )

            if progress is not None and task_id is not None:
                progress.update(task_id, advance=len(batch))

        return answers