"""SFT dataset for the Qwen3 FullFT baseline: prompt masked (-100), answer + eos supervised."""
from functools import partial

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.dataset import SYSTEM_PROMPT, load_mrqa_dataset

# Tokenizer-overhead safety buffer between the prompt+answer budget and max_length.
_TRUNCATION_MARGIN = 4


def build_prompt_text(tokenizer: AutoTokenizer, context: str, question: str,
                      enable_thinking: bool = True,
                      answer_instruction: str | None = None) -> str:
    """Qwen chat-template prompt (context before question), with the assistant cue.

    ``enable_thinking`` defaults to True to preserve the SFT training prompt. Set it
    False when evaluating a non-fine-tuned Qwen3 model so it answers directly instead
    of spending the generation budget on a <think> reasoning block.

    ``answer_instruction`` (eval-only) appends a formatting directive to the system
    message so a zero-shot, non-fine-tuned model emits a bare answer span instead of
    a full-sentence explanation. Leave None to preserve the exact SFT training prompt
    (required when evaluating the fine-tuned model).
    """
    system_content = SYSTEM_PROMPT
    if answer_instruction:
        system_content = f"{SYSTEM_PROMPT}\n\n{answer_instruction}"
    user_content = f"{context}\n{question}" if context else question
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def build_sft_example(prompt_ids, answer_ids, max_length: int):
    """Prompt + answer (eos included); labels -100 over prompt. Right-truncated. Pure."""
    input_ids = list(prompt_ids) + list(answer_ids)
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    return input_ids[:max_length], labels[:max_length]


def _truncate_context(tokenizer, context: str, question: str,
                      answer_len: int, max_length: int,
                      answer_instruction: str | None = None) -> str:
    """Trim the END of the context so that prompt + answer + eos fits in max_length."""
    prompt_text = build_prompt_text(tokenizer, context, question,
                                    answer_instruction=answer_instruction)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    budget = max_length - answer_len - _TRUNCATION_MARGIN
    if len(prompt_ids) <= budget:
        return context
    excess = len(prompt_ids) - budget
    ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    keep = len(ctx_ids) - excess
    if keep <= 0:
        return ""
    return tokenizer.decode(ctx_ids[:keep], skip_special_tokens=True)


def collate_fn_sft(batch, pad_id: int, max_length: int):
    """Right-pad {input_ids, labels} to batch max; mask 0 and labels -100 over pad."""
    max_len = min(max(len(ex["input_ids"]) for ex in batch), max_length)
    input_ids, labels, attention_mask = [], [], []
    for ex in batch:
        ids = ex["input_ids"][:max_len]
        lab = ex["labels"][:max_len]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


class MRQASFTDataset(Dataset):
    """MRQA examples framed as causal-LM SFT for full fine-tuning a Qwen3 teacher."""

    def __init__(self, dataset, tokenizer: AutoTokenizer, max_length: int = 1024,
                 use_context: bool = True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_context = use_context
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        question = sample["question"]
        answer = sample["answer"]

        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        answer_ids = answer_ids + [self.tokenizer.eos_token_id]

        context = sample["context"] if self.use_context else ""
        if context:
            context = _truncate_context(
                self.tokenizer, context, question, len(answer_ids), self.max_length,
            )

        prompt_text = build_prompt_text(self.tokenizer, context, question)
        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        input_ids, labels = build_sft_example(prompt_ids, answer_ids, self.max_length)
        return {"input_ids": input_ids, "labels": labels}


def build_sft_collate(tokenizer, max_length: int):
    """Bind tokenizer pad id + max_length into a Trainer-compatible data collator."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return partial(collate_fn_sft, pad_id=pad_id, max_length=max_length)
