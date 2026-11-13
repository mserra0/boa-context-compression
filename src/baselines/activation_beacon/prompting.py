# Adapted from Activation Beacon (https://github.com/FlagOpen/FlagEmbedding, research/Long_LLM/activation_beacon),
# Copyright (c) 2022 staoxiao, MIT License. Modified for the Boa baseline reproduction; see THIRD_PARTY_NOTICES.md.
"""Prompt construction shared by Activation Beacon inference and tests."""

from __future__ import annotations

import torch

from ...dataset import SYSTEM_PROMPT


def _render(tokenizer, question: str, context: str) -> str:
    user_content = f"{question}\n{context}" if context else question
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_qa_inputs(tokenizer, question: str, context: str, max_input_tokens: int) -> dict:
    """Build an unpadded Qwen chat prompt, truncating only the context tail."""
    if max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")

    def tokenize_context(candidate_context: str) -> list[int]:
        rendered = _render(tokenizer, question, candidate_context)
        return tokenizer(rendered, add_special_tokens=False)["input_ids"]

    input_ids = tokenize_context(context)
    if len(input_ids) > max_input_tokens and context:
        context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        low, high = 0, len(context_ids)
        best: list[int] | None = None
        while low <= high:
            keep = (low + high) // 2
            candidate = tokenizer.decode(context_ids[:keep], skip_special_tokens=True)
            candidate_ids = tokenize_context(candidate)
            if len(candidate_ids) <= max_input_tokens:
                best = candidate_ids
                low = keep + 1
            else:
                high = keep - 1
        if best is not None:
            input_ids = best

    if len(input_ids) > max_input_tokens:
        raise ValueError(
            "The question and chat template alone exceed max_input_tokens="
            f"{max_input_tokens}; increase the input budget."
        )

    ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
