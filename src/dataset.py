import json
import gzip
import math
from typing import NamedTuple

import numpy as np
import torch
import torch.distributed as dist
from colorama import Fore
from torch.utils.data import Dataset, Sampler
from transformers import AutoTokenizer
from datasets import load_from_disk, interleave_datasets, Dataset as HFDataset

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
# Safety buffer (tokens) between context-truncation budget and max_length,
# absorbing any tokenization-overhead estimation error.
_TRUNCATION_MARGIN = 4


class EncodedQwenMRQAExample(NamedTuple):
    """Shared Qwen representation for MRQA prompting and answer-supervised SFT."""

    input_ids: list[int]
    prompt_input_ids: list[int]
    labels: list[int] | None
    context: str


def build_qwen_mrqa_messages(
    question: str,
    context: str = "",
    answer: str | None = None,
) -> list[dict[str, str]]:
    """Build the canonical Qwen chat messages used by every MRQA pipeline."""
    user_content = f"{question}\n{context}" if context else question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def _chat_template_ids(tokenizer, messages, *, add_generation_prompt: bool) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(ids, torch.Tensor):
        ids = ids.squeeze(0).tolist()
    # Some light-weight test/fallback tokenizers ignore ``tokenize=True``.
    if isinstance(ids, str):
        ids = tokenizer(ids, add_special_tokens=False)["input_ids"]
    return list(ids)


def encode_qwen_mrqa_example(
    tokenizer,
    sample: dict,
    *,
    max_length: int,
    include_answer: bool,
    include_context: bool = True,
    context_ids: list[int] | None = None,
) -> EncodedQwenMRQAExample | None:
    """Encode MRQA once for either generation or answer-supervised fine-tuning.

    Context is truncated only from its tail. In answer-supervised mode, room is
    reserved for the complete assistant answer and labels mask every prompt token.
    """
    question = sample["question"]
    answer = sample.get("answer") if include_answer else None
    raw_context = sample.get("context", "") if include_context else ""
    if include_answer and answer is None:
        raise ValueError("Answer-supervised MRQA encoding requires sample['answer'].")
    if context_ids is None:
        context_ids = tokenizer(raw_context, add_special_tokens=False)["input_ids"]
    if not include_context:
        context_ids = []

    def encode_with_context(keep: int):
        context = tokenizer.decode(context_ids[:keep], skip_special_tokens=True)
        prompt_messages = build_qwen_mrqa_messages(question, context)
        prompt_ids = _chat_template_ids(
            tokenizer, prompt_messages, add_generation_prompt=True
        )
        if not include_answer:
            return prompt_ids, prompt_ids, None, context

        full_ids = _chat_template_ids(
            tokenizer,
            build_qwen_mrqa_messages(question, context, answer),
            add_generation_prompt=False,
        )
        prefix_length = 0
        for prompt_token, full_token in zip(prompt_ids, full_ids):
            if prompt_token != full_token:
                break
            prefix_length += 1
        labels = [-100] * prefix_length + full_ids[prefix_length:]
        return full_ids, prompt_ids, labels, context

    empty = encode_with_context(0)
    if len(empty[0]) > max_length:
        if include_answer:
            return None
        # Preserve CompressionDataset's historical tokenizer safety net for an
        # overlong question/template. SFT never clips here because the answer is
        # the supervised target and must remain complete.
        clipped = empty[0][:max_length]
        return EncodedQwenMRQAExample(
            input_ids=clipped,
            prompt_input_ids=clipped,
            labels=None,
            context="",
        )
    keep = min(len(context_ids), max_length - len(empty[0]))
    encoded = encode_with_context(keep)
    while len(encoded[0]) > max_length and keep > 0:
        keep = max(0, keep - (len(encoded[0]) - max_length) - _TRUNCATION_MARGIN)
        encoded = encode_with_context(keep)
    if len(encoded[0]) > max_length:
        return None

    input_ids, prompt_ids, labels, context = encoded
    if labels is not None and not any(label != -100 for label in labels):
        return None
    return EncodedQwenMRQAExample(
        input_ids=input_ids,
        prompt_input_ids=prompt_ids,
        labels=labels,
        context=context,
    )


class CompressionDataset(Dataset):
    def __init__(self, dataset, llm_tokenizer: AutoTokenizer, compressor_tokenizer: AutoTokenizer,
                 max_length: int = 512, context: bool = False, student_template: str = "bare",
                 compress_question: bool = True):
        self.dataset = dataset
        self.llm_tokenizer = llm_tokenizer
        self.compressor_tokenizer = compressor_tokenizer
        self.max_length = max_length
        self.context = context
        # "bare"    → Design A: compressor encodes the full Qwen chat template.
        # "wrapped" → Design B-qc: compressor encodes document content only; the model
        #             wraps the chat template around the soft prompts as token embeds.
        self.student_template = student_template
        # B-ctx (wrapped only): when False, the question is NOT compressed — it enters the
        # LLM prompt verbatim, only the context is compressed. __getitem__ then also emits
        # "question_ids" and "question_comp_len" (compressor-token context offset).
        self.compress_question = compress_question

        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        if self.compressor_tokenizer.pad_token is None:
            self.compressor_tokenizer.pad_token = self.compressor_tokenizer.eos_token
        # For controll warning of truncation.
        self._truncation_warned = set()

    def __len__(self):
        return len(self.dataset)

    def _build_prompt_text(self, context: str, question: str) -> str:
        # Chat template is Qwen-specific; both model families share the canonical
        # message builder so benchmark prompts cannot drift.
        return self.llm_tokenizer.apply_chat_template(
            build_qwen_mrqa_messages(question, context),
            tokenize=False,
            add_generation_prompt=True,
        )

    def _build_compressor_text(self, context: str, question: str) -> str:
        """Text fed to the compressor.

        - bare:    full chat template (Design A — teacher/student see the same string).
        - wrapped: document content only (question + context for B-qc), no chat template.
                   The template scaffolding is added later as real token embeddings around
                   the soft prompts in the model forward, keeping the frozen LLM in-dist.
        """
        if self.student_template == "wrapped":
            return f"{question}\n{context}" if context else question
        return self._build_prompt_text(context, question)

    def _truncate_context_for(self, tokenizer, context: str, question: str,
                              role: str = "compressor", build_text=None) -> str:
        """Trim the END of context so that the built prompt fits in max_length under
        `tokenizer`. Ensures question + assistant-prompt marker always survive.

        `build_text` selects the text whose length is budgeted (default: full template).
        The wrapped compressor budgets against its content-only text instead.
        """
        build_text = build_text or self._build_prompt_text
        full_text = build_text(context, question)
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        if len(full_ids) <= self.max_length:
            return context
        # !NOTE Warning abot the trucation is very importat, but if we do it here it floads the stdout and make the dataloader block.
        excess = len(full_ids) - self.max_length + _TRUNCATION_MARGIN
        ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        keep = len(ctx_ids) - excess
    
        if keep <= 0:
            return ""
        return tokenizer.decode(ctx_ids[:keep], skip_special_tokens=True)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        question = sample["question"]
        context = sample.get("context", "") if self.context else ""
        llm_encoded = encode_qwen_mrqa_example(
            self.llm_tokenizer,
            sample,
            max_length=self.max_length,
            include_answer=False,
            include_context=self.context,
        )
        if llm_encoded is None:
            raise ValueError(
                "Question and Qwen chat-template overhead exceed dataset max_length."
            )
        llm_inputs = llm_encoded.input_ids

        # The compressor may use a different tokenizer and representation, so it
        # still needs its own context budget. The Qwen side above is fully shared
        # with Activation Beacon fine-tuning.
        comp_context = self._truncate_context_for(
            self.compressor_tokenizer,
            context,
            question,
            role="compressor",
            build_text=self._build_compressor_text,
        ) if self.context else ""
        comp_prompt_text = self._build_compressor_text(comp_context, question)

        compressor_inputs = self.compressor_tokenizer(
            comp_prompt_text,
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        item = {"llm_inputs": llm_inputs, "compressor_inputs": compressor_inputs}
        if self.student_template == "wrapped" and not self.compress_question:
            # Literal question (Qwen tokens) goes into the student prompt verbatim.
            item["question_ids"] = self.llm_tokenizer(
                question, add_special_tokens=False
            )["input_ids"]
            # Context offset in COMPRESSOR-token space: where context begins after
            # "question\n". Derived by tokenizing the question + separator alone (robust;
            # no string search). comp_context may be "" -> offset can exceed comp length.
            item["question_comp_len"] = len(self.compressor_tokenizer(
                question + "\n", add_special_tokens=False
            )["input_ids"])
        return item
    
def _build_llm_batch(
    llm_inputs_list: list,
    llm_tokenizer,
    max_seq_len: int,
) -> dict:
    """Left-pad LLM input sequences to the longest in the batch (up to max_seq_len).

    Returns:
        dict with 'input_ids' [B, L] and 'attention_mask' [B, L].
    """
    clipped = [ids[:max_seq_len] for ids in llm_inputs_list]
    batch_max = max(len(ids) for ids in clipped) if clipped else 0

    input_ids, attention_masks = [], []
    for ids in clipped:
        pad_len = batch_max - len(ids)
        input_ids.append([llm_tokenizer.pad_token_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }


def collate_fn(batch, llm_tokenizer, compressor_tokenizer, max_length=256):
    """Left-padded collate for Mamba. Both LLM and compressor inputs are left-padded."""
    llm_inputs_list = [item["llm_inputs"] for item in batch]
    compressor_inputs_list = [item["compressor_inputs"] for item in batch]

    llm_batch = _build_llm_batch(llm_inputs_list, llm_tokenizer, max_length)

    comp_ids_list = [item['input_ids'][:max_length] for item in compressor_inputs_list]
    batch_comp_max = max(len(ids) for ids in comp_ids_list) if comp_ids_list else 0

    comp_input_ids, comp_attention_masks = [], []
    for ids in comp_ids_list:
        pad_len = batch_comp_max - len(ids)
        comp_input_ids.append([compressor_tokenizer.pad_token_id] * pad_len + ids)
        comp_attention_masks.append([0] * pad_len + [1] * len(ids))

    out = {
        "llm_inputs": llm_batch,
        "compressor_inputs": {
            "input_ids": torch.tensor(comp_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(comp_attention_masks, dtype=torch.long),
        },
    }

    if batch and "question_comp_len" in batch[0]:
        # B-ctx: attach the absolute (padded-tensor) index where the context
        # begins per row, plus the literal (un-tensorized) question token ids.
        comp_real_len = torch.tensor([len(ids) for ids in comp_ids_list])
        first_real = batch_comp_max - comp_real_len
        qlen = torch.tensor([item["question_comp_len"] for item in batch])
        context_start = (first_real + torch.minimum(qlen, comp_real_len)).long()
        out["compressor_inputs"]["context_start"] = context_start
        out["compressor_inputs"]["question_ids"] = [item["question_ids"] for item in batch]

    return out


def collate_fn_mamba2(batch, llm_tokenizer, compressor_tokenizer, max_length=256):
    """Packed (padding-free) collate for Mamba2.

    The compressor receives a single flat tensor of concatenated token IDs [1, T]
    plus a seq_idx tensor [1, T] that tells Mamba2 where each sequence starts/ends
    so the SSM state is reset at boundaries. No padding tokens reach the compressor.

    Returns:
        llm_inputs:
            input_ids      [B, L]   left-padded
            attention_mask [B, L]
        compressor_inputs:
            input_ids      [1, T]   packed
            seq_idx        [1, T]   int32
            lengths        [B]      int64
    """
    llm_inputs_list = [item["llm_inputs"] for item in batch]
    compressor_inputs_list = [item["compressor_inputs"] for item in batch]

    llm_batch = _build_llm_batch(llm_inputs_list, llm_tokenizer, max_length)

    packed_ids, seq_idx_list, lengths = [], [], []
    for i, item in enumerate(compressor_inputs_list):
        ids = item['input_ids'][:max_length]
        packed_ids.extend(ids)
        seq_idx_list.extend([i] * len(ids))
        lengths.append(len(ids))

    out = {
        "llm_inputs": llm_batch,
        "compressor_inputs": {
            "input_ids": torch.tensor(packed_ids, dtype=torch.long).unsqueeze(0),
            "seq_idx": torch.tensor(seq_idx_list, dtype=torch.int32).unsqueeze(0),
            "lengths": torch.tensor(lengths, dtype=torch.long),
        },
    }

    if batch and "question_comp_len" in batch[0]:
        # B-ctx: per-segment context offset (clamped to that segment's packed
        # length), plus the literal (un-tensorized) question token ids.
        context_starts = [
            min(item["question_comp_len"], L) for item, L in zip(batch, lengths)
        ]
        out["compressor_inputs"]["context_start"] = torch.tensor(context_starts, dtype=torch.long)
        out["compressor_inputs"]["question_ids"] = [item["question_ids"] for item in batch]

    return out

class DistributedLengthGroupedSampler(Sampler):
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, seed=0, shuffle=True):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()
                
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.shuffle = shuffle
        
        self.lengths = []
        for i in range(len(dataset)):
            item = dataset.dataset[i] 
            l = len(item.get("context", "")) + len(item.get("question", ""))
            self.lengths.append(l)
            
        self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        indices = np.argsort(self.lengths).tolist()
        
        if self.shuffle:
            # Group into megabatches
            megabatch_size = self.batch_size * self.num_replicas * 10
            megabatches = [indices[i:i + megabatch_size] for i in range(0, len(indices), megabatch_size)]
            
            # Shuffle megabatches
            megabatch_indices = torch.randperm(len(megabatches), generator=g).tolist()
            indices = []
            for i in megabatch_indices:
                mb = megabatches[i]
                if len(mb) > 0:
                    mb_perm = torch.randperm(len(mb), generator=g).tolist()
                    indices.extend([mb[j] for j in mb_perm])
        
        padding_size = self.total_size - len(indices)
        if padding_size > 0:
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
            
        assert len(indices) == self.total_size
        
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


MRQA_FILENAME = {
    "mrqa_squad": "SQuAD.jsonl.gz",
    "mrqa_newsqa": "NewsQA.jsonl.gz",
    "mrqa_trivia": "TriviaQA.jsonl.gz",
    "mrqa_hotpot": "HotpotQA.jsonl.gz",
}


def mrqa_generator(file_path):
    """Safely reads and flattens a single MRQA gzip file."""
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            if 'header' in data or 'context' not in data:
                continue
            
            context = data['context']
            
            # Flatten the nested questions
            for qa in data['qas']:
                if not qa['detected_answers']:
                    continue
                answers = [
                    answer['text']
                    for answer in qa['detected_answers']
                    if answer.get('text')
                ]
                if not answers:
                    continue
                yield {
                    'context': context,
                    'question': qa['question'],
                    # Keep the first alias as the canonical supervised target for
                    # training, but preserve all aliases for LongBench-style QA-F1.
                    'answer': answers[0],
                    'answers': answers,
                }

def load_mrqa_dataset(data_dir: str, seed: int = 42):
    """Loads multiple datasets and interleaves them evenly."""
    print("Loading and flattening individual MRQA datasets...")
    
    squad = HFDataset.from_generator(mrqa_generator, gen_kwargs={"file_path": f"{data_dir}/{MRQA_FILENAME['mrqa_squad']}"})
    newsqa = HFDataset.from_generator(mrqa_generator, gen_kwargs={"file_path": f"{data_dir}/{MRQA_FILENAME['mrqa_newsqa']}"})
    trivia = HFDataset.from_generator(mrqa_generator, gen_kwargs={"file_path": f"{data_dir}/{MRQA_FILENAME['mrqa_trivia']}"})
    hotpot = HFDataset.from_generator(mrqa_generator, gen_kwargs={"file_path": f"{data_dir}/{MRQA_FILENAME['mrqa_hotpot']}"})
    
    print("Blending datasets...")
    blended_dataset = interleave_datasets(
        [squad, newsqa, trivia, hotpot],
        probabilities=[0.25, 0.25, 0.25, 0.25],
        seed=seed,
        stopping_strategy="all_exhausted"
    )
    print(f"Total blended training steps available: {len(blended_dataset)}")
    return blended_dataset
