import json
import gzip
import os
import glob
import time

from colorama import Fore, Style
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_from_disk, interleave_datasets, load_dataset, Dataset as HFDataset

from .utils.dataloading import (
    _build_llm_batch,
    _is_main_process,
    collate_fn,
    collate_fn_mamba2,
    collate_fn_pretrain,
    collate_fn_pretrain_mamba1,
    DistributedLengthGroupedSampler,
    DistributedTokenBudgetSampler,
    compute_token_lengths,
    progress_iter,
    build_token_budget_loader,
)

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
# Safety token buffer absorbing tokenization estimation error during context truncation.
_TRUNCATION_MARGIN = 4


class CompressionDataset(Dataset):
    def __init__(self, dataset, llm_tokenizer: AutoTokenizer, compressor_tokenizer: AutoTokenizer,
                 max_length: int = 512, context: bool = False, student_template: str = "bare",
                 compress_question: bool = True):
        self.dataset = dataset
        self.llm_tokenizer = llm_tokenizer
        self.compressor_tokenizer = compressor_tokenizer
        self.max_length = max_length
        self.context = context
        # "bare": full chat template; "wrapped": document content only.
        self.student_template = student_template
        # If False under wrapped: compress context only, question passes uncompressed.
        self.compress_question = compress_question

        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        if self.compressor_tokenizer.pad_token is None:
            self.compressor_tokenizer.pad_token = self.compressor_tokenizer.eos_token
        self._truncation_warned = set()

    def __len__(self):
        return len(self.dataset)

    def _build_prompt_text(self, context: str, question: str) -> str:
        # Context-first format matching teacher SFT format.
        user_content = f"{context}\n{question}" if context else question
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return self.llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def _build_compressor_text(self, context: str, question: str) -> str:
        """Build input text for compressor according to template mode."""
        if self.student_template == "wrapped":
            return f"{question}\n{context}" if context else question
        return self._build_prompt_text(context, question)

    def _truncate_context_for(self, tokenizer, context: str, question: str,
                              role: str = "compressor", build_text=None) -> str:
        """Trim end of context so built prompt fits max_length under tokenizer."""
        build_text = build_text or self._build_prompt_text
        full_text = build_text(context, question)
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        if len(full_ids) <= self.max_length:
            return context
        excess = len(full_ids) - self.max_length + _TRUNCATION_MARGIN
        ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        keep = len(ctx_ids) - excess
    
        if keep <= 0:
            return ""
        return tokenizer.decode(ctx_ids[:keep], skip_special_tokens=True)

    def __getitem__(self, idx):
        question = self.dataset[idx]["question"]
        if self.context:
            context = self.dataset[idx]["context"]
            llm_context = self._truncate_context_for(self.llm_tokenizer, context, question, role="llm")
            comp_context = self._truncate_context_for(
                self.compressor_tokenizer, context, question, role="compressor",
                build_text=self._build_compressor_text,
            )
        else:
            llm_context = ""
            comp_context = ""

        llm_prompt_text = self._build_prompt_text(llm_context, question)
        comp_prompt_text = self._build_compressor_text(comp_context, question)

        llm_inputs = self.llm_tokenizer(
            llm_prompt_text,
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        compressor_inputs = self.compressor_tokenizer(
            comp_prompt_text,
            return_tensors=None,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        item = {"llm_inputs": llm_inputs, "compressor_inputs": compressor_inputs}
        if self.student_template == "wrapped" and not self.compress_question:
            item["question_ids"] = self.llm_tokenizer(
                question, add_special_tokens=False
            )["input_ids"]
            # Compressor-token context offset ("question\n")
            item["question_comp_len"] = len(self.compressor_tokenizer(
                question + "\n", add_special_tokens=False
            )["input_ids"])
        return item


# In-domain MRQA 2019 training subsets.
MRQA_FILENAME = {
    "mrqa_squad": "SQuAD.jsonl.gz",
    "mrqa_newsqa": "NewsQA.jsonl.gz",
    "mrqa_trivia": "TriviaQA.jsonl.gz",
    "mrqa_hotpot": "HotpotQA.jsonl.gz",
    "mrqa_searchqa": "SearchQA.jsonl.gz",
    "mrqa_naturalq": "NaturalQuestions.jsonl.gz",
}


def mrqa_generator(file_path):
    """Safely reads and flattens a single MRQA gzip file."""
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)

            if 'header' in data or 'context' not in data:
                continue
            
            context = data['context']
            
            for qa in data['qas']:
                if not qa['detected_answers']:
                    continue
                yield {
                    'context': context,
                    'question': qa['question'],
                    'answer': qa['detected_answers'][0]['text']
                }

def load_mrqa_dataset(data_dir: str, seed: int = 42):
    """Load and interleave MRQA subsets with equal weight."""
    print("Loading and flattening individual MRQA datasets...")
    subsets = []
    for key, fname in MRQA_FILENAME.items():
        path = f"{data_dir}/{fname}"
        if not os.path.exists(path):
            raise FileNotFoundError(f"MRQA subset '{key}' not found at {path}")
        subsets.append(HFDataset.from_generator(mrqa_generator, gen_kwargs={"file_path": path}))
    print(f"Loaded {len(subsets)} MRQA subsets: {list(MRQA_FILENAME.keys())}")

    print("Blending datasets...")
    n = len(subsets)
    blended_dataset = interleave_datasets(
        subsets,
        probabilities=[1 / n] * n,
        seed=seed,
        stopping_strategy="all_exhausted"
    )
    print(f"Total blended training steps available: {len(blended_dataset)}")
    return blended_dataset


def load_fineweb_dataset(data_dir: str, seed: int = 42, max_samples: int = None):
    """Load FineWeb-EDU parquet shards as an Arrow dataset."""
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet shards under {data_dir!r}")
    ds = load_dataset("parquet", data_files=files, split="train")
    if max_samples:
        ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))
    return ds


class PretrainDataset(Dataset):
    """Dataset wrapper for FineWeb-EDU pretraining with length filtering."""

    def __init__(self, hf_dataset, min_doc_tokens: int = 320):
        self.dataset = hf_dataset
        n = len(hf_dataset)
        t0 = time.perf_counter()
        if "token_count" in getattr(hf_dataset, "column_names", []):
            counts = hf_dataset["token_count"]
            read_dt = time.perf_counter() - t0
            self.indices, self.token_lengths = [], []
            for i, c in enumerate(progress_iter(
                    counts, "Filtering docs (token_count)", total=n)):
                if c >= min_doc_tokens:
                    self.indices.append(i)
                    self.token_lengths.append(int(c))
        else:
            texts = hf_dataset["text"]
            read_dt = time.perf_counter() - t0
            self.indices, self.token_lengths = [], []
            for i, t in enumerate(progress_iter(
                    texts, "Filtering docs (text len)", total=n)):
                if len(t) >= min_doc_tokens:
                    self.indices.append(i)
                    self.token_lengths.append(len(t) // 4)

        if _is_main_process():
            print(f"{Fore.GREEN}[PretrainDataset] {len(self.indices):,}/{n:,} docs "
                  f"kept (>= {min_doc_tokens} tok) | column read {read_dt:.1f}s | "
                  f"total {time.perf_counter() - t0:.1f}s{Style.RESET_ALL}", flush=True)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return {"text": self.dataset[self.indices[idx]]["text"]}

