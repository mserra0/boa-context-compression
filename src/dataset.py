import json
import gzip
import random
import os
import glob
import time

import torch.distributed as dist
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
    fast_indexed_column_numpy,
    progress_iter,
    build_token_budget_loader,
)

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
# Safety buffer (tokens) between context-truncation budget and max_length,
# absorbing any tokenization-overhead estimation error.
_TRUNCATION_MARGIN = 4


class CompressionDataset(Dataset):
    def __init__(self, dataset, llm_tokenizer: AutoTokenizer, compressor_tokenizer: AutoTokenizer,
                 max_length: int = 512, context: bool = False, include_answer: bool = False,
                 include_continuation: bool = False,
                 continuation_split_range: tuple = (0.5, 0.9),
                 max_ctx: int = 1024, max_pred: int = 256,
                 student_template: str = "bare", compress_question: bool = True):
        self.dataset = dataset
        self.llm_tokenizer = llm_tokenizer
        self.compressor_tokenizer = compressor_tokenizer
        self.max_length = max_length
        self.context = context
        self.include_answer = include_answer
        self.include_continuation = include_continuation
        # Paper (arXiv:2307.06945) doesn't specify a fixed split point between the
        # compressed span A and continuation span B — sampling per-example avoids
        # overfitting to one fixed boundary position.
        self.continuation_split_range = continuation_split_range
        # include_continuation only: separate caps for the compressed span (X) and the
        # continuation target (Y), mirroring collate_fn_pretrain's max_ctx/max_pred
        # (dataloading.py). Unlike Mamba's adaptive packed compression, ICAE compresses
        # into a FIXED n_mem slot count (n_mem = max_ctx * compression_ratio, set
        # statically in config) -> X should consistently fill max_ctx rather than vary
        # with document length, so the split point is chosen to guarantee a full-length
        # X whenever the document is long enough (see _split_continuation).
        self.max_ctx = max_ctx
        self.max_pred = max_pred
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

        # FineWeb carries a precomputed 'token_count' column (see load_fineweb_dataset) —
        # expose it as token_lengths so DistributedLengthGroupedSampler can pull it in one
        # vectorized read instead of falling back to its row-by-row Python loop over the
        # whole corpus (minutes-to-an-hour+ on the full 100BT FineWeb-EDU sample). MRQA has
        # no such column, so it's left unset and the sampler's existing fallback applies.
        if "token_count" in (getattr(dataset, "column_names", None) or []):
            self.token_lengths = compute_token_lengths(dataset, text_keys=("question", "context"))

    def __len__(self):
        return len(self.dataset)

    def _build_prompt_text(self, context: str, question: str) -> str:
        user_content = f"{question}\n{context}" if context else question
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        # Chat template is Qwen-specific; we always use the LLM's template so that
        # both tokenizers see the same raw string (teacher/student alignment).
        return self.llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def _build_compressor_text(self, context: str, question: str) -> str:
        """Text fed to the compressor.

        - bare + compress_question=False: context ALONE, no question, no chat
          template/system prompt — genuinely question-blind compression (ICAE). The
          question never reaches the compressor's tokens at all; it's reinserted
          downstream as literal decoder-side embeddings (see `question_ids` below).
          This combination is a no-op for the Mamba/PromptDestiller path (bare-mode
          forward never reads compress_question), so it's safe to repurpose here for
          ICAE without touching wrapped-mode (B-qc/B-ctx) behavior below.
        - bare (compress_question=True, the default): full chat template (Design A —
          teacher/student see the same string).
        - wrapped: document content only (question + context for B-qc), no chat template.
                   The template scaffolding is added later as real token embeddings around
                   the soft prompts in the model forward, keeping the frozen LLM in-dist.
                   Note this is NOT question-blind even when compress_question=False
                   (B-ctx): the compressor still sees the question (as "question\ncontext"),
                   just ordered so a causal compressor's state at the context positions is
                   already question-conditioned — see docs/architecture/student-templates.md on the `main` branch.
        """
        if self.student_template == "wrapped":
            return f"{question}\n{context}" if context else question
        if not self.compress_question:
            return context
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

    def _split_continuation(self, comp_ids: list) -> tuple[list, list]:
        """Split compressor-tokenized document ids into a leading span X (compressed,
        exactly max_ctx tokens whenever the document is long enough) and everything
        from the split point onward (the continuation Y is derived from the head of
        this tail downstream, after retokenizing with the LLM tokenizer).

        Unlike the Mamba pretrain collate's `[-max_ctx:]` (which slices AFTER an
        independently-sampled char split and can leave X shorter than max_ctx), the
        split point here is chosen directly in token space so X always ends exactly
        at the split *and* reaches max_ctx tokens whenever possible — ICAE's n_mem is
        a fixed slot count, so an under-filled X wastes compression capacity in a way
        Mamba's adaptive packed compression doesn't.

        continuation_split_range still biases WHERE within the document the split
        falls (as a fraction of total length), so the continuation isn't always
        squeezed against the very end of the document.
        """
        n = len(comp_ids)
        if n <= self.max_ctx:
            return comp_ids, []
        lo_r, hi_r = self.continuation_split_range
        lo = max(self.max_ctx, round(n * lo_r))
        hi = min(n, max(lo + 1, round(n * hi_r)))
        c = random.randint(lo, hi)
        return comp_ids[c - self.max_ctx:c], comp_ids[c:]

    def __getitem__(self, idx):
        if self.include_continuation:
            # Generic-text continuation pretraining: no question/chat-template framing,
            # just a leading span (compressed) and the trailing span that follows it.
            context = self.dataset[idx].get("context", "")
            comp_ids_full = self.compressor_tokenizer(
                context, add_special_tokens=False,
            )["input_ids"]
            x_ids, tail_ids = self._split_continuation(comp_ids_full)

            # Decode only a bounded slice of the tail (not the whole remainder of a
            # long document) so it can be retokenized with the LLM tokenizer — the
            # compressor and LLM vocabs can differ, so token counts aren't comparable
            # across them without going through text. 4x max_pred is a generous
            # safety margin against vocab-size differences inflating token counts.
            tail_text = self.compressor_tokenizer.decode(
                tail_ids[:self.max_pred * 4], skip_special_tokens=True
            ) if tail_ids else ""
            continuation_ids = self.llm_tokenizer(
                tail_text + self.llm_tokenizer.eos_token,
                return_tensors=None, padding=False, truncation=True,
                max_length=self.max_pred, add_special_tokens=False,
            )["input_ids"]

            return {
                # llm_inputs is unused by objective="lm" but kept so collate_fn /
                # train_epoch's unconditional batch["llm_inputs"] access doesn't break.
                "llm_inputs": x_ids,
                "compressor_inputs": {"input_ids": x_ids},
                "continuation_ids": continuation_ids,
            }

        question = self.dataset[idx]["question"]
        if self.context:
            context = self.dataset[idx]["context"]
            # Per-tokenizer context truncation (Mamba and Qwen have different vocabs,
            # so the same context yields different token counts).
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

        # truncation=True is a safety net — the context truncation above should make it a no-op. 
        # However, if it happens it is critical.
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

        result = {"llm_inputs": llm_inputs, "compressor_inputs": compressor_inputs}

        if self.include_answer:
            answer_text = self.dataset[idx].get("answer", "") + self.llm_tokenizer.eos_token
            result["answer_ids"] = self.llm_tokenizer(
                answer_text,
                return_tensors=None,
                padding=False,
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=False,
            )["input_ids"]

        if not self.compress_question:
            # Literal question (Qwen tokens) goes into the student prompt verbatim —
            # both Mamba B-ctx (wrapped) and ICAE's question-blind mode (bare) need this.
            result["question_ids"] = self.llm_tokenizer(
                question, add_special_tokens=False
            )["input_ids"]

        if self.student_template == "wrapped" and not self.compress_question:
            # Mamba B-ctx only: context offset in COMPRESSOR-token space, where context
            # begins after "question\n" in the combined compressor text (see
            # _build_compressor_text's wrapped branch). Meaningless for ICAE's
            # question-blind mode, where the compressor never sees the question at all —
            # left unemitted there rather than carrying bogus offset metadata.
            # Derived by tokenizing the question + separator alone (robust; no string
            # search). comp_context may be "" -> offset can exceed comp length.
            result["question_comp_len"] = len(self.compressor_tokenizer(
                question + "\n", add_special_tokens=False
            )["input_ids"])

        return result


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
                yield {
                    'context': context,
                    'question': qa['question'],
                    'answer': qa['detected_answers'][0]['text']
                }

def load_mrqa_dataset(data_dir: str, seed: int = 42):
    """Loads multiple datasets and interleaves them evenly."""
    print("Loading and flattening individual MRQA datasets...")

    # HFDataset.from_generator hits the same shared HF-cache builder lock as
    # load_fineweb_dataset (see that function's comment): gate to rank 0 so
    # concurrent ranks/nodes don't race to create/lock the generator cache.
    distributed = dist.is_available() and dist.is_initialized()
    if distributed and dist.get_rank() != 0:
        dist.barrier()

    squad = HFDataset.from_generator(mrqa_generator, gen_kwargs={"file_path": f"{data_dir}/{MRQA_FILENAME['mrqa_squad']}"})
    newsqa = HFDataset.from_generator(mrqa_generator, gen_kwargs={"file_path": f"{data_dir}/{MRQA_FILENAME['mrqa_newsqa']}"})
    trivia = HFDataset.from_generator(mrqa_generator, gen_kwargs={"file_path": f"{data_dir}/{MRQA_FILENAME['mrqa_trivia']}"})
    hotpot = HFDataset.from_generator(mrqa_generator, gen_kwargs={"file_path": f"{data_dir}/{MRQA_FILENAME['mrqa_hotpot']}"})

    if distributed and dist.get_rank() == 0:
        dist.barrier()

    print("Blending datasets...")
    blended_dataset = interleave_datasets(
        [squad, newsqa, trivia, hotpot],
        probabilities=[0.25, 0.25, 0.25, 0.25],
        seed=seed,
        stopping_strategy="all_exhausted"
    )
    print(f"Total blended training steps available: {len(blended_dataset)}")
    return blended_dataset


def load_fineweb_dataset(data_dir: str, seed: int = 42, max_samples: int = None):
    """Load FineWeb-EDU parquet shards as a memory-mapped Arrow dataset.

    Map-style + memory-mapped so DistributedSampler and mid-run resume keep working
    (an IterableDataset would break both). Expects `text` and `token_count` columns.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet shards under {data_dir!r}")
    print(f"[fineweb] Found {len(files)} parquet shard(s) under {data_dir!r}")

    # Every rank across every node calls this. Without gating, they all race to
    # create/lock the same builder lock file in the shared HF cache, which on a networked filesystem
    # surfaces as PermissionError instead of clean lock contention. Rank 0 builds
    # the cache first; everyone else waits, then hits the now-warm cache.
    distributed = dist.is_available() and dist.is_initialized()
    if distributed and dist.get_rank() != 0:
        print("[fineweb] Waiting for rank 0 to build/warm the Arrow cache...")
        dist.barrier()
    # First run against a given data_dir has no Arrow cache yet, so this converts
    # every parquet shard to Arrow on disk -- can take a long time (tens of minutes)
    # for a large sample with no console feedback otherwise, since HF's own tqdm bar
    # doesn't reliably render through srun/non-interactive stdout.
    print("[fineweb] Loading parquet shards into an Arrow dataset (first time for this "
          "data_dir builds the cache from scratch and can take a while; subsequent runs reuse it)...")
    t0 = time.time()
    ds = load_dataset("parquet", data_files=files, split="train")
    print(f"[fineweb] Loaded {len(ds)} rows in {time.time() - t0:.1f}s")
    if distributed and dist.get_rank() == 0:
        dist.barrier()

    if max_samples:
        # Same shared-HF-cache race as load_dataset() above: .shuffle().select() writes
        # a fingerprinted indices-mapping file to the same network-filesystem-backed cache. Gate it
        # identically so rank 0 builds it once and everyone else reuses the warm cache
        # instead of racing on the lock.
        if distributed and dist.get_rank() != 0:
            dist.barrier()
        t0 = time.time()
        ds = ds.shuffle(seed=seed).select(range(min(max_samples, len(ds))))
        print(f"[fineweb] max_samples shuffle+select -> {len(ds)} rows in {time.time() - t0:.1f}s")
        if distributed and dist.get_rank() == 0:
            dist.barrier()
    return ds


class PretrainDataset(Dataset):
    """Map-style wrapper over a FineWeb-EDU Arrow dataset for LM pretraining.

    Filters out documents too short to form an X (compress) + Y (predict) pair.
    The random X/Y split is performed later in collate_fn_pretrain so it can be
    re-randomized per pass; __getitem__ only returns the raw text.
    """

    def __init__(self, hf_dataset, min_doc_tokens: int = 320):
        self.dataset = hf_dataset
        n = len(hf_dataset)
        t0 = time.perf_counter()
        if "token_count" in getattr(hf_dataset, "column_names", []):
            # Single columnar read (materializes the token_count column once).
            counts = hf_dataset["token_count"]
            read_dt = time.perf_counter() - t0
            self.indices, self.token_lengths = [], []
            for i, c in enumerate(progress_iter(
                    counts, "Filtering docs (token_count)", total=n)):
                if c >= min_doc_tokens:
                    self.indices.append(i)
                    # Real token_count -> accurate packing budget for the sampler.
                    self.token_lengths.append(int(c))
        else:
            # No token_count column: single columnar text read + char/4 proxy
            # (avoids per-row Arrow random access, which is catastrophically slow).
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
