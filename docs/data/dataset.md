# Dataset Schemas and Preprocessing

## Overview

Boa supports two dataset pipelines:
1. **MRQA Shared Task 2019:** Extractive Question Answering dataset used for knowledge distillation and evaluation.
2. **FineWeb-EDU:** Large-scale educational web corpus used for continuation-LM pretraining (`PretrainDataset`).

---

## MRQA Shared Task 2019 Dataset

### Subsets and File Mapping

Six in-domain MRQA subsets are defined in `MRQA_FILENAME` (`src/dataset.py`):

| Subset Key | File Name | Domain |
|---|---|---|
| `mrqa_squad` | `SQuAD.jsonl.gz` | Wikipedia |
| `mrqa_newsqa` | `NewsQA.jsonl.gz` | News articles |
| `mrqa_trivia` | `TriviaQA.jsonl.gz` | Web / Trivia |
| `mrqa_hotpot` | `HotpotQA.jsonl.gz` | Multi-hop Wikipedia |
| `mrqa_searchqa` | `SearchQA.jsonl.gz` | Web search snippets |
| `mrqa_naturalq` | `NaturalQuestions.jsonl.gz` | Google search queries |

Total raw in-domain train split contains ~517K QA samples across gzip files under `data_dir`.

### Raw JSONL Schema and Generator (`mrqa_generator`)

Each `.jsonl.gz` file contains header metadata on line 1 followed by JSON records:

```json
{
  "context": "The context passage text...",
  "qas": [
    {
      "question": "What is the capital of France?",
      "detected_answers": [
        {"text": "Paris", "char_spans": [[42, 47]]}
      ]
    }
  ]
}
```

The generator function `mrqa_generator(file_path)` (`src/dataset.py`) parses each file:
1. Opens `.jsonl.gz` with UTF-8 text decompression.
2. Skips lines with `'header'` key or missing `'context'`.
3. Flattens nested `qas` list.
4. Filters out items where `detected_answers` is empty (`if not qa['detected_answers']: continue`).
5. Extracts answer text from the first detected answer (`qa['detected_answers'][0]['text']`).
6. Yields a flattened dict:
   ```python
   {
       'context': context,
       'question': qa['question'],
       'answer': qa['detected_answers'][0]['text']
   }
   ```

### Interleaved Dataset Loading (`load_mrqa_dataset`)

`load_mrqa_dataset(data_dir, seed=42)` (`src/dataset.py`):
1. Verifies existence of all six subset files (fails fast with explicit path if missing).
2. Converts each generator into a HuggingFace `Dataset` via `HFDataset.from_generator`.
3. Blends subsets with equal probability ($p = 1/6$) using `interleave_datasets(..., stopping_strategy="all_exhausted")`. Smaller subsets are oversampled so no training data is truncated.

---

## FineWeb-EDU Pretraining Dataset (`load_fineweb_dataset` & `PretrainDataset`)

### Loading Parquet Shards (`load_fineweb_dataset`)

`load_fineweb_dataset` loads memory-mapped Arrow datasets from `.parquet` files under `data_dir`.

### Filtering and Pretrain Wrapper (`PretrainDataset`)

`PretrainDataset` (`src/dataset.py`):
- Wraps FineWeb Arrow dataset for continuation pretraining.
- Filters out documents shorter than `min_doc_tokens` (default 320).
- Performs single columnar read over `token_count` (or text length proxy `len(text) // 4`) for fast filtering without per-row Arrow overhead.
- Returns raw document text in `__getitem__`: `{"text": text}`. The random prefix $X$ / suffix $Y$ split is performed during collation in `collate_fn_pretrain`.

---

## `CompressionDataset` and Prompt Construction

`CompressionDataset` (`src/dataset.py`) wraps the raw QA dataset and handles tokenization and context truncation.

### Prompt Ordering (Context-First)

Prompts are formatted using the Qwen chat template (`src/dataset.py`):

```python
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

def _build_prompt_text(self, context: str, question: str) -> str:
    user_content = f"{context}\n{question}" if context else question
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return self.llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
```

> **Note on Prompt Order:** The user message uses **Context-First** ordering (`context\nquestion`). This matches the Qwen3 FullFT teacher's SFT format, ensuring the frozen teacher model operates in-distribution.

### Student Template Modes (`student_template`)

- **`"bare"` (Design A):** Compressor encodes the full Qwen chat template string (`self._build_prompt_text`).
- **`"wrapped"` (Design B):** Compressor encodes document content only (`f"{question}\n{context}"` for B-qc, or `context` for B-ctx). Chat template markers are added as real token embeddings in the LLM forward pass.

When `student_template == "wrapped"` and `compress_question == False` (B-ctx mode), `__getitem__` also returns:
- `question_ids`: Un-padded token IDs for the literal question under the LLM tokenizer.
- `question_comp_len`: Length of `question + "\n"` under the compressor tokenizer (used to compute context start offsets).

---

## Per-Tokenizer Context Truncation

Because Mamba and Qwen use different tokenizers and vocabularies, context truncation is computed **independently** for each tokenizer (`src/dataset.py`).

```python
def _truncate_context_for(self, tokenizer, context: str, question: str, role: str = "compressor", build_text=None) -> str:
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
```

### Safety Features

1. **`_TRUNCATION_MARGIN = 4`:** Safety buffer absorbing any character-to-token re-encoding mismatch.
2. **End-Trimming:** Trims tokens from the **END** of the context passage so that the question and system instructions are never dropped.
3. **Independent Evaluation:** Called separately for `self.llm_tokenizer` and `self.compressor_tokenizer` in `__getitem__`.
