# Datasets: download and layout

## MRQA 2019 (fine-tuning and evaluation)

Boa is fine-tuned on the six in-domain MRQA 2019 training sets and evaluated on their dev sets and on six out-of-domain dev sets.

```bash
bash scripts/download_mrqa.sh data/MRQA          # train/ + dev/        (about 1.5 GB)
bash scripts/download_mrqa.sh data/MRQA --ood    # additionally dev_ood/ (needed for src/evaluation)
```

The script downloads the official release (<https://github.com/mrqa/MRQA-Shared-Task-2019>) and renames two files to the names expected by `src/dataset.py`:

```
data/MRQA/
  train/   SQuAD  NewsQA  TriviaQA  HotpotQA  SearchQA  NaturalQuestions        (*.jsonl.gz)
  dev/     SQuAD  NewsQA  TriviaQA  HotpotQA  SearchQA  NaturalQuestions        (*.jsonl.gz)
  dev_ood/ BioASQ  DROP  DuoRC  RACE  RelationExtraction  TextbookQA           (*.jsonl.gz)
```

| Official file | Local name |
|---|---|
| `TriviaQA-web.jsonl.gz` | `TriviaQA.jsonl.gz` |
| `NaturalQuestionsShort.jsonl.gz` | `NaturalQuestions.jsonl.gz` |
| `DuoRC.ParaphraseRC.jsonl.gz` | `DuoRC.jsonl.gz` |

Point `data.data_dir` in the config (or `datasets.mrqa_data_dir` in the evaluation config) at the MRQA root. The first run converts the files into a Hugging Face Arrow cache, which takes a few minutes. Schema and preprocessing: [dataset.md](dataset.md).

## FineWeb-Edu (LM pretraining)

Pretraining configs read parquet shards from `data.data_dir`. The stage-1 configs (`configs/pretrain/mn5/**` and `configs/ablations/8x/*/pretrain/`) use the **`sample/100BT`** subset of [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), roughly 300 GB of parquet files; the small configs (`configs/pretrain/config_pretrain.yaml` and `configs/pretrain/overfit/`) use `sample/10BT`, roughly 30 GB. Download with the Hugging Face CLI:

```bash
# stage-1 paper configs
hf download HuggingFaceFW/fineweb-edu --repo-type dataset \
    --include "sample/100BT/*" --local-dir /path/to/data/fineweb-edu-10B

# configs that use sample/10BT
hf download HuggingFaceFW/fineweb-edu --repo-type dataset \
    --include "sample/10BT/*" --local-dir /path/to/data/fineweb-edu-10B
```

This gives `/path/to/data/fineweb-edu-10B/sample/{100BT,10BT}/*.parquet`, which is the `data_dir` the configs expect. The pretraining runs consume only the token budget set by `max_steps`, so a partial download of the subset (fewer shards, with `data_dir` unchanged) works for smoke tests.
