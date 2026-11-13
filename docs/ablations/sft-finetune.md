# Ablation: supervised fine-tuning (SFT)

This branch adds the supervised fine-tuning objective used by the 8x ablation arms: the compressor is trained on the ground-truth MRQA answers with cross-entropy instead of distilling a teacher (`objective: sft`, handled by `MRQASupervisedTrainer` in `src/model.py` and dispatched from `main.py`). It supports `compressor_type` `mamba` and `mamba2` with `prompt_mode: kv`. Paths such as `/path/to/...` and `<your_account>` in configs and scripts are placeholders; replace them with your own (the full list is in `docs/setup.md` on `main`).

## Arms

| Config | Arm |
|---|---|
| `configs/ablations/8x/qwen3_1-7b/finetune/m2_kv_lm.yaml` | LM pretrain -> SFT finetune (Mamba2, KV) |
| `configs/ablations/8x/qwen3_1-7b/finetune/m2_kv_offkd_lm.yaml` | off-policy KD pretrain -> SFT finetune; the pretraining step runs on `ablation/kd-pretrain` |
| `configs/train/mn5/mambaL2_sft_01comp_from_pretrain.yaml`, `mambaL_sft_01comp_from_pretrain.yaml`, `mambaL_sft_01comp_from_kd_pretrain.yaml` | full-scale SFT counterparts of the KD fine-tuning runs |
| `configs/train/overfit_sft_mamba{1,2}_bctx.yaml` | overfitting smoke tests |

The overall ablation matrix and the branch each arm runs on are described in `configs/ablations/README.md` and `docs/benchmarks/ablations.md` on `main`.

## Running

Set `start_from` to the pretraining run (`/path/to/checkpoints/<pretrain_run_dir>/best_model.pt`, the timestamped directory written by the pretraining config), then:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_training.sh configs/ablations/8x/qwen3_1-7b/finetune/m2_kv_lm.yaml
```

Cluster example: `scripts/utils/mn5_train_mamba2.slurm <config>` (see `docs/cluster.md` on `main`).

## Evaluation

The MRQA evaluation suite (`src/evaluation`) is on `main`. This branch additionally contains:

| File | Purpose |
|---|---|
| `src/evaluate.py`, `scripts/eval/run_eval.sh` | LLM-as-judge MRQA evaluation of a checkpoint (`bash scripts/eval/run_eval.sh <experiment-dir>`); needs a judge server. |
| `scripts/eval/vllm_judge.sh` | starts the vLLM judge server (`Qwen/Qwen3-32B`, 2 GPUs: `JUDGE_GPUS`, default `0,1`). |
| `scripts/eval/run_ruler.sh` | RULER synthetic long-context evaluation: generates data with RULER's `prepare.py`, runs `src/eval/ruler_predict.py`, scores with RULER's `evaluate.py`. Requires a [RULER](https://github.com/NVIDIA/RULER) checkout in `../RULER` (the script stops with an error otherwise); RULER's data-generation packages are not project dependencies, install them once with `uv pip install --python .venv/bin/python nltk wonderwords html2text` (the script runs `uv run --no-sync`). |
| `src/eval/ruler_predict.py` | RULER predictor: splits each RULER input into context (compressed) and trailing question (kept verbatim) and writes RULER-format predictions. |
| `src/eval/mini_niah.py` | graded needle-in-a-haystack in RULER format (needle type x context length) for a quick retrieval check. |
| `src/eval/sanity_qa.py` | quick check on MRQA dev: student vs teacher with and without context. |

## Citation

If you use this code, please cite the Boa paper ([arXiv:2610.33006](https://arxiv.org/abs/2610.33006)):

```bibtex
@misc{serraortega2026lineartime,
  title={Linear-Time Context Compression with State-Space Encoding},
  author={Marc Serra Ortega and Andrey Barsky and Carlos Boned Riera and Tom{\'a}s Ockier and Dimosthenis Karatzas},
  year={2026},
  eprint={2610.33006},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2610.33006}
}
```
