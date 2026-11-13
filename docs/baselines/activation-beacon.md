# Baseline: Activation Beacon

This branch contains the code to reproduce the Activation Beacon baseline on the MRQA setup used in the Boa paper. It is adapted from the official implementation (see [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)) and is independent of the Boa compressor code on `main`. Paths such as `/path/to/...` and `<your_account>` in configs and scripts are placeholders; replace them with your own (the full list is in `docs/setup.md` on `main`).

Run commands from the repository root. Install with `uv sync` (prerequisites: `docs/setup.md` on `main`).

## Layout

- `src/baselines/activation_beacon/`: Activation Beacon backends and runner. `qwen3_official.py` is the trainable Qwen3 backend that ports the official recurrent beacon mechanism; `qwen3_simple.py` is a simplified Qwen3 backend (token insertion) used for scalable pretraining; `qwen2_official.py` is the Qwen2/Qwen2.5 backend that loads the authors' published remote code.
- `src/pretrain_activation_beacon.py`: FineWeb-Edu pretraining entrypoint.
- `src/train_activation_beacon.py`: MRQA fine-tuning entrypoint.
- `src/evaluate_activation_beacon.py`: MRQA in-domain / out-of-domain evaluation entrypoint.
- `src/dataset.py`, `src/utils/evaluate_utils.py`: MRQA loading, prompt handling, QA-F1/EM metrics, result writing.
- `configs/pretrain/mn5/`, `configs/train/`, `configs/eval/`: pretraining, fine-tuning and evaluation configs.
- `scripts/`: local launchers; `scripts/utils/*.slurm`: example Slurm scripts (see `docs/cluster.md` on `main`).

Checkpoints and datasets are not included; config files contain placeholder locations.

## Method summary

Activation Beacon inserts learned beacon activations into the causal sequence so the model can compress the context while preserving answer-relevant information. In the MRQA setup the question is placed before the context, so compression is question-conditioned. The experiments use Qwen3 with these settings:

```text
pretraining (qwen3_simple, configs/pretrain/mn5/):  beacon_window = beacon_stride = 512, beacon_ratio = [2, 4, 8], step-random mixing
fine-tuning (qwen3_official, configs/train/):        beacon_window = beacon_stride = 128, beacon_ratio = [2, 4, 8, 16], step-random mixing
evaluation:                                          fixed beacon_ratio; window/stride taken from the checkpoint
```

## Configuration

Relevant config fields:

- `model.base_model_name_or_path`: base model for pretraining, or the pretrained beacon checkpoint for fine-tuning.
- `model.beacon_code_model`: used by the `qwen2_official` backend as the Hub id whose remote code it loads (`namespace-Pt/beacon-qwen-2-7b-instruct` in the pretraining/official configs; the `qwen3_simple` backend ignores it during pretraining). The Qwen3 backends read it only when `base_model_name_or_path` is an adapter-only checkpoint produced by pretraining; it then names the plain Qwen3 model the architecture is built from (as in the MRQA fine-tuning config).
- `data.data_dir`: FineWeb-Edu (pretraining) or MRQA (fine-tuning) root; `data.encoded_cache_dir`: tokenized MRQA cache (change it when the model, window or dataset settings change).
- `paths.output_dir`: output checkpoint directory.
- `activation_beacon.model_name_or_path`: fine-tuned checkpoint used for evaluation; `activation_beacon.teacher_model_name_or_path`: uncompressed teacher for the optional baseline conditions; `activation_beacon.run_name` / `beacon_ratio` / `beacon_ratio_mix`: evaluation run name and fixed ratio.
- `datasets.mrqa_data_dir`: MRQA evaluation root; `datasets.in_domain` / `datasets.out_of_domain`: datasets to evaluate.
- `evaluation.metric: qa_f1` (exact QA-F1/EM, no LLM judge) and `evaluation.conditions: ["student"]`.
- `wandb.enabled`: whether training initialises Weights & Biases (`WANDB_MODE=disabled` turns it off).

Configs: `configs/pretrain/mn5/pretrain_activation_beacon_qwen3_17b_fullft_fineweb_edu_10b.yaml` (single node), `..._10b_32n.yaml` (32 nodes x 4 GPUs); `configs/train/config_activation_beacon_qwen3_17b_fullft_fineweb_edu_10b_official.yaml` (FineWeb-Edu pretraining with the `qwen3_official` backend, window 128); `configs/train/config_activation_beacon_qwen3_17b_fullft_after_fineweb_edu_10b_mrqa_official.yaml` (MRQA fine-tuning); `configs/eval/config_activation_beacon.yaml`, `configs/eval/config_activation_beacon_qwen3_official.yaml` (evaluation). The Qwen3 base model is the fine-tuned teacher described in `docs/training/teacher.md` on `main`.

## Pretraining

```bash
bash scripts/pretrain_activation_beacon.sh \
  configs/pretrain/mn5/pretrain_activation_beacon_qwen3_17b_fullft_fineweb_edu_10b.yaml

# 32 nodes x 4 GPUs = 128 ranks (batch_size 1 per device, gradient_accumulation_steps 2, beacon_ratio [2, 4, 8], 10B-token budget)
sbatch scripts/utils/mn5_pretrain_activation_beacon_32n.slurm \
  configs/pretrain/mn5/pretrain_activation_beacon_qwen3_17b_fullft_fineweb_edu_10b_32n.yaml
```

Gradient checkpointing (`model.gradient_checkpointing: true`) is implemented by the `qwen3_official` backend only, i.e. the configs under `configs/train/`; the `qwen3_simple` pretraining configs above do not use it. Exact restart is opt-in (`data.num_workers: 0`, `training.save_restart_state: true`) and requires the same world size as the run that saved the checkpoint; resume with `--resume-from-checkpoint <checkpoint-dir>` (also accepted by the Slurm script).

## MRQA fine-tuning

`scripts/train_activation_beacon.sh` and `scripts/finetune_qwen3_after_pretrain_activation_beacon.sh` (the latter defaults to the Qwen3 MRQA config) wrap `uv run --with transformers==<version> --with deepspeed==0.19.2 python src/train_activation_beacon.py`. The Transformers version comes from `ACTIVATION_BEACON_TRANSFORMERS_VERSION` (default `4.40.2` for the Qwen2 remote-code backend, `4.57.1` when the config selects a `qwen3_*` backend); `scripts/pretrain_activation_beacon.sh` does the same for pretraining. Pins are applied per run through `uv run --with`, so the project environment itself is unchanged.

```bash
bash scripts/train_activation_beacon.sh \
  configs/train/config_activation_beacon_qwen3_17b_fullft_after_fineweb_edu_10b_mrqa_official.yaml

# multi-node: scripts/utils/mn5_train_activation_beacon_after_pretrain.slurm <config>
```

Validation runs after each epoch; checkpoints are written to `<output_dir>/checkpoint-epoch-<N>/`.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_activation_beacon.sh configs/eval/config_activation_beacon.yaml cuda
```

The default config evaluates SQuAD, NewsQA, TriviaQA, HotpotQA, SearchQA and NaturalQuestions (in-domain) and BioASQ, DROP, DuoRC, RACE, RelationExtraction and TextbookQA (out-of-domain). To evaluate a fixed ratio without overwriting another run, change `activation_beacon.run_name`, `beacon_ratio` and set `beacon_ratio_mix: "fixed"`. Results are written to `evals/experiments/<run_name>/` (summary: `summary_qa_f1.json`).

Configs can be validated without loading weights: `python src/pretrain_activation_beacon.py --config <config> --validate-only` (same flag for `src/train_activation_beacon.py`).

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

Activation Beacon (upstream method):

```bibtex
@misc{zhang2024longcontextcompressionactivation,
  title={Long Context Compression with Activation Beacon},
  author={Peitian Zhang and Zheng Liu and Shitao Xiao and Ninglu Shao and Qiwei Ye and Zhicheng Dou},
  year={2024},
  eprint={2401.03462},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2401.03462}
}
```
