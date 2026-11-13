# Baseline: SAC

This branch contains the code to reproduce the SAC baseline (compression into a KV-style memory with contextual semantic anchors) on the MRQA setup used in the Boa paper. The upstream SAC code is kept isolated under `SAC/` (modified, see [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md)); repository-level wrappers and configs expose it through the same evaluation protocol as the other baselines. Paths such as `/path/to/...` and `<your_account>` in configs and scripts are placeholders; replace them with your own (the full list is in `docs/setup.md` on `main`).

Run commands from the repository root. Install with `uv sync` (prerequisites: `docs/setup.md` on `main`).

## Layout

- `SAC/`: SAC model (`SAC/model/modeling.py`), pretraining (`SAC/pretrain/pre_trainer.py`), MRQA instruction fine-tuning (`SAC/sft/instruction_trainer.py`) and the original evaluation utilities.
- `src/baselines/sac_runner.py`: adapter that exposes SAC through the evaluation protocol shared with Activation Beacon.
- `src/evaluate_sac.py`: MRQA in-domain / out-of-domain evaluation entrypoint.
- `src/dataset.py`, `src/utils/evaluate_utils.py`: MRQA loading, prompt handling, QA-F1/EM metrics, result writing.
- `configs/pretrain/`, `configs/train/sac/`, `configs/eval/`: pretraining configs, MRQA fine-tuning configs for compression ratios 2, 4, 8 and 16, and evaluation configs.
- `scripts/`: local launchers; `scripts/utils/*.slurm`: example Slurm scripts (see `docs/cluster.md` on `main`).

## Method summary

SAC compresses the context into a smaller KV-style memory and decodes the answer conditioned on that memory plus the raw question prompt:

```text
compressor input:  ### Context:\n<context>
decoder input:     ### Question:\n<question>\n### Answer:\n
```

The compressor is bidirectional over context chunks; the decoder sees the uncompressed question after compression, so compression is online with respect to the question.

## Configuration

- Pretraining: `configs/pretrain/mn5/pretrain_sac_qwen3_17b_fullft_fineweb_edu_10b_r8_lm_32n.json`. `pretrain_training_config` holds the base model, batch sizes, learning rate, token budget and world size; `pretrain_task_config` the compression geometry (`chunk_size`, `mem_size`, `compress_ratio`); `data_config` the FineWeb source (`dataset_repo`, a local parquet directory or glob), `model_id` and the MRQA root used for fine-tuning (`instruction_dataset_repo`: `train/` for SFT, `dev/` + `dev_ood/` for evaluation). The 32-node setting is 32 x 4 GPUs = 128 ranks, `compress_ratio` 8, `chunk_size` 512, `mem_size` 64, 10B tokens.
- Fine-tuning: `configs/train/sac/config_sac_qwen3_17b_fullft_fineweb_edu_10b_mrqa_r{2,4,8,16}.json`. `configs/pretrain/sac/config_sac_qwen3_17b_fullft_fineweb_mrqa_r8.json` is the single-node pretraining config used by the local launchers (`scripts/run_sac_pretrain.sh`). Fine-tuning reads the pretrained adapter from `<work_dir>/output/adapter.pt` and writes `<work_dir>/output/instruction_adapter.pt`.
- Evaluation: `configs/eval/config_sac_r{2,4,8,16}_mrqa.yaml` (`sac.work_dir`: checkpoint directory; `evaluation.metric: qa_f1`, exact F1/EM without an LLM judge; `evaluation.clean_predictions: true` removes Qwen thinking tags and scores the first non-empty generated line).

## Pretraining

```bash
bash scripts/run_sac_pretrain.sh \
  configs/pretrain/mn5/pretrain_sac_qwen3_17b_fullft_fineweb_edu_10b_r8_lm_32n.json /path/to/checkpoints/sac/r8

sbatch scripts/utils/mn5_sac_pretrain_32n.slurm \
  configs/pretrain/mn5/pretrain_sac_qwen3_17b_fullft_fineweb_edu_10b_r8_lm_32n.json /path/to/checkpoints/sac/r8
```

The upstream trainer in `SAC/` spawns one process per visible GPU of a single node (`torch.multiprocessing.spawn`); the 32-node script launches `torchrun` directly, which the pretraining trainer supports through global ranks. The other `scripts/utils/mn5_sac_*.slurm` scripts request a single node.

## MRQA fine-tuning

```bash
CUDA_VISIBLE_DEVICES=0,1,2 bash scripts/run_sac_finetune.sh \
  configs/train/sac/config_sac_qwen3_17b_fullft_fineweb_edu_10b_mrqa_r8.json /path/to/checkpoints/sac/r8 14528

sbatch scripts/utils/mn5_sac_finetune_r8.slurm        # ratio-specific variants: _r2, _r4, _r16
```

The second argument is the work directory of the pretrained checkpoint to fine-tune; the third is the rendezvous port.

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_sac_eval_protocol.sh configs/eval/config_sac_r8_mrqa.yaml cuda
```

This evaluates SQuAD, NewsQA, TriviaQA, HotpotQA, SearchQA, NaturalQuestions (in-domain) and BioASQ, DROP, DuoRC, RACE, RelationExtraction, TextbookQA (out-of-domain); results go to `evals/experiments/<run_name>/`. Use it for comparisons with the other methods. The original SAC evaluator is available as `bash scripts/run_sac_eval.sh <work_dir>` (it runs through `uv run --with matplotlib --with rouge --with nltk`, because those packages are not project dependencies). `WANDB_MODE=disabled` turns off Weights & Biases logging. The Slurm scripts activate the environment given by `SAC_ENV` (default `/path/to/venv`) and log to the W&B project `boa` (`WANDB_PROJECT`).

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

SAC (upstream method):

```bibtex
@misc{liu2025autoencodingfreecontextcompressionllms,
      title={Autoencoding-Free Context Compression for LLMs via Contextual Semantic Anchors},
      author={Xin Liu and Runsong Zhao and Pengcheng Huang and Xinyu Liu and Junyi Xiao and Chunyang Xiao and Tong Xiao and Shengxiang Gao and Zhengtao Yu and Jingbo Zhu},
      year={2025},
      eprint={2510.08907},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.08907}
}
```
