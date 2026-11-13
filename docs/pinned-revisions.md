# Pinned model revisions

Configs refer to Hugging Face models by Hub id (for example `state-spaces/mamba2-1.3b`), which resolves to the
latest revision of that repo and can drift over time. The experiments were run with these exact revisions:

| Hub id | Revision (commit hash) | Used by |
|---|---|---|
| `EleutherAI/gpt-neox-20b` | `c292233c833e336628618a88a648727eb3dff0a7` | `configs/ablations`, `configs/pretrain` |
| `Qwen/Qwen2.5-0.5B-Instruct` | `7ae557604adf67be50417f59c2c2f167def9a775` | `configs/pretrain` |
| `state-spaces/mamba-1.4b-hf` | `6e46eae61c27280517feef46f536d16b91076f08` | `configs/pretrain` |
| `state-spaces/mamba2-1.3b` | `c5b59d00ec85d313adea86a08cad2a43c962dd3b` | `configs/ablations`, `configs/pretrain` |

To pin a model, download that revision and point the config's model path at the downloaded snapshot directory:

```bash
huggingface-cli download <hub_id> --revision <commit_hash>   # or: hf download <hub_id> --revision <commit_hash>
```

### Frozen LLM / teacher checkpoints

The configs use locally fine-tuned Qwen3 checkpoints (`/path/to/checkpoints/qwen3_1_7b_fullft/final`, and `qwen3_4b_fullft/final` for the 4B runs) as the frozen LLM / teacher. They are produced on the `baseline/qwen3-fullft` branch; see `docs/baselines/qwen3-fullft.md` there and `docs/training/teacher.md` on `main`. The 1.7B run started from `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`. The Qwen3-4B base revision was not recorded; use the Hub revision you intend to reproduce against.
