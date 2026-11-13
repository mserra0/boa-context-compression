# KD Pretraining (teacher-forced continuation KD)

## Motivation

The main KD objective (`docs/training/knowledge-distillation.md`) distills a QA teacher into the
compressor **per finetuning dataset**: every run pays the cost of teacher generation. This
experiment moves that expensive alignment **once, into pretraining**.

The approach: pretrain the compressor on a large corpus (FineWeb-EDU) with a KD continuation
objective, then finetune cheaply with plain LM-CE-SFT on any downstream dataset, warm-started
from the KD-pretrained checkpoint. The soft-label alignment is paid once and reused.

KD pretraining is a drop-in **alternative continuation objective** to LM-CE pretraining
(`docs/training/loop.md` on `main` / the `LMPretrainer` path). It shares the model, the data, the
dataloading, and the student forward — only the *target* changes from the hard corpus token to
the teacher's soft distribution.

---

## Objective

Given a document, sample a random character boundary and split it into a prefix **X** and a
continuation **Y**.

- **Student** sees only the *compressed* X (Mamba soft prompts / KV) and must predict Y.
- **Teacher** is the SAME frozen decoder, but it sees the *plaintext* X (real Qwen tokens) and
  teacher-forces Y in one forward pass.

The student is trained to match the teacher's distribution over Y (KL, temperature `T`). The
plaintext advantage the teacher has over the student is exactly the information the compressed
representation must recover.

Both sides score the **same right-padded Y positions**, so a KD-pretrain run is directly
comparable to an LM-CE-pretrain run: same X, same corpus Y, soft vs. hard labels.

```
document text  ──► [ prefix X | continuation Y ]
                        │            │
        ┌───────────────┘            └───────────────┐
        ▼                                             ▼
  compress X  ──► soft prompts ─┐              plaintext X (Qwen toks)
                                │                     │
                                ▼                     ▼
                   STUDENT: frozen decoder    TEACHER: frozen decoder
                   over [soft | Y]            over [X | Y]  (teacher-forced)
                                │                     │
                                ▼                     ▼
                    student_logits [B,G,V]   teacher_logits [B,G,V]
                                └────────► kd_lm_loss (KL·T²) ◄────────┘
```

---

## Teacher pass geometry (`KDPretrainer._teacher_logits`)

There is **no autoregressive generation** — the teacher runs a single teacher-forced forward.
Alignment is purely index arithmetic:

- X is **left-padded** `[B, Lx]`, so every row's real prefix ends exactly at column `Lx-1`.
- Y is **right-padded** `[B, G]`, concatenated directly after X → `[B, Lx+G]`.
- The logit at column `Lx-1` (last real X token) predicts `Y_0`; column `Lx-1+i` predicts `Y_i`.
- So the slice `[Lx-1 : Lx-1+G]` lands on exactly the Y positions the student is scored on.

`position_ids` come from a cumsum of the padding mask (pad positions parked at 1), so left-padded
X and right-padded Y both get correct positions. The teacher pass is **compression-agnostic** — it
never touches the compressor — so it works unchanged for `embed`/`kv` prompt modes and for
mamba1/mamba2.

---

## Student pass (shared seam)

`LMPretrainer._student_logits` is the **objective-agnostic seam**: it returns student logits
`[B,G,V]` aligned to the Y targets, dispatching on `prompt_mode` (`embed` → prepended soft-prompt
embeds; `kv` → per-layer K/V cache + learned trigger token). Both `LMPretrainer` (LM-CE) and
`KDPretrainer` (KD) consume the *same* seam, which is what keeps the two objectives numerically
comparable. `KDPretrainer` inherits it untouched — only the loss target differs.

---

## Loss (`kd_lm_loss`, `src/losses.py`)

```python
kd_lm_loss(student_logits, teacher_logits, target_mask, temperature)
```

Token-uniform KL over the Y positions, scaled by `T²` (Hinton et al. 2015), pad-aware via
`target_mask` and NaN-safe on a fully-padded row.

Difference from the QA `kl_loss`: it is **pad-aware, not eos-aware**. The continuation Y is raw
document text with no answer/eos semantics, so *every* real Y token is supervised and only the
right-padding tail is dropped (no `valid_positions_mask` / first-eos logic).

---

## Evaluation stays comparable

The KD *training* loss is a KL and is not comparable across runs (or against an LM-CE run). So
validation always reports the **student cross-entropy on Y** via `LMPretrainer.ce_readout`,
regardless of objective. An LM-CE run and a KD run therefore log the same `val/ce` and `val/ppl`
metric and can be compared directly.

---

## Data path (`src/utils/dataloading.py`)

`_pretrain_xy_pairs(..., teacher=True)` additionally emits the LLM-tokenized prefix X (same
character boundary, same last-`max_ctx`-token tail cap) so the teacher sees the *plaintext* X the
student had to compress. The KD collates wrap it into `teacher_inputs` (left-padded `[B, Lx]` via
`_build_llm_batch`):

| compressor | LM-CE collate | KD collate |
|---|---|---|
| mamba2 (packed `[1,T]`) | `collate_fn_pretrain` | `collate_fn_pretrain_kd` |
| mamba1 (left-pad `[B,I]`) | `collate_fn_pretrain_mamba1` | `collate_fn_pretrain_kd_mamba1` |

The compressor/target contract is identical to the LM-CE collates; KD only *adds* the
`teacher_inputs` key. In the training loop (`src/pretrain.py`), `batch.get("teacher_inputs")`
selects the 3-arg forward; the LM-CE path has no such key and calls the 2-arg forward.

---

## Configuration

Selected by a single flag in `main.py`:

```yaml
model:
  pretrain_loss: "kd"     # "ce" (LMPretrainer, hard token) | "kd" (KDPretrainer, soft teacher)
  temperature: 2.0        # KD temperature (only used when pretrain_loss: kd)
  prompt_mode: "embed"    # "embed" | "kv" — student injection, unchanged by KD
```

`pretrain_loss: kd` selects `KDPretrainer` + the `_kd` collate + temperature; everything else
(model, data, packing, sampler, scheduler) is shared with the LM-CE path.

---

## Files

| File | Role |
|---|---|
| `src/losses.py` | `kd_lm_loss` — pad-aware token-uniform KL·T² |
| `src/pretrain_model.py` | `KDPretrainer(LMPretrainer)` + `_teacher_logits`; extracted `_student_logits`/`ce_readout` seam |
| `src/utils/dataloading.py` | `collate_fn_pretrain_kd(_mamba1)` + `_pretrain_xy_pairs(teacher=True)` |
| `src/pretrain.py` | loop passes `teacher_inputs` when present; `evaluate_lm` uses `ce_readout` |
| `main.py` | `pretrain_loss: ce\|kd` selects trainer + collate + temperature |

---

## Distillation regimes: why teacher-forcing here, generation elsewhere

This experiment deliberately uses a **teacher-forced** KD objective, while the main-branch QA
finetuning path (`docs/training/knowledge-distillation.md`) uses a **generation-based** one. That
is not an inconsistency — they sit at different points on a well-studied axis. There are three
regimes, distinguished by *who produces the target sequence Y the student is scored on*:

| Regime | Who produces Y | Teacher pass | Where we use it |
|---|---|---|---|
| **1. Teacher-forced on fixed data** (word/token KD) | the corpus (Y is given) | one parallel forward over `[X\|Y]` | **this experiment (KD pretrain)** |
| **2. Teacher-generated sequences** (sequence-level KD) | the teacher greedy/beam-decodes Y | autoregressive `llm.generate` | **main branch QA finetuning** (`src/model.py:generate_teacher`) |
| **3. On-policy / student-generated** (GKD, MiniLLM) | the **student** decodes Y, teacher scores it | student rollout + teacher forward | not implemented |

**Why this experiment can use regime 1.** In continuation pretraining, Y is literally the next
chunk of the document — it is *given*, free, and arbitrarily long. Because we already hold Y, we
never generate: concatenate the known `[X | Y]` and run a single teacher-forced forward, reading
the soft targets off the `[Lx-1 : Lx-1+G]` slice (see *Teacher pass geometry* above). One parallel
pass yields all G target distributions — no autoregressive loop, no eos masking.

**Why the main branch uses regime 2.** In QA the "continuation" is the answer. A ground-truth span
exists, so regime 1 is *possible*, but the main branch generates instead because (a) MRQA answers
are short spans → teacher-forcing them yields only a handful of KD targets, whereas greedy
generation yields up to `max_new_tokens` positions of dark knowledge; and (b) the student is fed
the teacher's *own* tokens, so it imitates the teacher's actual trajectory. This is classic
**sequence-level KD** (Kim & Rush, 2016), and it is why that path needs `valid_positions_mask`/eos
logic — generated length is variable.

**The known trade-off (exposure bias).** Regime 1's weakness is *distribution mismatch*: the
student trains on fixed prefixes it never produces at inference, so errors can compound at
generation time. This is the exposure-bias problem that motivates regimes 2 and 3. **GKD** (Agarwal
et al., 2023) shows that on-policy KD — training the student on *its own* self-generated sequences
scored by the teacher — closes this gap and outperforms fixed-target KD on free-generation tasks.

**Design rationale.** Generation-based KD is expensive: every finetuning run pays for G serial
teacher forwards, whereas teacher-forcing is one parallel forward. This objective uses teacher-forcing
where it is cheap and long — corpus continuation at pretraining scale — to pay the soft-label
alignment **once**, then finetunes with plain LM-CE-SFT (no teacher). A purely teacher-forced pretrain
does not address exposure bias; on-policy (regime 3) or teacher-generated (regime 2) phases are
complementary to word-level KD (Kim & Rush), not a replacement.

**References.**
- Kim & Rush, *Sequence-Level Knowledge Distillation*, EMNLP 2016 — [aclanthology D16-1139](https://aclanthology.org/D16-1139.pdf). Word-level vs sequence-level KD; the two are synergistic.
- Agarwal et al., *On-Policy Distillation of Language Models (GKD)*, 2023 — [arXiv 2306.13649](https://arxiv.org/abs/2306.13649). Supervised (fixed-target) vs on-policy KD; generalized divergences.
- Gu et al., *MiniLLM: Knowledge Distillation of LLMs*, 2023 — [arXiv 2306.08543](https://arxiv.org/abs/2306.08543). Reverse-KL, student-generated (on-policy) KD.
- Hinton et al., *Distilling the Knowledge in a Neural Network*, 2015 — the `T²`-scaled soft-target recipe used by `kd_lm_loss`.
- Background on exposure bias: Bengio et al., *Scheduled Sampling*, NeurIPS 2015; Ranzato et al., *MIXER*, ICLR 2016.

---

## Running the overfit smoke on a multi-GPU node

Two overfit configs mirror the real KD run (real Qwen3-1.7B-FullFT teacher, mamba snapshots,
FineWeb-EDU 100BT) but shrink to a 200-step curve over a fixed 512-doc subset (`overfit: true` →
the loader is cycled). They check that the KD pipeline runs end-to-end.

```bash
# From the repository root on a 4-GPU node:
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_training.sh configs/pretrain/overfit/config_pretrain_overfit_kd_kv_mamba2_mn5.yaml
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_training.sh configs/pretrain/overfit/config_pretrain_overfit_kd_kv_mamba1_mn5.yaml
```

`run_training.sh` launches `torchrun` on the listed GPUs (KD always runs under DDP); on nodes without
internet access also set `HF_HUB_OFFLINE=1` and `WANDB_MODE=offline`. Expected: `val/ce` drops steadily over the 200 steps on the
tiny subset — a memorization curve, not a real generalization signal.

**4-GPU (DDP world_size 4) notes.** The `512`-doc subset is sized so the samplers keep every rank
non-degenerate after `min_doc_tokens` filtering — the token-budget sampler (mamba2) assigns
`total_packs // world_size` packs/rank and would drop a rank to **zero batches (no-op run)** if
total packs < 4. `overfit: true` cycles the loader, so all 4 ranks execute exactly `max_steps`
regardless of shard size → no DDP step desync. Effective optimizer batch =
`batch_size × grad_accum × 4` (mamba1: `2 × 4 × 4 = 32` docs/step; mamba2: token-budget packs ×4).

The full KD run is `configs/pretrain/mn5/pretrain_mamba2_kd.yaml` (10k steps), launched with
`sbatch scripts/utils/mn5_train_mamba2.slurm configs/pretrain/mn5/pretrain_mamba2_kd.yaml`.

