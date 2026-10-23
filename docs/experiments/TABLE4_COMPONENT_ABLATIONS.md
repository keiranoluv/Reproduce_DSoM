# Table 4 — A5-matched NEKD-only and NNKD-only student runs

Status: preparation and CPU checks only. The user reviews, commits, and runs
GPU smoke/full training. No GPU job is launched during this investigation.

## Scope and protocol

Two missing student rows from [DSoM v1, Table 4](https://arxiv.org/html/2507.20738v1):

| Component | CLI coefficient | Paper MRR | H@1 | H@3 | H@10 |
|---|---|---:|---:|---:|---:|
| RC + NEKD | alpha=1, beta=0 | 0.4100 | 0.3307 | 0.4493 | 0.5600 |
| RC + NNKD | alpha=0, beta=1 | 0.4233 | 0.3393 | 0.4639 | 0.5808 |

The launcher copies the full configuration from the completed A5 result:
`runs/student/MMKB-DB15K/learned_rc_adam_full/20261021_173746_2988787/checkpoints/0/result.json`.
It validates learned RC with separate policy Adam LR 1e-5, binary +1/-10
rewards, and the full original MMKB-DB15K benchmark split. Student remains
Adagrad LR 0.1, rank 2000, batch 1000, N3 0.01, KD temperature 4, CE/KD weights
1/2, max 200 epochs, validation every 3 epochs, and seed 2022.

The existing `min` scheduler is retained to match A5, including its known
direction mismatch with validation Hits@10. All settings are copied, including
input normalization, optimizer betas, feature paths, and teacher checkpoint.
Only alpha/beta, run identity, and output directory change for full training.

The student and policy start fresh; the teacher is reused from Exp01. This
does not resume A5 student weights. Teacher parameters stay frozen. RC remains
learned; neither component run uses `source_frozen`. Full NDKD still routes
through `--kd_loss_type kdrl --rc_distill_loss ndkd --teacher_strategy rc`.
The two coefficients turn off one distillation term, not supervised CE/N3 or RC.

## Mathematical audit: released code versus paper

Current `multi_dkd_loss` in `src/kd_optimizers.py` computes:

1. NEKD: KL between two-bin distributions consisting of total neighbor
   probability mass and total non-neighbor mass.
2. NNKD: KL between distributions renormalized over non-neighbor entities,
   excluding all training neighbors, including the current target.
3. The weighted sum `alpha * NEKD + beta * NNKD`, with temperature-squared
   scaling and batch averaging for both terms.

NEKD is **not** KL between individual neighbor probabilities. Training neighbors
come only from training triples and their reciprocals, not validation/test filters.

**Discrepancy:** Section 4.5 describes the neighbor bin as the **average** of
neighbor probabilities, divided by neighbor count. The released implementation
uses their **sum**, via `cat_mask`. The two definitions coincide for a single
neighbor and generally differ for multiple neighbors.

These runs deliberately preserve the released SUM definition used in A5,
A9/A10, and the existing reproduction. There is no source-loss modification.
They are source-faithful component ablations, not verification of the paper's
MEAN formula. Changing to MEAN would require a new matched full-NDKD baseline.

CPU tests independently compute grouped and conditional KL values and gradients,
verify the two components reconstruct default NDKD, check singleton equivalence
with DKD, and explicitly demonstrate the SUM/MEAN discrepancy. Tests also check
the actual dataset loader's training-only reciprocal neighbor sets and all four
generated full/smoke CLI configurations without importing CUDA setup.

## Review and commit

All work here is in three new files; existing `src/` training code is unchanged:

```bash
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python -m pytest -q \
  tests/test_table4_components.py

CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/run_table4_components.py --mode plan

git add scripts/run_table4_components.py tests/test_table4_components.py \
  docs/experiments/TABLE4_COMPONENT_ABLATIONS.md
git diff --cached --check
git diff --cached --stat
git commit -m "exp(db15k): add A5-matched NEKD and NNKD ablation workflow"
```

Default `plan` mode only checks references, hashes inputs, and writes commands.
It does not import Torch/CUDA or start training. A full run requires a clean
tracked working tree and verifies the relevant files have been committed.
The shared metadata helpers in `scripts/run_table2.py` are already committed.

## GPU commands for the user

First smoke-test both jobs (1 epoch, 3000 training examples, 64 queries per
evaluation split). The full training-only neighbor map is retained:

```bash
./dsom/bin/python scripts/run_table4_components.py --mode smoke --workers 1
```

After both jobs pass, run the two full student jobs sequentially with W&B:

```bash
WANDB_PROJECT=DSoM-Reproduction \
./dsom/bin/python scripts/run_table4_components.py \
  --mode full --workers 1 --wandb
```

Use `--components nekd` or `--components nnkd` to run just one component.
`--workers 2` starts both training jobs concurrently; default is 1. W&B is
opt-in even if `DSOM_WANDB` was exported in the shell. Each component gets its
own W&B run and checkpoint directory in a common suite group.

Outputs use NEW timestamped paths under `runs/table4_components/{plan,smoke,full}/`.
`--output runs/table4_components/full/CUSTOM_NAME` selects a new directory;
existing output directories are never overwritten. Original result JSONs,
teacher/features, and datasets are hashed before training and checked again
before summarization. Failures stop queued jobs; no partial optimizer resume
is implemented. Rerun a failed component in a new directory.

## Results and limits

The launcher checks all saved arguments, the selected-checkpoint test metrics,
student checkpoint existence, and at least one nonzero RC policy-weight update.
It writes `summary.json` and `table4_student_results.md` using `final_result`,
not the last W&B test summary. Existing A4/A5/A9/A10 rows are included as
references. Smoke scores have no paper-MRR delta and are not benchmark scores.

To regenerate a summary using CPU/file checks only:

```bash
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/run_table4_components.py \
  --mode summarize --output runs/table4_components/full/YOUR_RUN_DIRECTORY
```

With both full runs present, `component_pair_completed=true` completes the six
student rows together with A4/A5/A9/A10. **Whole Table 4 remains incomplete:**
direct Teacher Avg., ML-on-validation, and RC-on-validation require distinct
teacher-ensemble evaluations. A4 is the average-teacher-distilled student, not
direct Teacher Avg. inference. Single seed, reconstructed features, and the
original DB15K validation contamination remain interpretation limits.
