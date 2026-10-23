# Table 7 — WN9 teacher/student workflow

Status: **user completed full WN9 teacher/student training and checkpoint
evaluation; CPU/file verification PASS on 2026-10-24**. The DSoM row is complete
under the documented source-frozen protocol; competitor rows were not trained.
The assistant did not launch GPU jobs. Data preparation is documented in
[WN9_SOURCE_AUDIT.md](WN9_SOURCE_AUDIT.md).

## Recorded full run — 2026-10-24

Suite: `runs/table7/full/20261023_173538_401388/` (folder timestamp is UTC).
Training source: `cc00d7e94c57266a2e5ebfd6df56e483f17ef45a`; seed 2022;
policy optimizer `source_frozen`. Both stages and E2E summary report **PASS**;
`full_table7_dsom_row_completed=true`, `exact_author_protocol_reproduced=false`.
Portable evidence: [table7_results_20261024.json](table7_results_20261024.json).

| Setting | MR | MRR | H@1 | H@3 | H@10 |
|---|---:|---:|---:|---:|---:|
| Paper DSoM | — | 0.921300 | 0.905200 | — | 0.955600 |
| Reproduction, filtered | 6.131160 | 0.919634 | 0.903336 | 0.928355 | 0.949583 |
| Reproduction, raw | 14.506444 | 0.618763 | 0.520470 | 0.669447 | 0.810083 |

Relative to [paper Table 7](https://arxiv.org/html/2507.20738v1#A4), filtered
MRR/H@1/H@10 differ by **-0.167 / -0.186 / -0.602 percentage points**.
This numerical closeness is conditional on filtered evaluation. Raw MRR differs
by -30.254 pp; the evaluation setting has a much larger effect than the filtered
paper gap. Both settings use the same checkpoint selected using filtered valid
MRR; raw did not receive a separate checkpoint search. This does not resolve the
paper's protocol, establish statistical significance, or verify learned RC.

Teacher selected epoch **33**, stopped at epoch 44; student selected epoch **15**,
stopped at epoch 26. Student selected validation MRR is .916170.
The saved independent checkpoint evaluation covers all **1,337 validation and
1,319 test queries per direction** (RHS and reciprocal LHS).
Teacher oracle scores are excluded from the comparison above. Tracking:
[teacher](https://wandb.ai/keiranoluv-hcmus/DSoM-Reproduction/runs/cj92k47u),
[student](https://wandb.ai/keiranoluv-hcmus/DSoM-Reproduction/runs/8k36o3o8).

The CPU review re-ran the summarizer's validation with writes intercepted and
required generated summary text to match the existing JSON/Markdown exactly.
It verified data/feature input hashes, teacher and student checkpoint/result
hashes, teacher-to-student linkage, validation-based epoch selection, evaluation
hash and complete coverage, and saved reloaded-checkpoint metrics against training
metrics (tolerance 2e-6). No Torch import, checkpoint inference or training was
performed during this review; original artifacts were left byte-for-byte intact.
Inverse-fact leakage, reconstructed feature equivalence, SUM NDKD, retained
scheduler direction and single-seed limitations below still apply.

## Independent stages and E2E composition

- `scripts/run_wn9_teacher.py`: fresh multimodal WN9 teacher via existing `learn.py`.
- `scripts/run_wn9_student.py`: fresh structural student via existing `kd_learn.py`,
  given a completed, validated WN9 teacher stage directory.
- `scripts/run_table7.py`: calls the same two stages sequentially, then aggregates
  the separately reloaded student checkpoint's raw/filtered evaluation.
- `scripts/eval_wn9_student.py` / `src/wn9_evaluation.py`: standalone raw/filtered
  scoring of the selected student, never teacher minimum/oracle ranks.
- `scripts/wn9_workflow.py`: shared metadata, configuration, subprocess launching,
  checkpoint/provenance validation, and summary. No Torch import in plans.

No existing training, loss, dataset evaluator, or model implementation is edited.
Teacher/student processes are separate, so GPU memory is released between stages.
The new evaluator runs as another separate process after student training exits.
Teacher failure prevents student launch. Student/evaluation failure prevents a
passing E2E summary; logs and `failure.json` remain for inspection.

Both standalone and E2E entrypoints default to **plan**. Plan checks prepared
WN9 inputs and hashes data without Torch/CUDA or training. E2E plan's student
checkpoint is explicitly pending; actual student commands are generated only
after selecting and validating the new teacher checkpoint. Each executed
stage's `plan.json` and `command.sh` record its actual command.

## Default protocol

Full teacher: WN9 6555 entities / 18 reciprocal relations, rank 2000, Adagrad LR
.1, batch 1000, N3 .01, init .001, modality split S/V/D, alpha 1, scale 20,
max 200 epochs, validate every 3, early stopping 10, seed 2022.
The existing `min` scheduler direction is retained. Teacher selection follows
the released training implementation: **highest filtered validation Hits@10**,
with minimum rank across modalities (oracle diagnostic). Teacher `best_mrr` is
the MRR at that selected epoch, not necessarily the maximum MRR. These teacher
metrics are not the final Table 7 prediction performance.

Student: same structural ComplEx rank/optimizer/regularizer/schedule, CE/KD weights
1/2, temperature 4, NDKD alpha/beta 1/1, seven-action binary +1/-10 RC, hidden
1024, policy temperature 1, no policy input normalization. Teacher is frozen.
Student checkpoint selection is **highest filtered validation MRR**.

Default policy optimizer is **source_frozen**, consistent with the prior main
table baselines. Policy parameters are initialized/sampled but excluded from
optimization. This does not verify learned RL. Optional
`--policy-optimizer Adam --policy-lr 0.00001` runs the separately adapted learned
RC variant; keep it in its own new suite. Full Adam students require recorded
nonzero policy updates. Smoke only checks update instrumentation: zero updates
can be legitimate when a tiny run has no nonzero binary reward advantages.

Smoke retains rank 2000 and all entity/relation mappings/features, but runs each
stage for 1 epoch on 1000 reciprocal-augmented training examples, validates each
epoch, and evaluates 64 queries per direction. Training masks still use the full
training-only neighbor map. Smoke metrics are execution checks, not Table 7 scores.
Full E2E always trains a **new full teacher and new student**, never resumes or
reuses smoke checkpoints.

## Raw and filtered evaluation

The saved student is reloaded, checked as a unimodal `ComplExMDR` with WN9 shape,
and evaluated on valid/test RHS and reciprocal LHS. Both protocols use identical
queries and pessimistic `>=` ties; the target itself is excluded. Raw keeps all
other entities; filtered additionally excludes known answers in the original
split-union filters. NumPy unsigned IDs are explicitly converted to long indices.
Batch size 2000 and `get_queries @ get_rhs` / `score` match the legacy student
ranking path. No oracle modality minimum or target-aware action choice is used.

Full student stage verifies complete valid/test coverage and agreement between
reloaded filtered validation/test MRR and the selected training checkpoint, plus
test Hits@1/3/10 (tolerance 2e-6). Smoke uses a fixed seed 2022 evaluation subset;
it differs from the legacy trainer's random smoke subsets, so equal smoke
metrics are not required.

[DSoM Table 7](https://arxiv.org/html/2507.20738v1#A4) gives MRR .9213, H@1 .9052,
H@10 .9556. The exact evaluation protocol for that table remains unresolved:
[DySarl §4.1.3](https://openreview.net/pdf?id=Gf94NMabeh) reports raw metrics,
whereas the DSoM released dataset evaluator uses filtered ranking. This workflow
reports both settings on the **same filtered-validation-selected checkpoint**.
It does not select a different checkpoint based on raw test performance, or
claim that the raw result uses raw-validation checkpoint selection.

Other retained limits: inverse facts in WN9 splits, reconstructed frozen-encoder
features and eight inherited missing-image vectors, released SUM NDKD versus
paper MEAN, single seed, and the original scheduler direction. No graph-trained
WN18 checkpoint is accepted. The teacher must be a completed workflow stage
with matching WN9 data/feature hashes and a verified checkpoint hash. Full
student rejects smoke teacher; standalone smoke student may use a full teacher.

## CPU review and commit

```bash
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python -m pytest -q tests/test_table7.py
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/run_table7.py --mode plan
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/run_wn9_teacher.py --mode plan

git add scripts/run_wn9_teacher.py scripts/run_wn9_student.py \
  scripts/run_table7.py scripts/wn9_workflow.py scripts/eval_wn9_student.py \
  src/wn9_evaluation.py tests/test_table7.py \
  docs/experiments/TABLE7_WN9_REPRODUCTION.md
git diff --cached --check
git diff --cached --stat
git commit -m "exp(wn9): add separate teacher/student stages and Table 7 E2E workflow"
```

Full mode requires a clean tracked working tree and committed workflow sources.
Plans/smoke permit review of uncommitted sources and save source snapshots.
All commands below are **for the user to execute on GPU**, not assistant actions.

## Recommended E2E smoke, then full

```bash
./dsom/bin/python scripts/run_table7.py --mode smoke
```

Wait for `PASS E2E` and inspect `runs/table7/smoke/<SUITE>/summary.json`.
After smoke passes:

```bash
WANDB_PROJECT=DSoM-Reproduction \
./dsom/bin/python scripts/run_table7.py --mode full --wandb
```

This launches **one fresh teacher, then one fresh student**, sequentially, plus
checkpoint evaluation. No competitors are trained. W&B is opt-in, one run per
training stage, grouped under the E2E suite. W&B's last `test/*` entries need not
equal selected-checkpoint results; final raw/filtered metrics live in the saved
JSON/Markdown, not W&B's last epoch summary. Standalone evaluation does not open
an additional W&B run.

To run the learned-RC adaptation in a separate suite, append:

```text
--policy-optimizer Adam --policy-lr 0.00001
```

Use the same option on its E2E smoke before its full run. The default
source-frozen suite does not need to be rerun solely because this variant exists.

## Running the two stages independently

Teacher smoke:

```bash
./dsom/bin/python scripts/run_wn9_teacher.py --mode smoke
```

Then student smoke, using the teacher stage directory printed by that command:

```bash
./dsom/bin/python scripts/run_wn9_student.py --mode smoke \
  --teacher-run runs/table7/teacher/smoke/TEACHER_SUITE
```

For independent full training, first run a full teacher:

```bash
WANDB_PROJECT=DSoM-Reproduction \
./dsom/bin/python scripts/run_wn9_teacher.py --mode full --wandb
```

Then a full student from that teacher:

```bash
WANDB_PROJECT=DSoM-Reproduction \
./dsom/bin/python scripts/run_wn9_student.py --mode full --wandb \
  --teacher-run runs/table7/teacher/full/TEACHER_SUITE
```

Standalone student automatically runs checkpoint evaluation too. It saves
`stage_result.json` and `evaluation.json`; the E2E launcher additionally combines
the stage results into Table 7 Markdown. Standalone student plan requires an
already completed **full** teacher because plan previews full settings.
`--teacher-run` accepts a stage directory, not a checkpoint `.pth` path or an
arbitrary legacy teacher run. For a teacher produced by E2E, use its `teacher/`
subdirectory.

## Outputs and rechecking

E2E: `runs/table7/{plan,smoke,full}/<UTC timestamp>/` with `teacher/`, `student/`,
`summary.json`, `table7_results.md`, root/stage plans, source snapshots, commands,
training/evaluation logs, selected checkpoints, and integrity hashes.
Independent stages: `runs/table7/{teacher,student}/{mode}/<UTC timestamp>/`.
Every invocation creates a NEW directory and refuses overwriting. `--output`
can select a new directory within `runs/table7/`. There is no optimizer resume
or implicit GPU launch from a plan. Existing dataset/feature files are verified
before and after, not rewritten. Teacher checkpoint/results are protected during
student training.

Regenerate an already completed E2E summary with CPU/file checks only:

```bash
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/run_table7.py --mode summarize \
  --output runs/table7/full/YOUR_SUITE
```

`full_table7_dsom_row_completed=true` requires both full stages, fresh WN9
checkpoint linkage, complete checkpoint evaluation and metric agreement.
`exact_author_protocol_reproduced=false` remains explicit. Smoke always leaves
the full-row flag false and omits paper deltas.

CPU tests include a real tiny multimodal teacher optimizer step, teacher save/load,
frozen teacher/student distillation, source-frozen policy preservation, student
save/load, raw/filtered scoring compared with legacy `get_ranking`, NumPy uint64
filters, CLI parsing without executing CUDA setup, and separate/E2E orchestration
with mocked training processes. These do not replace the user's real GPU smoke.
