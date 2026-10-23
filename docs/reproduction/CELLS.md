# Run one dataset/method cell in Tables 1–7

Use this guide after [environment setup](ENVIRONMENT.md) and
[dataset/feature preparation](DATA.md). Commands assume the repository root and
an activated environment. These are commands for the user to execute; no training
is required to read or validate this guide.

A cell means one **method × dataset**, returning all its MR/MRR/Hits metrics.
Selecting just Hits@1 does not require a separate training run. Competitor rows
in Tables 1/3/7 are outside the implemented workflows. Table 5 contains dataset
statistics, not trainable models.

For every training command below, first replace `--mode full` with `--mode plan`
for CPU/file checks, then `--mode smoke` for a small execution check. Plan does
not launch training; smoke/full use GPU. Full starts fresh and requires committed
sources and a clean tracked tree. `--wandb` is optional; plan does not open W&B.
The examples do not resume smoke checkpoints.

`--registry` registers a completed full student; `--references` reads registered
results. Registration only takes place in full mode. Existing registry entries
are not overwritten: when intentionally retraining a registered baseline/profile,
use a separate registry filename. Reuse the same registry consistently for
dependent commands. See [reference setup](TABLES.md#explicit-reference-registry).

## Table 1: DSoM × one multimodal dataset

Choose exactly one of `MMKB-DB15K`, `MKG-W`, `MKG-Y`:

```bash
DATASET=MKG-Y
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_reproduction.py --dataset "$DATASET" --profile baseline \
  --stage e2e --mode full --wandb --registry references.local.json
```

This creates one fresh teacher followed by one fresh structural student, and
registers `baselines/DATASET`. It reproduces the source-frozen DSoM row, not the
other methods printed in the paper. If a full teacher is already available,
train only a fresh student using its actual **stage directory**, not a `.pth`:

```bash
TEACHER_RUN=runs/reproduction/MKG-Y/baseline/full/YOUR_SUITE/teacher
python scripts/run_reproduction.py --dataset MKG-Y --profile baseline \
  --stage student --teacher-run "$TEACHER_RUN" --mode full --wandb \
  --registry references.local.json
```

For a new independent teacher, use `--stage teacher` and omit `--registry`.
Full/plan students need a full teacher. Student smoke may use a full teacher.

## Table 2: one teacher-selection method × one dataset

Datasets: `MMKB-DB15K`, `MKG-W`, `MKG-Y`. Needs only the selected dataset's
registered full Table 1 baseline, including its full teacher and feature files.

| Paper row | `--methods` |
|---|---|
| ConfTeacher | `best_conf` |
| BestTeacher | `best_teacher` |
| BestStrategy | `best_strategy` |
| MulDE adaptation | `mulde` |
| DSoM | Reuse the Table 1 baseline; to retrain it, use the Table 1 command |

```bash
DATASET=MKG-Y
METHOD=best_conf
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table2.py --datasets "$DATASET" --methods "$METHOD" \
  --mode full --workers 1 --wandb --references references.local.json
```

One pair trains **one new student** with the existing teacher. The summary also
includes the selected dataset's reused DSoM baseline. Multiple datasets and methods
form a Cartesian product. `--mode` is execution mode, not the method selector.

## Table 3: DSoM × one structural benchmark

Choose exactly one of `FB15K-237`, `WN18`:

```bash
DATASET=WN18
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_reproduction.py --dataset "$DATASET" --profile baseline \
  --stage e2e --mode full --wandb --registry references.local.json
```

This creates one new teacher and one new source-frozen student. Separate stages
work exactly as in Table 1. The paper reports MR/Hits for Table 3; an available
local MRR is not a paper MRR target. Competitor retraining is not implemented.

## Table 4: one ablation row × DB15K

Table 4 has only `MMKB-DB15K`. Its nine rows use three entrypoints below.
This table's learned-RC profiles differ from the source-frozen main baseline.

### Four reference student rows

Use one shared full DB15K teacher from the portable workflow for every profile.
After the Table 1 DB15K E2E command, set the actual stage directory:

```bash
DB_TEACHER=runs/reproduction/MMKB-DB15K/baseline/full/YOUR_SUITE/teacher
```

| Row | `--profile` | Registry key |
|---|---|---|
| Average teacher + NDKD student | `table4-average` | `A4_average_NDKD` |
| RC + NDKD student | `table4-rc-ndkd` | `A5_RC_NDKD` |
| RC + vanilla KD student | `table4-rc-kd` | `A9_RC_KD` |
| RC + DKD student | `table4-rc-dkd` | `A10_RC_DKD` |

Choose **one** profile and run:

```bash
PROFILE=table4-rc-ndkd
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_reproduction.py --dataset MMKB-DB15K --stage student \
  --teacher-run "$DB_TEACHER" --profile "$PROFILE" --mode full --wandb \
  --registry references.local.json
```

This trains one fresh student and reuses the shared full teacher. Repeat with a
different profile only when that row is desired. All four full profiles must be
registered before using the current component/ensemble launchers, even when
requesting only one component or ensemble method; they load all four references.

### Two component student rows

Needs the four registered reference profiles above with matching DB15K teacher.

| Row | `--components` |
|---|---|
| RC + NEKD | `nekd` |
| RC + NNKD | `nnkd` |

```bash
COMPONENT=nekd
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table4_components.py --components "$COMPONENT" \
  --mode full --workers 1 --wandb --references references.local.json
```

This trains one new student. Its summary includes the four reference students;
`component_pair_completed=false` is expected for a one-component suite.

### Three teacher ensemble rows

The current launcher additionally requires **one completed full component-pair
suite**, with both NEKD and NNKD in the same suite. Two separate single-component
suites are not automatically combined. To create that prerequisite, run:

```bash
python scripts/run_table4_components.py --components nekd nnkd --mode full \
  --references references.local.json --wandb
```

If such a suite already exists, use it directly; there is no need to retrain it.
Choose one teacher method:

| Row | `--methods` | Behavior |
|---|---|---|
| Teacher Avg. | `avg` | Evaluation only; reports mean metrics and mean logits |
| Teacher + ML-on-val | `ml` | Trains one adapted validation fusion network |
| Teacher + RC-on-val | `rc` | Trains one adapted validation policy |

```bash
T4_COMPONENT_SUITE=runs/table4_components/full/YOUR_COMPLETED_PAIR_SUITE
METHOD=ml
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table4_ensembles.py --methods "$METHOD" --mode full --wandb \
  --references references.local.json --student-suite "$T4_COMPONENT_SUITE"
```

This selects one teacher method; it does not train another graph teacher or
student. Common teacher diagnostics and the existing student rows are still
reported. Teacher Avg. interpretation and ML/RC adaptations are documented in
[the full Table 4 workflow](TABLES.md#table-4-six-students-plus-three-teacher-rows).
The `avg` GPU evaluation is still invoked through smoke/full, despite no training.

## Table 5: dataset statistics

There is no model or training command for Table 5. Select and prepare a dataset
following [DATA.md](DATA.md), then verify it. For a standard prepared dataset:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/prepare_dataset.py \
  --dataset WN18 --verify-only
```

For WN9 use its source/feature verification instead:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/prepare_wn9_from_mose.py --mode verify
```

Entity/relation/split counts are listed in the data guide and preparation reports.
Count base relations before reciprocal augmentation. Preserve the documented
DB15K valid/test source-versus-paper count discrepancy; do not swap splits.

## Table 6: one teacher-modality ablation × one dataset

Datasets: `MMKB-DB15K`, `MKG-W`, `MKG-Y`. Needs the selected dataset's full
registered Table 1 baseline and teacher.

| Row | `--methods` |
|---|---|
| Full DSoM | `full` |
| S teacher | `structural` |
| V teacher | `visual` |
| D teacher | `description` |

```bash
DATASET=MKG-Y
METHOD=visual
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table6.py --datasets "$DATASET" --methods "$METHOD" \
  --rerun-existing --mode full --wandb --references references.local.json
```

`--rerun-existing` guarantees one **new student** for the selected pair, including
`--methods full`; it still reuses the existing full teacher. Without this flag,
the launcher may reuse an existing cell, especially the full DSoM baseline.
Use the flag consistently for plan/smoke/full if the intention is fresh training.
Every ablation still trains a structural student; the method selects its teacher
modality. The Table 6 launcher does not automatically register newly trained cells.

## Table 7: DSoM × WN9

WN9 is the only dataset and DSoM the only implemented row. Its E2E command is
already a single-cell workflow: one new teacher, then one new student, then
raw **and** filtered evaluation on the selected student checkpoint.

```bash
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table7.py --mode full --wandb --policy-optimizer source_frozen
```

To run the stages separately:

```bash
python scripts/run_wn9_teacher.py --mode full --wandb
# Set the actual completed full teacher stage printed above.
WN9_TEACHER=runs/table7/teacher/full/YOUR_SUITE
python scripts/run_wn9_student.py --mode full --wandb \
  --teacher-run "$WN9_TEACHER" --policy-optimizer source_frozen
```

Standalone student plan/full requires an already completed full teacher; it cannot
plan against a nonexistent placeholder directory. WN18 graph checkpoints are
not valid WN9 teachers. Optional `--policy-optimizer Adam --policy-lr 0.00001`
selects a separate learned-RC experiment, not the recorded source-frozen row.
Raw/filtered are evaluation settings, not different training commands. See the
[completed result and protocol limits](RESULTS.md#recorded-table-7-full-result).

## Interpreting subset completion

Each invocation creates a separate suite. A passing selected cell does not mean
the full table is complete; pair/grid flags can correctly remain false. Separate
cell suites are not automatically merged into a complete table report. Preserve
each plan, result, source snapshot and checkpoint; use the selected validation
checkpoint's metrics, not the last W&B test log entry. Output paths and summary
commands are in [TABLES.md](TABLES.md) and [RESULTS.md](RESULTS.md).
