# Reproduce the paper tables

Start with [ENVIRONMENT.md](ENVIRONMENT.md) and [DATA.md](DATA.md). Commands below
assume the repository root, activated Python environment, verified features,
and a source revision containing these workflows. **Smoke/full/extract use GPU**;
plan and summarize are CPU/file checks. Full mode requires committed workflow
sources and no uncommitted tracked changes. Data, `.venv/`, and the local registry
are ignored, so creating them does not dirty the tracked tree.

These workflows reproduce DSoM rows/ablations, not all competitor training runs.
Model/optimizer/loss limits are in [PROTOCOL.md](PROTOCOL.md).
For commands that select one row/dataset at a time in every table, see
[Run one dataset/method cell in Tables 1–7](CELLS.md).

## Fresh baselines: Tables 1 and 3

Supported datasets: `MMKB-DB15K`, `MKG-W`, `MKG-Y`, `FB15K-237`, `WN18`.
Each `run_reproduction.py` E2E invocation trains its own new teacher then its own
new student. All datasets must be prepared before their invocation. WN9 uses the
separate Table 7 workflow below.

First choose a single prepared dataset, for example MKG-Y:

```bash
python scripts/run_reproduction.py --dataset MKG-Y --mode plan
python scripts/run_reproduction.py --dataset MKG-Y --mode smoke
```

After smoke passes, create/register a full baseline:

```bash
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_reproduction.py --dataset MKG-Y --mode full --wandb \
  --registry references.local.json
```

Repeat those commands with DB15K/MKG-W for Table 1 and FB15K-237/WN18 for Table 3.
For an explicitly requested sequential full grid after data and smoke checks:

```bash
for dataset in MMKB-DB15K MKG-W MKG-Y FB15K-237 WN18; do
  WANDB_PROJECT=DSoM-Reproduction \
  python scripts/run_reproduction.py --dataset "$dataset" --mode full --wandb \
    --registry references.local.json || break
done
```

Do not run this loop if a registry already has one of those baselines unless
you intend to choose a new registry first: duplicates are refused before
training. This is not optimizer resume. The experiment seed remains 2022.
Reference results are stored under
`runs/reproduction/DATASET/baseline/full/TIMESTAMP/student/checkpoints/0/result.json`.
Use `final_result` for the selected student's test metrics. Table 3's paper
reports MR/Hits, not MRR; do not invent a paper MRR target.

## Independent teacher/student stages

The same launcher supports separated stages:

```bash
python scripts/run_reproduction.py --dataset MKG-Y --stage teacher --mode smoke
python scripts/run_reproduction.py --dataset MKG-Y --stage student --mode smoke \
  --teacher-run runs/reproduction/MKG-Y/baseline/smoke/TEACHER_SUITE/teacher
```

Use the actual teacher stage directory printed by the first command, not the
`.pth` path. Independent full training uses `--mode full` for both stages, a full
teacher directory, and `--registry references.local.json` on the student command.
Full/plan student rejects a smoke teacher. E2E full always starts a new teacher;
it does not continue a smoke run.

## Explicit reference registry

The first completed full student creates `references.local.json`; optionally
start from `configs/references.example.json`. Registered paths are relative to
the repository. The schema is:

```json
{
  "schema_version": 1,
  "baselines": {
    "MKG-Y": "runs/reproduction/MKG-Y/baseline/full/YOUR_SUITE/student/checkpoints/0/result.json"
  },
  "table4": {},
  "table6": {}
}
```

This JSON is machine-local and ignored by Git. `--registry` on the fresh-stage
launcher adds completed full student references; `--references` on table launchers
consumes them. These flags have different purposes. The registry is not a
checkpoint and is never accepted in place of `--teacher-run`.
Do not change it concurrently while creating a plan. Each table plan saves the
registry contents/hash and hashes all selected result/data files, so later adding
unrelated entries does not change an existing suite's chosen artifacts.

Default historical reference lookup remains available when `--references` is
omitted. On a fresh clone, **pass the registry** instead of relying on those
old local directory layouts or selecting the newest run automatically.

## Table 2: selection and multi-teacher distillation

Requires registered full baselines and their teachers for the selected multimodal
datasets (all three for the complete table). The launcher reuses those full DSoM baseline rows and trains 12 new
students: ConfTeacher, BestTeacher, BestStrategy, MulDE-adapted × three datasets.

```bash
python scripts/run_table2.py --mode plan --references references.local.json
python scripts/run_table2.py --mode smoke --workers 1 --references references.local.json
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table2.py --mode full --workers 1 --wandb \
  --references references.local.json
```

Outputs: `runs/table2/full/SUITE/{summary.json,table2_results.md}`.
For a dataset subset, add `--datasets MKG-Y`; it is not a full Table 2 grid.
[Detailed protocol](../experiments/TABLE2_REPRODUCTION.md) includes the unpublished
MulDE adaptation choices. The reference DSoM row is source-frozen RC.

### Select one Table 2 cell

Use `--datasets` for the dataset column and `--methods` for the method row.
`--mode` controls execution (`plan`, `smoke`, `full`, `summarize`), not the method.
One dataset/method pair trains one new student and reports all MR/MRR/Hits metrics;
the selected dataset's DSoM baseline is also included as a reused reference row.
It reuses the baseline's full teacher rather than retraining the teacher.

| Paper method | CLI `--methods` |
|---|---|
| ConfTeacher | `best_conf` |
| BestTeacher | `best_teacher` |
| BestStrategy | `best_strategy` |
| MulDE (adapted) | `mulde` |
| DSoM reference | Reused from `baselines` in the registry; generated with `run_reproduction.py` |

For example, MKG-Y + ConfTeacher needs only a completed full MKG-Y baseline in
`references.local.json`. Generate/register that baseline using the earlier
teacher/student instructions before running this cell.

```bash
# File checks only; inspect plan.json and confirm exactly one job.
CUDA_VISIBLE_DEVICES='' python scripts/run_table2.py --mode plan \
  --datasets MKG-Y --methods best_conf --references references.local.json

# GPU execution check: one new student, one epoch.
python scripts/run_table2.py --mode smoke --workers 1 \
  --datasets MKG-Y --methods best_conf --references references.local.json

# GPU: one fresh full student after review/commit and a passing smoke.
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table2.py --mode full --workers 1 --wandb \
  --datasets MKG-Y --methods best_conf --references references.local.json
```

Multiple values form a Cartesian product: `--datasets MKG-Y MKG-W --methods
best_conf mulde` schedules four new students. Omitting `--methods` selects all
four methods; omitting `--datasets` selects all three datasets.
A successful single-cell full run has `status=PASS`, `planned_jobs=1`,
`completed_new_cells=1`, and `full_benchmark_grid_completed=false`; that last
flag is expected for a subset. Separate invocations create separate suites;
they are not automatically merged into one completed full-grid summary.

Selection in other launchers uses these options (their documented dependencies
and reference checks still apply):

| Workflow | Dataset / method selection |
|---|---|
| Tables 1/3 baseline | `run_reproduction.py --dataset MKG-Y` (or another supported dataset); `--stage teacher/student/e2e` |
| Table 4 reference students | `run_reproduction.py --dataset MMKB-DB15K --stage student --profile ...` |
| Table 4 component students | `run_table4_components.py --components nekd` or `nnkd`; DB15K only |
| Table 4 teacher ensembles | `run_table4_ensembles.py --methods avg`, `ml`, or `rc`; DB15K only |
| Table 6 | `run_table6.py --datasets MKG-Y --methods visual` (methods: `full`, `structural`, `visual`, `description`); existing references may be reused, use `--rerun-existing` to force a fresh selected student |
| Table 7 | WN9 only; `run_wn9_teacher.py` / `run_wn9_student.py` for separate stages |

## Table 4: six students plus three teacher rows

Use **one and the same full DB15K teacher** for all four reference profiles.
Get the stage directory from your registered baseline, without copying a
historical timestamp into source code:

```bash
DB_TEACHER=$(python - <<'PY'
import json
from pathlib import Path
registry = json.loads(Path('references.local.json').read_text())
result = json.loads(Path(registry['baselines']['MMKB-DB15K']).read_text())
print(Path(result['teacher_checkpoint']).resolve().parents[2])
PY
)
```

Then run four full reference students (GPU), each reusing that frozen teacher:

```bash
for profile in table4-average table4-rc-ndkd table4-rc-kd table4-rc-dkd; do
  WANDB_PROJECT=DSoM-Reproduction \
  python scripts/run_reproduction.py --dataset MMKB-DB15K --stage student \
    --teacher-run "$DB_TEACHER" --profile "$profile" --mode full --wandb \
    --registry references.local.json || break
done
```

The keys A4/A5/A9/A10 in the registry correspond respectively to average-NDKD,
learned-RC NDKD, learned-RC KL, and learned-RC DKD. All use the same matched
hyperparameters; policy Adam LR is 1e-5 for the learned rows. This is not the
Table 1 source-frozen full row. To smoke-test these profiles before full, use
the same student commands with `--mode smoke` and omit `--registry`; the full
teacher is permitted for student smoke.

Once the four **full** references exist, run the two component ablations:

```bash
python scripts/run_table4_components.py --mode plan --references references.local.json
python scripts/run_table4_components.py --mode smoke --references references.local.json
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table4_components.py --mode full --workers 1 --wandb \
  --references references.local.json
```

Select the actual completed component suite printed by that command:

```bash
T4_COMPONENT_SUITE=runs/table4_components/full/YOUR_COMPLETED_SUITE

python scripts/run_table4_ensembles.py --mode plan --references references.local.json \
  --student-suite "$T4_COMPONENT_SUITE"
python scripts/run_table4_ensembles.py --mode smoke --references references.local.json \
  --student-suite "$T4_COMPONENT_SUITE"
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table4_ensembles.py --mode full --wandb \
  --references references.local.json --student-suite "$T4_COMPONENT_SUITE"
```

The last suite's `table4_results.md` combines all nine rows and an additional
mean-logit diagnostic. Teacher Avg. interpretation is ambiguous; both mean
metrics and mean logits are saved. ML and standalone RC-on-val are adaptations,
not bit-exact author implementations. Details:
[components](../experiments/TABLE4_COMPONENT_ABLATIONS.md),
[teacher ensembles](../experiments/TABLE4_TEACHER_ENSEMBLES.md).

## Table 6: modality ablations

Requires full baselines for DB15K/MKG-W/MKG-Y. With a freshly generated registry,
the default reuses those 3 full cells and trains **9 new S/V/D students**.
The historical default without a registry reused two extra DB15K ablations;
that is why earlier local runs launched 7 new jobs. Fresh users need no A1/A2
history. Optional `table6` registry keys `MMKB-DB15K/structural` or
`MMKB-DB15K/description` can name independently reviewed existing results.

```bash
python scripts/run_table6.py --mode plan --references references.local.json
python scripts/run_table6.py --mode smoke --references references.local.json
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table6.py --mode full --wandb --references references.local.json
```

To train all 12 selected cells anew (teachers still reused), append
`--rerun-existing` consistently to plan/smoke/full. The three full-modality rows
remain source-frozen RC. Outputs: `runs/table6/full/SUITE/table6_results.md` and
`summary.json`. Do not interpret one seed or the source-frozen comparison as
evidence for learned-RC gains. [Recorded historical results](../experiments/TABLE6_REPRODUCTION.md).

## Table 7: WN9 E2E

Recorded locally: the user's full DSoM row completed and CPU artifact review
passed on 2026-10-24. Filtered MRR is .919634; raw MRR is .618763 on the same
checkpoint. See the [result and protocol limits](RESULTS.md#recorded-table-7-full-result).
The commands below describe a fresh reproduction, not required reruns of that suite.

First prepare WN9 as described in [DATA.md](DATA.md#wn9-from-pinned-mose--verified-wn18-features).
Its wrapper verifies parent/source hashes; it does not use this baseline registry.

```bash
python scripts/run_table7.py --mode plan
python scripts/run_table7.py --mode smoke
WANDB_PROJECT=DSoM-Reproduction python scripts/run_table7.py --mode full --wandb
```

It trains one new WN9 teacher, one new student, then reloads the selected student
for raw/filtered evaluation. No WN18 graph checkpoint is reused. Optional learned
RC is `--policy-optimizer Adam --policy-lr 0.00001`, in a separate suite and smoke.
Default is source-frozen. Separate stages are available:

```bash
python scripts/run_wn9_teacher.py --mode smoke
python scripts/run_wn9_student.py --mode smoke \
  --teacher-run runs/table7/teacher/smoke/YOUR_SUITE
```

Outputs: `runs/table7/full/SUITE/{summary.json,table7_results.md}`. The full-row flag
requires complete checkpoint evaluation, not just process exit 0. Both raw and
filtered results share the filtered-validation-selected checkpoint; the paper's
exact Table 7 protocol remains unresolved.
[Detailed WN9 workflow](../experiments/TABLE7_WN9_REPRODUCTION.md).

## Summaries without GPU

For completed suites:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/run_table2.py --mode summarize --output runs/table2/full/YOUR_SUITE
CUDA_VISIBLE_DEVICES='' python scripts/run_table4_components.py --mode summarize --output runs/table4_components/full/YOUR_SUITE
CUDA_VISIBLE_DEVICES='' python scripts/run_table6.py --mode summarize --output runs/table6/full/YOUR_SUITE
CUDA_VISIBLE_DEVICES='' python scripts/run_table7.py --mode summarize --output runs/table7/full/YOUR_SUITE
```

The teacher-ensemble Table 4 runner writes its summary at completion and does not
have a summarize mode. Generic baseline stages save `stage_result.json` and
training `result.json`. See [RESULTS.md](RESULTS.md) for correct metric selection.
