# Table 2 — teacher-selection reproduction workflow

Status: **code and CPU checks ready; GPU smoke/full training NOT run**.
The user will review, commit, and execute training. The default launcher mode
only writes a plan. No new Table 2 benchmark scores are claimed here.

## Scope and reference

Target: [DSoM v1, Table 2 and Section 5.4](https://arxiv.org/html/2507.20738v1).
Run four Student comparisons on each of MMKB-DB15K, MKG-W, MKG-Y:

| CLI method | Report label | Teacher selection / supervision |
|---|---|---|
| `best_conf` | ConfTeacher | Highest maximum softmax probability among S/V/D; then NDKD |
| `best_teacher` | BestTeacher | Lowest ground-truth cross-entropy among S/V/D; then NDKD |
| `best_strategy` | BestStrategy | Lowest cross-entropy among seven S/V/D subset means; then NDKD |
| `mulde` | MulDE-adapted | Student top-K queries; trainable Senior scaling and contrast attention; KL + Senior BCE |

These are 12 new Student runs. The existing Exp01 source-frozen RC results
provide the three reference cells, making 15 method/dataset cells in total.
Only the Student is evaluated at test time; label-dependent teacher selection
is confined to training batches. NDKD neighbors come from training triples
and their reciprocals, through `Dataset.train_to_skip`.

Baseline result JSONs are resolved from each dataset's existing
`runs/student/DATASET/full/` directory. Their exact teacher checkpoints,
feature paths, and training settings are copied into a reviewable plan.
No Teacher retraining is needed. The launcher refuses ambiguous baselines.

The original benchmark split is used for DB15K, with its known contaminated
validation. **CleanV1 and all prior results remain separate.** The reused DSoM
reference is explicitly labeled `DSoM-source-frozen`; it is not a learned-RC
reference. Paper targets for all four metrics are stored as fractions in
[table2_paper_targets.json](table2_paper_targets.json).

## What is matched

From each corresponding Exp01 Student result: rank, batch size, optimizer,
learning rate, regularizer and strength, scheduler, temperature, CE/KD weights,
alpha/beta, initialization, scaling, optimizer betas, epoch limit, validation
interval, and early stopping. Current runs use rank 2000, batch size 1000,
Adagrad LR 0.1, N3 0.01, temperature 4, CE/KD weights 1/2, max 200 epochs,
validation every 3 epochs, and seed 2022.

The existing `min` scheduler monitoring validation Hits@10 is retained to
match the baseline, including its known direction mismatch. Checkpoints are
selected by validation MRR. The result table reads `final_result`, not the last
epoch's `test` field or a W&B summary. Test metrics logged by existing training
code do not select checkpoints.

The three heuristic classes preserve the released selection behavior:
ConfTeacher uses raw-score softmax at temperature 1; BestTeacher averages
teachers tied for minimum CE; BestStrategy uses the first minimum among
`S,V,D,SV,SD,VD,SVD`, with arithmetic-mean logits for each combination.
`BestStrategy.strategy` is now a registered buffer so it follows `.to(device)`.

## MulDE adaptation boundary

The public [DSoM source](https://github.com/OreOZhao/DSoM), inspected at commit
`171e59cb069d50abf039fa824b079e23b8956696`, contains the heuristic selectors but
does not supply its MulDE adaptation or the missing adaptation settings.
The source tree listing was saved under `runs/audit/table2_sources/`.

The implemented comparison follows [original MulDE, Sections 3.2–3.4,
Equations 7 and 10–14](https://arxiv.org/html/2010.07152v3), with these explicit
choices for the multimodal DSoM setting:

- Three pretrained ComplEx modality branches replace the original hyperbolic
  model teachers. They remain frozen in every training step.
- Junior selects its own top 300 entity candidates (`min(300, entity_count)`),
  without forcing the positive answer into the list.
- Senior has a sigmoid relation-specific scaling matrix, including reciprocal
  relation IDs, initialized to zero before sigmoid (initial scale 0.5).
- Contrast attention uses the unscaled teacher top-K distributions and detached
  Junior distributions. Equation 11 averages KL over K. Its temperature is
  `exp(floor(e/5))` for zero-based epoch `e`; Equation 12 multiplies attention
  weights by the teacher count, three.
- Senior's BCE labels indicate whether the true training target occurs in the
  top-K list. If absent, the entire label vector is zero. Equation 13 sums over
  K and averages over queries.
- The Student keeps DSoM's full-entity CE and N3 rather than original MulDE's
  sampled hard-label BCE. Soft supervision is temperature-4 KL on top-K with
  weight 2, not NDKD. Senior BCE has weight 1. KL and Senior BCE jointly update
  Senior; the attention's Junior input is detached.
- Student uses the matched Adagrad optimizer; Senior has separate Adam at
  0.001, without the Student scheduler. Both optimizer memberships are checked,
  and the first epoch must show a nonzero Senior weight update.

This is **MulDE-adapted**, not a verified exact recreation of the authors'
unpublished DSoM/MulDE adaptation. Its numbers can be compared with the paper
as a reference, but must retain the adaptation label and these assumptions.
No hyperparameter search or test-driven tuning was performed.

## CPU review and commit

Run from the repository root. These commands do not train on a GPU:

```bash
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python -m pytest -q \
  tests/test_table2.py tests/test_db15k_clean_split.py

CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/run_table2.py --mode plan

git diff --check
git status --short
```

CPU tests forbid CUDA initialization. They cover mathematical selection,
tie behavior, a pair outperforming every single teacher, Senior formulas,
gradient flow, frozen Teachers, actual optimizer branches on tiny synthetic
batches, the true CLI parser without importing the training entry point,
plan-only behavior, the commit guard, correct best-checkpoint metric selection,
missing checkpoints, and original-input hash preservation.

Review the pending changes and commit before full training. There are also
uncommitted files from the earlier CleanV1 task; choose commits deliberately.
The launcher never commits files. Full mode requires a clean **tracked** tree
and checks that the relevant source/test/target/protocol files are tracked.
An untracked PDF does not block it.

## Commands for the user to run after review/commit

Start with GPU smoke runs; these were intentionally left to the user:

```bash
./dsom/bin/python scripts/run_table2.py --mode smoke --workers 1
```

After all 12 smoke cells pass, run the full suite:

```bash
./dsom/bin/python scripts/run_table2.py --mode full --workers 1
```

For a first check on one smaller dataset or selected methods:

```bash
./dsom/bin/python scripts/run_table2.py --mode smoke --datasets MKG-Y
./dsom/bin/python scripts/run_table2.py --mode full --datasets MKG-Y
```

To select exactly one new student, also pass `--methods best_conf` (or
`best_teacher`, `best_strategy`, `mulde`). Dataset and method lists form a
Cartesian product. `--mode` is the execution mode, not the method selector.
Fresh checkouts must pass `--references references.local.json` containing the
selected dataset's completed full baseline. See the
[single-cell recipe](../reproduction/TABLES.md#select-one-table-2-cell).

Add `--wandb` to enable W&B. Without it, logging is local even if an inherited
`DSOM_WANDB` variable is set. Use `CUDA_VISIBLE_DEVICES` in the shell to choose
the device. `--workers 1` is the default; a larger number starts concurrent
training processes and requires sufficient available GPU memory.

Outputs use new timestamped directories under `runs/table2/{plan,smoke,full}/`.
`--output runs/table2/full/CUSTOM_NAME` optionally selects a new directory.
Existing directories are never overwritten. Every job has `command.sh`,
`train.log`, and its own checkpoints; the suite also records source snapshots,
source/input SHA256 digests, Git HEAD, environment versions, baseline paths,
and a working-tree patch. No original checkpoint or dataset is written.

A failed job prevents queued jobs from starting; already running jobs may
finish. The summary marks missing jobs explicitly rather than treating the
table as complete. There is no optimizer-state resume; use dataset/method
filters in a new run directory for a failed cell.

## Results and status

After training, `summary.json` and `table2_results.md` contain the selected
Student's MRR and Hits@1/3/10, paper reference values, result paths, and MRR
differences. Smoke mode does not compute paper-score differences. All input
and baseline hashes are rechecked before summarization.

To regenerate a summary without running training:

```bash
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/run_table2.py \
  --mode summarize --output runs/table2/full/YOUR_RUN_DIRECTORY
```

`status=PASS` means the requested cells passed. A subset run can pass without
finishing all of Table 2. Only `full_benchmark_grid_completed=true` signifies
that all 12 new full-training cells are present in that suite. The result still
retains the source-frozen/MulDE-adapted limitations, reconstructed features,
single seed, and DB15K validation contamination.

Current execution status: CPU tests and plan checks only. **GPU smoke and full
benchmark results are pending the user's runs.**
