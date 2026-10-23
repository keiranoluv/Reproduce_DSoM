# Table 6 — single-modality teacher ablation

Status: launcher prepared and checked on CPU. The assistant does not launch GPU
jobs; the user reviews, commits, and runs smoke/full. No existing run is overwritten.

## Scope

[DSoM v1, Table 6 and Appendix E](https://arxiv.org/html/2507.20738v1#A5)
compare full DSoM with structural (S), visual (V), and description (D) teachers
on DB15K, MKG-W, and MKG-Y: 4 rows × 3 datasets = 12 metric cells.
Paper targets for MRR and H@1/3/10 are in `table6_paper_targets.json`.

The launcher reuses the existing Exp01 multimodal teacher checkpoint for each
dataset, with frozen parameters. A single-teacher run selects ONLY that
modality's raw logits; every run still trains a structural ComplEx student
from scratch with CE + N3 + NDKD. It does not train a visual student, remove
feature files, or retrain the teacher. All teacher modalities are loaded because
the existing fixed-strategy distiller computes all three outputs.

The only core changes are additive: `visual: (0,1,0)` in
`KD_FixedStrategy.STRATEGIES`, and `visual` in the training CLI's choices.
Existing strategies, defaults, optimizer branches, and loss functions are unchanged.

## Default reuse and fresh runs

| Dataset | Full-modality row | S teacher | V teacher | D teacher |
|---|---|---|---|---|
| MKG-Y | Reuse Exp01 | New | New | New |
| MKG-W | Reuse Exp01 | New | New | New |
| MMKB-DB15K | Reuse Exp01 | Reuse A1 | New | Reuse A2 |

Default full training launches **7 new student jobs**, sequentially in the order
shown, and reuses 5 cells. DB15K A1/A2 are reused only after checking the teacher,
features, matched hyperparameters, full-data limits, checkpoint presence, and
best-validation checkpoint's test metrics. Ambiguous original references fail
instead of silently selecting a run. The launcher does not automatically reuse
arbitrary subsequent Table 6 runs.

`--rerun-existing` launches **12 new students**, including the three full-modality
source-frozen baselines. Teachers are still reused. This option creates new
directories and preserves the original results. By default smoke also launches
the same 7 missing jobs, using 1 epoch / 1000 train examples / 64 evaluation
queries per direction. Reused full results are labelled references; smoke scores
have no paper delta and never mark the full table complete.

All configs copy each dataset's original Exp01 settings: rank 2000, Adagrad .1,
batch 1000, N3 .01, temperature 4, CE/KD weights 1/2, NDKD alpha/beta 1/1,
max 200 epochs, validation every 3 epochs, early stopping 10, seed 2022.
Single-modality jobs use `fixed_ndkd` and no RC policy. The original `min`
scheduler direction is retained for comparison with existing runs.
Training neighbor masks come from training triples and reciprocals only;
validation/test filtering uses the existing benchmark filters.

## Interpretation limits

The full-modality references are the same **source_frozen RC** results used in
Tables 1/2. They do not establish the effectiveness of learned RL. DB15K's learned
A5 checkpoint from Table 4 is a different protocol and is not substituted.
With `--rerun-existing`, the full row retains `source_frozen`, rather than silently
switching to learned RC. Fresh and reused results have the same intended configs,
but policy initialization can change minibatch RNG trajectories relative to a
fixed-teacher job.

NDKD preserves the released neighbor SUM definition; the paper describes MEAN.
Original DB15K validation overlap, reconstructed features, and single-seed limits
remain. Completing the 12 cells is a complete source-faithful grid under this
documented protocol, not an exact reproduction of the authors' private runs.

## Review and user-run commands

```bash
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python -m pytest -q tests/test_table6.py

# CPU-only plan: hashes inputs, snapshots source, writes commands; starts no jobs.
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/run_table6.py --mode plan

git add src/kd.py src/kd_learn.py scripts/run_table6.py tests/test_table6.py \
  docs/experiments/TABLE6_REPRODUCTION.md \
  docs/experiments/table6_paper_targets.json
git diff --cached --check
git diff --cached --stat
git commit -m "exp: add Table 6 modality ablation workflow"

# User runs these GPU commands, in order.
./dsom/bin/python scripts/run_table6.py --mode smoke

WANDB_PROJECT=DSoM-Reproduction \
./dsom/bin/python scripts/run_table6.py --mode full --wandb
```

For a fresh run of all 12 cells, use `--rerun-existing` in plan, smoke, and full:

```bash
WANDB_PROJECT=DSoM-Reproduction \
./dsom/bin/python scripts/run_table6.py --mode full --rerun-existing --wandb
```

The runner is sequential and has no `--workers` flag. W&B is opt-in, one run per
new job, in a shared group. No new W&B run is created for reused cells.
The first failed job stops queued jobs and leaves logs/results for inspection.
There is no optimizer resume. To retry a failed cell in a NEW suite, use filters:

```bash
WANDB_PROJECT=DSoM-Reproduction \
./dsom/bin/python scripts/run_table6.py --mode full \
  --datasets MKG-Y --methods visual --wandb
```

Subset suites are labelled partial and never claim the full table is complete.
Results from separate subset suites are not automatically merged. The normal
full command is recommended for one combined table.

## Artifacts and verification

Each invocation writes a NEW `runs/table6/{plan,smoke,full}/<UTC timestamp>/`.
`--output` accepts a new directory within `runs/table6/`; existing directories
are refused. Full mode requires committed workflow files and a clean tracked tree.
Saved artifacts include `plan.json`, config/commands, source snapshot, input
SHA256s (teacher, features, data, maps, reference results), per-job `train.log`,
checkpoints, `summary.json`, and `table6_results.md`. Original artifacts are
checked again at summary time. Selected tests use `final_result` from the best
validation-MRR checkpoint, not the last W&B `test/*` values.

Regenerate a summary with CPU/file checks only:

```bash
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/run_table6.py --mode summarize \
  --output runs/table6/full/YOUR_SUITE
```

`full_table6_grid_completed=true` requires all 12 full-data cells, including
verified reused references; smoke/partial suites cannot set it.
`exact_author_protocol_reproduced=false` explicitly records the limits above.
Paper targets are read from the suite's source snapshot during summarization.

CPU tests exercise real fixed-NDKD optimizer updates for S/V/D and the existing
pair/average strategies, correct logit selection, frozen teachers, all generated
training CLI variants, reuse validation, seven/default versus twelve/fresh jobs,
input integrity, selected-checkpoint metrics, failed-job stopping, and summary
completion flags. Synthetic optimizer tests redirect legacy `.cuda()` transfers
to CPU and forbid CUDA initialization.
