# Table 6 — single-modality teacher ablation

Status: user completed full training on 2026-10-23; CPU/file verification **PASS**,
with all 12 cells present. Results are recorded provisionally below. The assistant
has not launched GPU jobs; existing benchmark artifacts are preserved.

## Recorded full run — 2026-10-23

`runs/table6/full/20261023_110036_163721/` completed with status **PASS**:
**12/12 cells**, comprising 7 newly trained students and 5 verified reused results.
Training source commit: `f717d2444d05bbbe3782c78f5cb8728c6fe7812b`. Seed: 2022.
The seven new jobs took approximately 33 minutes in total.
A portable metric snapshot, artifact hashes, and result paths are saved in
[table6_results_20261023.json](table6_results_20261023.json).

| Dataset | Method | Reused | Best epoch | Test MRR | H@1 | H@3 | H@10 | Paper MRR | MRR delta (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MKG-Y | full | True | 42 | 0.406919 | 0.360120 | 0.424146 | 0.489861 | 0.4002 | +0.672 |
| MKG-Y | structural | False | 18 | 0.382687 | 0.349230 | 0.400864 | 0.441795 | 0.3833 | -0.061 |
| MKG-Y | visual | False | 33 | 0.375854 | 0.344536 | 0.389974 | 0.429215 | 0.3392 | +3.665 |
| MKG-Y | description | False | 27 | 0.384521 | 0.333646 | 0.403305 | 0.472212 | 0.3729 | +1.162 |
| MKG-W | full | True | 54 | 0.405840 | 0.335400 | 0.431914 | 0.537319 | 0.4090 | -0.316 |
| MKG-W | structural | False | 15 | 0.366692 | 0.308961 | 0.386640 | 0.478708 | 0.3681 | -0.141 |
| MKG-W | visual | False | 3 | 0.295746 | 0.247309 | 0.315863 | 0.384183 | 0.3118 | -1.605 |
| MKG-W | description | False | 30 | 0.400567 | 0.326158 | 0.428638 | 0.542583 | 0.3923 | +0.827 |
| MMKB-DB15K | full | True | 63 | 0.413046 | 0.331701 | 0.454656 | 0.565744 | 0.4268 | -1.375 |
| MMKB-DB15K | structural | True | 48 | 0.412724 | 0.333922 | 0.451878 | 0.562563 | 0.4134 | -0.068 |
| MMKB-DB15K | visual | False | 45 | 0.381125 | 0.306201 | 0.420521 | 0.525449 | 0.3736 | +0.753 |
| MMKB-DB15K | description | True | 18 | 0.381768 | 0.304938 | 0.418552 | 0.527924 | 0.4130 | -3.123 |

CPU/file verification regenerated the suite summary, rechecked input/reference
SHA256s and snapshotted sources, and verified all seven full configs and nonempty
student checkpoint files. Every selected epoch is the first global maximum of
recorded validation MRR; its test curve entry equals `final_result`. All recorded
training losses are finite and all seven logs contain W&B run links. This review
did not perform fresh checkpoint inference or launch GPU jobs. Evidence is saved
in the suite's `cpu_review.json`.

Observed outcomes:

- Structural-only MRR is close to the paper on all three datasets: mean absolute
  error 0.090 percentage points.
- Largest single-teacher discrepancies: MKG-Y visual +3.665 pp, DB15K description
  -3.123 pp, and MKG-W visual -1.605 pp versus paper. Their causes remain unresolved.
- The full-modality row has the highest MRR on all three datasets. Its gain over
  the best single teacher is +2.240 pp on MKG-Y, +0.527 pp on MKG-W, and only
  +0.032 pp on DB15K. These single-seed differences do not establish significance.
- Full-modality does not dominate every Hits metric: DB15K structural has higher
  H@1, and MKG-W description has higher H@10.
- MKG-W visual selected epoch 3 and stopped after 14 epochs; all seven new runs
  ended with early stopping, without a detected configuration/execution failure.

Provisional conclusion: the Table 6 grid is complete under the documented
protocol; numerical agreement with the paper is mixed. The full rows remain
`source_frozen`, so these results do not establish learned-RC gains. Existing
feature-equivalence, neighbor SUM/MEAN, DB15K validation, and single-seed limits
remain. No corrective retraining has been launched.

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
