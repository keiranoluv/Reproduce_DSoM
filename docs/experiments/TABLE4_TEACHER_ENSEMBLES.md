# Table 4 — remaining teacher rows and protocol limits

Preparation only: the assistant ran CPU checks, not GPU jobs. The user runs
smoke/full after reviewing and committing these files. Existing source training
paths and results are preserved; the new module is imported only by this runner.

## Completed student audit

`runs/table4_components/full/20261023_075850_052725/summary.json` was regenerated
with CPU/file checks. Status PASS; original input hashes match, component configs
match A5, selected-checkpoint tests match the corresponding curve entries,
student checkpoints exist, and both runs recorded nonzero policy updates.

| Component | Best epoch | Test MRR | H@1 | H@3 | H@10 | Paper MRR | MRR delta (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RC + NEKD | 75 | .411020 | .333115 | .450010 | .560038 | .4100 | +.102 |
| RC + NNKD | 42 | .412105 | .332256 | .452030 | .563775 | .4233 | -1.120 |

Six student rows now have results, including A4/A5/A9/A10. The policy is learned
Adam, not source-frozen. NEKD uses released-code neighbor SUM, not paper MEAN.
NNKD exceeds NEKD by .1084 pp MRR, much less than the paper's 1.33 pp.
These checks do not rerun checkpoint inference on the GPU.

## Sources and interpretation

[DSoM §5.3/Table 4](https://arxiv.org/html/2507.20738v1#S5.SS3) describes Teacher
Avg. as the average performance of three modal teachers. Earlier planning
treated it as mean-logit ensemble inference; that interpretation is uncertain.
This runner saves BOTH the arithmetic mean of individual modality metrics
(`teacher_mean_metrics`, primary literal interpretation) AND ranking the mean
raw logits (`teacher_mean_logits`, diagnostic). It also saves each modality's
metrics. Neither is the oracle minimum rank used by the original teacher eval.

[MoSE's released meta learner](https://github.com/OreOZhao/MoSE4MKGC/blob/192d9ff2882b5be6ef8239a37cbd90c5dd102b5a/src/meta_learner.py)
applies Linear(3,32), ReLU, Linear(32,1), ReLU independently to each entity's
three raw modality scores. It uses validation triples, both directions, Adagrad
LR .1, batch 256, 20 epochs. This architecture and schedule are retained; the
runner fixes the source's unrestricted squeeze to preserve singleton batches.
The released MoSE code selects checkpoints using TEST Hits@10. Here the final
epoch is fixed before execution and the test set is evaluated only after fitting.
Thus this is a MoSE architecture adaptation, not a bit-exact source run.
Source commit and downloaded SHA256 are recorded in `table4_ensemble_sources.json`.

DSoM does not provide sufficient standalone RC-on-validation implementation
details in the inspected release/paper. The new **RC-on-val-adapted** protocol is:

- Freeze the existing Exp01 teacher; no student is created or trained.
- Teacher raw scores [batch, entities, S/V/D] enter the DSoM-style policy:
  flatten, dropout .1, Linear(3*entities,1024), ReLU, Linear(1024,7).
- Seven actions: S, V, D, SV, SD, VD, SVD. Multi-modal actions mean raw logits.
- Sample an action during training. If its unfiltered target CE beats the
  average-teacher CE, reward +1; otherwise -10. Average-reference reward is -10,
  hence REINFORCE advantage is 11 for improvements and 0 otherwise. No entropy
  term or additional student comparison. Stable Categorical log probabilities.
- Train Adam LR 1e-5 for a fixed 20 epochs on validation RHS then LHS. Use seed
  2022, retain validation duplicates, shuffle each direction each epoch.
- Inference is deterministic argmax action with dropout off; no labels enter
  action selection. Apply filtered ranking only AFTER score fusion.

This reward reference and training horizon are explicit reproduction choices,
not verified author settings. A5 student-policy checkpoints are not reused.
Full training uses all original validation entries, including known contamination.
Test labels are used for metrics/filtering only. Validation metrics after fitting
are in-sample diagnostics, not generalization estimates. No hyperparameter search
or test-based checkpoint selection is performed.

## Review and launch commands

```bash
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python -m pytest -q \
  tests/test_table4_ensembles.py tests/test_table4_components.py tests/test_table2.py

CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/run_table4_ensembles.py --mode plan

git add src/table4_ensembles.py scripts/run_table4_ensembles.py \
  tests/test_table4_ensembles.py docs/experiments/TABLE4_TEACHER_ENSEMBLES.md \
  docs/experiments/table4_ensemble_sources.json \
  docs/experiments/TABLE4_COMPONENT_ABLATIONS.md
git diff --cached --check
git diff --cached --stat
git commit -m "exp(db15k): prepare Table 4 teacher ensemble ablations"

# User executes GPU smoke: both heads fit 1 epoch, 64 queries per direction.
./dsom/bin/python scripts/run_table4_ensembles.py --mode smoke

# User executes all three rows sequentially, W&B enabled.
WANDB_PROJECT=DSoM-Reproduction \
./dsom/bin/python scripts/run_table4_ensembles.py --mode full --wandb
```

Default is plan only: stdlib metadata/hashing, no Torch import or execution.
Full mode requires the source files committed and a clean tracked working tree.
There is no workers flag; methods run sequentially. Use `--methods avg`,
`--methods ml`, or `--methods rc` for a single row; a partial suite is not marked
complete. `--ml-epochs` and `--rc-epochs` explicitly change the fixed horizon;
keep defaults for the documented experiment. W&B is opt-in and creates one run
per method. No hidden DSOM_WANDB environment behavior.

Every invocation uses a fresh directory in `runs/table4_ensembles/{mode}/`.
`--output` can choose another NEW directory within that tree; existing directories
are refused. Full source snapshots, input hashes, config, per-head training
history, final head/optimizer state, and results are saved. Only the heads are
checkpointed; teacher/features/data are not duplicated. Existing benchmark
artifacts are hashed before and after. The default student suite is the completed
20261023_075850_052725 run; `--student-suite` overrides it.

Outputs: each method's `result.json`, ML/RC `final_head.pth` and `history.json`,
suite `summary.json`, and combined `table4_results.md` with the six existing
student rows. A full three-row run sets `all_nine_rows_have_results=true`, but
`exact_author_protocol_reproduced=false` remains: ML checkpoint selection and
standalone RC settings are adaptations, Teacher Avg is ambiguous, original
DB15K validation overlaps remain, and this is one seed with reconstructed features.

CPU tests cover reciprocal queries/duplicates, pessimistic filtered ties,
independent ranking checks, mean-metric versus mean-logit distinction, MoSE
architecture, RC advantage, label-free action selection, validation-only head
updates, frozen teacher weights, saved-head reload, default plan, and full guards.
