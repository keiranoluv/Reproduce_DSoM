# Results, integrity, and troubleshooting

## Recognize success

Process exit 0 alone is not the full scientific check. For the portable baseline
launcher, each completed stage has `stage_result.json` with `status=PASS`, result
and checkpoint hashes. For table suites, inspect their `summary.json`, `failures`,
and full-grid/row flag. Smoke flags never certify a full benchmark result.

| Workflow | Final artifacts |
|---|---|
| Portable baseline/Table 4 reference profile | `teacher/stage_result.json`, `student/stage_result.json`, selected training `result.json` |
| Table 2 | `summary.json`, `table2_results.md` |
| Table 4 component pair | `summary.json`, `table4_student_results.md` |
| Table 4 teacher ensembles | `summary.json`, combined `table4_results.md` |
| Table 6 | `summary.json`, `table6_results.md` |
| Table 7 | `summary.json`, `table7_results.md`, `student/evaluation.json` |

Only Table 7's workflow independently reloads and re-evaluates the selected
student automatically. Other table launchers validate saved configs, file hashes,
checkpoint presence and selected curve entries; they do not claim fresh checkpoint
inference unless it was performed separately.

## Correct metric selection

For students use `result.json["final_result"]`, corresponding to the checkpoint
selected by maximum **validation MRR**, not `result.json["test"]` or the last W&B
`test/*` values. `best_epoch` identifies that checkpoint. Teacher selection in
`learn.py` is maximum validation Hits@10 despite some historical messages saying
"best Valid MRR". With modality split, teacher ranks take the minimum over
modalities and are an oracle upper-bound diagnostic, not deployable inference.

Read registered baseline student results:

```bash
python - <<'PY'
import json
from pathlib import Path
registry = json.loads(Path('references.local.json').read_text())
for dataset, filename in registry['baselines'].items():
    result = json.loads(Path(filename).read_text())
    print(dataset, 'best epoch:', result['best_epoch'], result['final_result'])
PY
```

Metrics are fractions: `.413` MRR equals `41.3%`. A difference of `.01` equals
**one percentage point**, not one percent relative improvement. Table 3's paper
does not provide MRR; compare MR and Hits there. Single-seed differences do not
establish statistical significance. WN9 raw/filtered results use the same saved
student, and both settings must be labelled.

## Checkpoint and data handling

All training/extraction outputs go to new directories or refuse overwriting
final files. A full command starts fresh, not from smoke. There is no implicit
optimizer resume. Frozen teacher checkpoints, features, data/maps and selected
reference result files are hashed and checked again where the launcher supports
it. Preserve source snapshots/manifests along with JSON results.

Full-object torch checkpoint loading uses `weights_only=False` for compatibility
with locally saved models. Load only checkpoints you created or otherwise trust;
do not replace local checkpoints with arbitrary downloaded pickle files.
Teacher-only stages cannot be registered as student baselines. Full students
must use full teachers with matching dataset and input hashes.

The registry snapshots which artifacts were selected; adding a new baseline to
the registry later does not edit old suite plans. Moving datasets/checkpoints
to another filesystem path can require reconfiguration: old manifests often
record absolute paths. Do not rewrite an old manifest to make a verification
pass; create a new documented preparation/run or preserve the old path layout.

## Common failures

| Symptom | Action |
|---|---|
| `No module named torch/transformers/...` | Activate the intended venv; run `check_environment.py` with the relevant groups and `python -m pip check` |
| Child Python under `dsom/` missing | Use current Python launchers from this guide; historical Bash commands can retain that path |
| `Commit tracked changes` / untracked workflow source | Review and commit source/config changes before full mode; no need to commit ignored data/runs |
| Missing baseline/reference or old timestamp | Pass `--references references.local.json`; generate/register the required full students first |
| Duplicate registry entry | Select a new registry for a fresh experiment; existing entry is deliberately not overwritten |
| Existing dataset/feature/run output | Verify it, or use a separate fresh checkout/output; do not delete benchmark results automatically |
| Source mapping/count/coverage mismatch | Check exact dataset version, filenames and ent_links/SameAs mapping; do not swap splits or use H5 vectors to bypass it |
| Google Drive/Baidu cannot download | Use the linked original source manually; retain exact filenames and record availability; no silent fallback |
| Missing encoder weights offline | Populate Hugging Face/torch caches on an allowed network before extraction |
| Feature checkpoint fingerprint mismatch | Preserve the old checkpoint; do not resume with changed raw inputs/mapping/backend |
| CUDA/driver/VRAM error | Run the explicit environment GPU check on your host; smoke at documented rank, inspect training logs; changing batch/rank creates a different protocol |
| W&B authentication error | Authenticate, use `WANDB_MODE=offline`, or omit `--wandb`; local results do not require tracking |
| Teacher smoke rejected by full student | Train a new full teacher first; full never continues smoke |
| Selected-checkpoint/metric/hash mismatch | Stop and inspect the saved plan/logs; do not report or overwrite the suspect artifacts |

Failed stages save logs, and supported orchestration stops queued dependent jobs.
The portable feature adapter checks paths before importing a backend; its audit
does not decode every image or establish feature equivalence with the paper.

## What was validated for this guide?

The installed environment and public availability of direct pins were inspected.
A new isolated Python 3.12 venv installed the CPU torch build and all requirement
groups; `pip check` and all 160 tests passed. All ten historical encoder modules
imported, and a tiny randomly initialized BERT completed a CPU forward without
pretrained downloads. Tests also include tiny real teacher/student optimizer steps
and mocked orchestration; no full GPU experiment was executed.

A relocated local clone with spaces in its path, no historical runs, and no
repo-local `dsom/bin/python` passed environment checks, feature verification and
baseline planning. This test linked existing prepared MKG-Y features rather than
redownloading/re-encoding them. The new portable feature adapter and generic
baseline entrypoint have not been run through full extraction/training on a new
GPU machine. Historical core runs and recent WN9 smoke are separate evidence.
The compact [validation record](validation.json) distinguishes these scopes.
