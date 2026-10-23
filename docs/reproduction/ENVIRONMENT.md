# Environment setup

Run commands from the reproduction repository root. Use Linux x86_64 and Python
3.12 for the documented profile. The observed environment used Python 3.12.3,
torch 2.7.0+cu128, torchvision 0.22.0+cu128 and an RTX 5090 32 GiB. Neither minimum
VRAM nor other GPU families were certified. Training/ViT extraction use CUDA;
CPU supports preparation, integrity checks, plans, tests, and optional BERT extraction.

Install Git, Python 3.12 with its `venv` module, and `unzip` through your OS package
manager first. For example, Ubuntu 24.04 provides `git`, `python3.12`,
`python3.12-venv`, and `unzip`; system installation may require administrator access.

## Create and activate

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

No Conda environment named `dsom` is required. The supported Python launchers
invoke their current `sys.executable`, including when the repository/environment
is elsewhere. Run them with `python scripts/NAME.py` after activation. Older
historical Bash scripts can still expect `./dsom/bin/python`; they are not the
fresh-checkout path documented here.

## Install the torch build first

For the historical CUDA 12.8 profile:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

This follows the [official PyTorch 2.7.0 instructions](https://pytorch.org/get-started/previous-versions/#v270).
An appropriate NVIDIA driver is required; a Python package install cannot fix
an incompatible host driver. The CUDA wheel build is distinct from the driver
reported by `nvidia-smi`. Torchaudio is not used by these workflows.

For **CPU checks only**, install the CPU wheels instead:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

That does not make the legacy training entrypoints CPU-compatible. Use a separate
CPU environment if you do not want to replace a working CUDA installation.

## Install project dependencies

```bash
python -m pip install -c requirements/constraints-tested.txt \
  -r requirements/train.txt -r requirements/data-features.txt \
  -r requirements/dev.txt
```

Groups:

| File | Use |
|---|---|
| `train.txt` | NumPy and progress reporting; torch installed separately |
| `data-features.txt` | BERT, ViT, Pillow, downloads, HDF5 legacy audits |
| `dev.txt` | pytest and Ruff |
| `tracking.txt` | Optional W&B |
| `constraints-tested.txt` | Observed direct/key dependency pins; not a complete transitive lock |
| `environment-observed.json` | Machine/environment evidence, not an install script |
| `lock-cpu-py312.txt` | Full resolved isolated CPU environment, including optional groups; CPU only |

The NumPy pin requires Python >=3.12. Do not replace this profile with the old
MMRNS/Python/PyTorch requirements from another upstream repository. Source
repositories are used for their data, not installed as competing training stacks.
Direct pins were checked against public package metadata. A new isolated Python
3.12 CPU venv installed torch/torchvision from the official CPU index and all
four requirement groups successfully: `pip check` and all 160 tests passed.
The resolved CPU packages are captured in `lock-cpu-py312.txt`. To recreate that
CPU environment after installing CPU torch/torchvision above:

```bash
python -m pip install -r requirements/lock-cpu-py312.txt
```

Do not apply that CPU lock to a CUDA environment: it pins `+cpu` wheels.
Clean CUDA installation/GPU training in an empty environment have not been
performed for this documentation change. See [validation evidence](validation.json).

## Verify before GPU work

```bash
python scripts/check_environment.py --features --dev
python -m pip check
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests
```

The environment checker reads installed distribution metadata without importing
Torch. It fails on missing or mismatched profile versions. CPU tests import
Torch, prohibit CUDA initialization in GPU-sensitive tests, and use synthetic
small fixtures where optimizer steps are needed.

Only when you choose to initialize your GPU:

```bash
python scripts/check_environment.py --features --gpu
```

This explicitly creates a small CUDA tensor. Training/extraction smoke should
follow on your own hardware. Package-version equality alone does not establish
driver support, VRAM sufficiency or numeric equivalence on another GPU.

## Tracking and model downloads

```bash
python -m pip install -c requirements/constraints-tested.txt -r requirements/tracking.txt
python scripts/check_environment.py --wandb
wandb login
```

Add `--wandb` to a launcher to enable tracking. Without it, the supported launchers
set `DSOM_WANDB=0` even if the shell had enabled it previously. For offline tracking:

```bash
WANDB_MODE=offline WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_reproduction.py --dataset MKG-Y --mode full --wandb
```

W&B is not required for local result files. The teacher-ensemble runner uses the
W&B API directly when requested; the other launchers enable existing training hooks.

BERT and ViT pretrained weights are downloaded by their libraries on first
extraction. Keep the Hugging Face and torch hub caches if rerunning; offline
execution requires prepopulated caches. These weights are separate from graph
teacher/student checkpoints. Setting a visible GPU, for example
`CUDA_VISIBLE_DEVICES=0`, is the caller's choice; the launchers do not select a
physical GPU or queue work in the background.
