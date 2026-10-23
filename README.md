# DSoM reproduction

Reproduction workflows for **Dark Side of Modalities: Reinforced Multimodal
Distillation for Multimodal Knowledge Graph Reasoning** (ACM MM 2025).
[Paper](https://arxiv.org/abs/2507.20738v1) ·
[Authors' release](https://github.com/OreOZhao/DSoM) ·
[Reproduction fork](https://github.com/keiranoluv/Reproduce_DSoM)

This fork provides audited data preparation, reconstructed BERT/ViT features,
separate teacher/student training, and paper-table orchestration. The supported
model is `ComplExMDR`: a multimodal teacher and structural student. Other legacy
classes/CLI combinations and competitors such as TMR/DySarl are not supported by
these reproduction workflows.

**Protocol matters:** main baseline rows use `source_frozen` RC, not learned RL.
Learned-RC and teacher-ensemble adaptations are labelled separately. Exact author
feature equivalence is unverified; DB15K validation contamination, inverse facts
in WordNet benchmarks, SUM/MEAN NDKD differences, and raw/filtered evaluation
settings are recorded in [protocol notes](docs/reproduction/PROTOCOL.md).

## Start here

1. [Create the environment](docs/reproduction/ENVIRONMENT.md).
2. [Download, prepare, and verify datasets/features](docs/reproduction/DATA.md).
3. [Run teacher/student and reproduce Tables 1/2/3/4/6/7](docs/reproduction/TABLES.md).
4. [Read results and diagnose failures](docs/reproduction/RESULTS.md).

Commands below assume the repository root and the activated environment. Processed
data, features, weights, historical runs, and checkpoints are **not included in a fresh clone**.
Some FB15K-237/WN18 raw split files are tracked; preprocessing reuses them only
when their ordered triples match the downloaded source, without rewriting them.
Download instructions include manual Google Drive/Baidu steps where needed.

## Environment

Use the revision of this fork containing this README, not only the upstream
release. If cloning a default branch without these files, checkout the branch
or commit where you found this guide before continuing. Record `git rev-parse HEAD`.

```bash
git clone https://github.com/keiranoluv/Reproduce_DSoM.git DSoM
cd DSoM
# Checkout the reproduction branch/revision containing this guide if necessary.

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Historical GPU profile: PyTorch 2.7.0 + CUDA 12.8.
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c requirements/constraints-tested.txt \
  -r requirements/train.txt -r requirements/data-features.txt \
  -r requirements/dev.txt

python scripts/check_environment.py --features --dev
python -m pip check
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests
```

The venv may have any name; Python launchers use `sys.executable`. Optional W&B:

```bash
python -m pip install -c requirements/constraints-tested.txt -r requirements/tracking.txt
# Login only if you want online tracking; alternatively set WANDB_MODE=offline.
wandb login
```

The recorded machine used Python 3.12.3 and an RTX 5090 with 32 GiB VRAM. This is
an observed setup, not a measured minimum requirement. CPU tests do not establish
GPU compatibility. On your GPU host, explicitly check it with:

```bash
python scripts/check_environment.py --features --gpu
```

That last command initializes CUDA; ordinary environment checks and plans do not.
See [CPU-only installation and dependency details](docs/reproduction/ENVIRONMENT.md).

## First complete run

**Prepare one dataset first**, following [DATA.md](docs/reproduction/DATA.md).
For example, after `data/MKG-Y/` contains verified triples, maps, filters and the
`(15000,768)` text / `(15000,1000)` image features:

```bash
# CPU/file checks; no Torch import or training in plan mode.
CUDA_VISIBLE_DEVICES='' python scripts/run_reproduction.py \
  --dataset MKG-Y --mode plan

# GPU: fresh teacher -> saved teacher -> fresh student.
python scripts/run_reproduction.py --dataset MKG-Y --mode smoke
```

After smoke passes, commit any code/config edits before full training. Then:

```bash
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_reproduction.py --dataset MKG-Y --mode full --wandb \
  --registry references.local.json
```

Full starts fresh; it does not resume smoke. Each invocation creates a new
`runs/reproduction/MKG-Y/baseline/<mode>/<timestamp>/` containing `teacher/`
and `student/`. A completed full student can be registered in the machine-local
`references.local.json`. No historical checkpoint filename is required.
Teacher selection follows validation Hits@10; student selection follows
validation MRR. Teacher oracle metrics are not the final prediction result.

## Which tables can I run?

| Table | Workflow | Prerequisites / limits |
|---|---|---|
| 1 | Fresh baselines on DB15K, MKG-W, MKG-Y | DSoM row only; source-frozen RC |
| 2 | `run_table2.py --references references.local.json` | Three full baselines; MulDE adaptation |
| 3 | Fresh baselines on FB15K-237, WN18 | DSoM row only; source-frozen RC |
| 4 | Four reference profiles -> two component runs -> teacher ensembles | DB15K teacher; adapted ML/RC-on-val |
| 6 | `run_table6.py --references references.local.json` | Three full baselines; default fresh registry trains 9 single-teacher cells |
| 7 | `run_table7.py` | Prepared WN9; new teacher/student; raw and filtered evaluation |

[Full commands, separate stages, and dependency order](docs/reproduction/TABLES.md).
The historical launchers remain available; their default references can require
old local runs. Use the portable commands and explicit registry in this guide
on a fresh clone. `--references` avoids guessing the newest matching checkpoint.

WN9 has its own E2E workflow once WN18 features and MoSE WN9 inputs are prepared:

```bash
python scripts/run_table7.py --mode plan
python scripts/run_table7.py --mode smoke
# After smoke passes:
WANDB_PROJECT=DSoM-Reproduction python scripts/run_table7.py --mode full --wandb
```

These are GPU commands except plan. The assistant-prepared CPU checks do not
replace running smoke on your hardware. Teacher/student are also independently
runnable; see [Table 7 details](docs/experiments/TABLE7_WN9_REPRODUCTION.md).

## Evidence and limitations

Historical baseline and ablation runs have been completed locally for Tables
1/2/3/4/6, with differences from paper. The user also completed the **Table 7
DSoM row** with a fresh WN9 teacher/student; artifact verification passed on
2026-10-24. The selected student achieved filtered MRR **0.919634**, H@1
**0.903336**, H@10 **0.949583**, versus paper .9213/.9052/.9556 (MRR -0.167
percentage points). Raw MRR on the same checkpoint is **0.618763**. The paper's
exact raw/filtered protocol remains unresolved; this is a completed source-frozen
run, not confirmation of exact author settings or learned RC. Competitor rows
were not retrained. See the [Table 7 result record](docs/experiments/TABLE7_WN9_REPRODUCTION.md#recorded-full-run--2026-10-24).
The additive portable
setup passed 160 tests in a newly created isolated CPU venv, plus relocated
checkout checks without historical runs or a repo-local environment. A new
clean-machine GPU reproduction has not been performed for this guide.

- [Environment snapshot](requirements/environment-observed.json)
- [CPU/setup validation evidence](docs/reproduction/validation.json)
- [Table 3 results](docs/experiments/TABLE3_FB15K237_WN18_SOURCE_FROZEN_RC.md)
- [Table 6 recorded results](docs/experiments/TABLE6_REPRODUCTION.md)
- [Table 7 recorded results and artifact hashes](docs/experiments/table7_results_20261024.json)
- [DB15K source/sensitivity audit](docs/experiments/DB15K_CLEAN_SPLIT_SENSITIVITY.md)
- [WN9 provenance and feature reuse](docs/experiments/WN9_SOURCE_AUDIT.md)

Historical `docs/experiments/` files preserve run-specific commands and results;
use `docs/reproduction/` as the current installation/download/run guide. Do not
report source-frozen results as learned-RC gains, or smoke scores as benchmark
results. Figures 3/4/5 and retraining all competitor baselines are outside the
completed table workflows.

## Citation
Please cite the following paper as reference if you find our work useful.

```bibtex
@inproceedings{zhao2025DSoM,
author = {Zhao, Yu and Zhang, Ying and Sui, Xuhui and Zhou, Baohang and Zhu, Haoze and Pan, Jeff Z. and Yuan, Xiaojie},
title = {Dark Side of Modalities: Reinforced Multimodal Distillation for Multimodal Knowledge Graph Reasoning},
year = {2025},
isbn = {9798400720352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746027.3754576},
doi = {10.1145/3746027.3754576},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {2506–2515},
numpages = {10},
keywords = {knowledge distillation, multimodal knowledge graphs, multimodal knowledge reasoning, multimodal learning},
location = {Dublin, Ireland},
series = {MM '25}
}
```
