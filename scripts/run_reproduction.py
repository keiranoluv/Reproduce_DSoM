#!/usr/bin/env python3
"""Portable baseline/Table 4 stage launcher using existing training entrypoints.

No historical runs needed. Default plan; explicit smoke/full launches GPU processes.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from prepare_dataset import EXPECTED
from run_table2 import digest, git, verify_inputs
from wn9_workflow import build_command, training_result

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "baseline": ("rc", "source_frozen", "ndkd"),
    "table4-average": ("average", "same", "ndkd"),
    "table4-rc-ndkd": ("rc", "Adam", "ndkd"),
    "table4-rc-kd": ("rc", "Adam", "kd"),
    "table4-rc-dkd": ("rc", "Adam", "dkd"),
}
LABELS = {
    "table4-average": "A4_average_NDKD",
    "table4-rc-ndkd": "A5_RC_NDKD",
    "table4-rc-kd": "A9_RC_KD",
    "table4-rc-dkd": "A10_RC_DKD",
}
SOURCES = (
    "scripts/run_reproduction.py",
    "scripts/run_table2.py",
    "scripts/wn9_workflow.py",
    "scripts/prepare_dataset.py",
    "src/learn.py",
    "src/kd_learn.py",
    "src/models.py",
    "src/kd.py",
    "src/kd_optimizers.py",
    "src/datasets.py",
    "src/optimizers.py",
    "src/regularizers.py",
    "src/utils.py",
)


def data_inputs(dataset):
    import pickle

    import numpy as np

    data = ROOT / "data" / dataset
    n = EXPECTED[dataset][0]
    entries = [line.split("\t") for line in (data / "ent_id").read_text().splitlines()]
    if len(entries) != n or sorted(int(v) for _, v in entries) != list(range(n)):
        raise ValueError("Unexpected entity mapping")
    relations = EXPECTED[dataset][1]
    rel_entries = [
        line.split("\t") for line in (data / "rel_id").read_text().splitlines()
    ]
    if len(rel_entries) != relations or sorted(int(v) for _, v in rel_entries) != list(
        range(relations)
    ):
        raise ValueError("Unexpected relation mapping")
    for split, count in zip(("train", "valid", "test"), EXPECTED[dataset][2]):
        with (data / f"{split}.pickle").open("rb") as stream:
            triples = pickle.load(stream)
        if (
            triples.shape != (count, 3)
            or triples.dtype.kind not in "iu"
            or triples.min() < 0
            or triples[:, [0, 2]].max() >= n
            or triples[:, 1].max() >= relations
        ):
            raise ValueError(f"Invalid processed split: {split}")
        if split == "train" and (
            int(triples[:, [0, 2]].max()) + 1,
            int(triples[:, 1].max()) + 1,
        ) != (n, relations):
            raise ValueError(
                "Train-max dataset shape inference differs from complete mappings"
            )
    for name, dim in (("img_vec.pickle", 1000), ("dscp_vec.pickle", 768)):
        with (data / name).open("rb") as f:
            values = pickle.load(f)
        if (
            values.shape != (n, dim)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Invalid feature array: {name}")
    names = (
        "ent_id",
        "rel_id",
        "train.pickle",
        "valid.pickle",
        "test.pickle",
        "to_skip.pickle",
        "probas.pickle",
        "img_vec.pickle",
        "dscp_vec.pickle",
    )
    return {str(data / name): digest(data / name) for name in names}


def command(stage, dataset, profile, mode, directory, checkpoint=None):
    strategy, policy, loss = PROFILES[profile]
    cmd, cfg = build_command(
        stage,
        mode,
        directory,
        git("rev-parse", "HEAD"),
        policy,
        teacher_checkpoint=checkpoint,
    )
    cfg.update(
        dataset=dataset,
        img_info=str(ROOT / "data" / dataset / "img_vec.pickle"),
        dscp_info=str(ROOT / "data" / dataset / "dscp_vec.pickle"),
        note=f"Reproduction_{dataset}_{stage}_{profile}_{mode}_Seed2022",
        experiment_id="Portable_Reproduction",
    )
    if stage == "student":
        cfg.update(
            teacher_checkpoint=checkpoint or f"<PENDING_SELECTED_{dataset}_TEACHER>",
            teacher_strategy=strategy,
            kd_loss_type="fixed_ndkd" if strategy == "average" else "kdrl",
            rc_distill_loss=loss,
        )
    cmd = [sys.executable, "-u", "learn.py" if stage == "teacher" else "kd_learn.py"]
    for key, value in cfg.items():
        if value is None:
            continue
        if key in ("policy_input_norm", "l5_counterfactual"):
            if value:
                cmd.append("--" + key)
        else:
            cmd += ["--" + key, str(value)]
    return cmd, cfg


def teacher_checkpoint(directory, mode, dataset, hashes):
    plan = json.loads((directory / "plan.json").read_text())
    if plan["stage"] != "teacher" or (mode != "smoke" and plan["mode"] != "full"):
        raise ValueError("Use a completed full teacher for full/plan student settings")
    if plan["config"]["dataset"] != dataset:
        raise ValueError("Teacher dataset mismatch")
    verify_inputs(plan)
    if plan["input_sha256"] != hashes:
        raise ValueError("Teacher dataset/feature hashes differ")
    path, checkpoint, _ = training_result(directory, "teacher", plan["config"])
    completed = json.loads((directory / "stage_result.json").read_text())
    if (
        completed["status"] != "PASS"
        or completed["checkpoint_sha256"] != digest(checkpoint)
        or completed["result_sha256"] != digest(path)
    ):
        raise ValueError("Teacher result/checkpoint changed")
    return checkpoint


def stage(stage_name, args, directory, hashes, teacher=None):
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    checkpoint = (
        teacher_checkpoint(teacher, args.mode, args.dataset, hashes)
        if teacher
        else None
    )
    cmd, cfg = command(
        stage_name,
        args.dataset,
        args.profile,
        args.mode,
        directory,
        str(checkpoint) if checkpoint else None,
    )
    protected = dict(hashes)
    if checkpoint:
        protected[str(checkpoint)] = digest(checkpoint)
    plan = dict(
        stage=stage_name,
        mode=args.mode,
        profile=args.profile,
        config=cfg,
        command=cmd,
        input_sha256=protected,
        source_sha256={name: digest(ROOT / name) for name in SOURCES},
    )
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (directory / "command.sh").write_text(
        "cd " + shlex.quote(str(ROOT / "src")) + "\n" + shlex.join(cmd) + "\n"
    )
    with zipfile.ZipFile(
        directory / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in SOURCES:
            archive.write(ROOT / name, name)
    print(f"{stage_name}: {directory / 'plan.json'}", flush=True)
    if args.mode == "plan":
        return directory
    (directory / "checkpoints").mkdir()
    (directory / "wandb").mkdir()
    env = dict(
        os.environ,
        DSOM_WANDB="1" if args.wandb else "0",
        DSOM_RUN_DIR=str(directory),
        WANDB_DIR=str(directory / "wandb"),
        WANDB_RUN_GROUP=directory.parent.name,
        WANDB_NAME=cfg["note"],
    )
    try:
        with (directory / "train.log").open("w") as log:
            code = subprocess.run(
                cmd, cwd=ROOT / "src", env=env, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        if code:
            raise RuntimeError(f"Exit {code}; see {directory / 'train.log'}")
        result_path, checkpoint, result = training_result(directory, stage_name, cfg)
        verify_inputs(plan)
        if (
            args.profile.startswith("table4-rc")
            and stage_name == "student"
            and args.mode == "full"
        ):
            import re

            updates = re.findall(
                r"Batches with updated policy weights:\s*(\d+)",
                (directory / "train.log").read_text(errors="replace"),
            )
            if not any(int(v) for v in updates):
                raise ValueError("Learned policy did not update")
        out = dict(
            status="PASS",
            mode=args.mode,
            stage=stage_name,
            dataset=args.dataset,
            checkpoint=str(checkpoint),
            checkpoint_sha256=digest(checkpoint),
            result=str(result_path),
            result_sha256=digest(result_path),
            final_result=result["final_result"],
        )
        (directory / "stage_result.json").write_text(json.dumps(out, indent=2) + "\n")
    except Exception as error:
        (directory / "failure.json").write_text(
            json.dumps({"error": repr(error)}, indent=2) + "\n"
        )
        raise
    print("PASS:", directory / "stage_result.json", flush=True)
    return directory


def register(registry, stage_dir, key):
    if registry.exists():
        record = json.loads(registry.read_text())
    else:
        record = {"schema_version": 1, "baselines": {}, "table4": {}, "table6": {}}
    plan = json.loads((stage_dir / "plan.json").read_text())
    if plan["mode"] != "full" or plan["stage"] != "student":
        raise ValueError("Only completed full students can be registered")
    path, _, _ = training_result(stage_dir, "student", plan["config"])
    if json.loads((stage_dir / "stage_result.json").read_text())["status"] != "PASS":
        raise ValueError("Student stage did not pass")
    section = "baselines" if key == "baseline" else "table4"
    label = plan["config"]["dataset"] if section == "baselines" else LABELS[key]
    if label in record.setdefault(section, {}):
        raise ValueError(
            f"Registry already has {section}/{label}; use a new registry or explicitly review it"
        )
    record[section][label] = str(path.relative_to(ROOT))
    registry.parent.mkdir(parents=True, exist_ok=True)
    tmp = registry.with_suffix(registry.suffix + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(registry)
    print("Registered:", section, label, "in", registry)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=EXPECTED, required=True)
    p.add_argument("--stage", choices=("teacher", "student", "e2e"), default="e2e")
    p.add_argument("--profile", choices=PROFILES, default="baseline")
    p.add_argument("--mode", choices=("plan", "smoke", "full"), default="plan")
    p.add_argument("--teacher-run", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--registry",
        type=Path,
        help="Register completed full student result for Table 2/4/6",
    )
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()
    if args.profile != "baseline" and args.dataset != "MMKB-DB15K":
        p.error("Table 4 profiles are DB15K only")
    if args.stage == "student" and args.teacher_run is None:
        p.error("--teacher-run is required for the separate student stage")
    if args.stage == "teacher" and args.profile != "baseline":
        p.error("Use baseline profile for the teacher stage")
    if args.mode == "full" and args.profile != "baseline" and args.stage != "student":
        p.error(
            "Full Table 4 profiles require --stage student --teacher-run with the same shared DB15K teacher"
        )
    if args.registry and args.stage == "teacher":
        p.error("A registry records student results, not teacher-only stages")
    if args.registry and args.registry.exists() and args.mode == "full":
        record = json.loads(args.registry.read_text())
        section = "baselines" if args.profile == "baseline" else "table4"
        label = args.dataset if section == "baselines" else LABELS[args.profile]
        if label in record.get(section, {}):
            p.error(
                f"Registry already contains {section}/{label}; select a new registry before retraining"
            )
    if args.mode == "full":
        if git("status", "--porcelain", "--untracked-files=no"):
            p.error("Commit tracked changes before full training")
        for name in SOURCES:
            git("ls-files", "--error-unmatch", name)
    hashes = data_inputs(args.dataset)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    base = ROOT / "runs/reproduction"
    directory = (
        args.output or base / args.dataset / args.profile / args.mode / stamp
    ).resolve()
    if not directory.is_relative_to(base) or directory.exists():
        p.error("Use a NEW directory within runs/reproduction/")
    if args.mode == "plan" and args.stage == "e2e":
        directory.mkdir(parents=True)
        jobs = {
            kind: command(kind, args.dataset, args.profile, args.mode, directory / kind)
            for kind in ("teacher", "student")
        }
        (directory / "plan.json").write_text(
            json.dumps({"jobs": jobs, "input_sha256": hashes}, indent=2) + "\n"
        )
        print("PLAN ONLY:", directory / "plan.json")
        return
    teacher = args.teacher_run.resolve() if args.teacher_run else None
    if args.stage in ("teacher", "e2e"):
        teacher = stage("teacher", args, directory / "teacher", hashes)
    if args.stage in ("student", "e2e"):
        student = stage("student", args, directory / "student", hashes, teacher)
        if args.registry and args.mode == "full":
            register(args.registry, student, args.profile)


if __name__ == "__main__":
    main()
