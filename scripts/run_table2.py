#!/usr/bin/env python3
"""Plan, run, and summarize DSoM Table 2. Default mode never starts training.

GPU execution requires the user's explicit --mode smoke or --mode full command.
The full mode also requires committed source and a clean tracked working tree.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import shlex
import subprocess
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("MMKB-DB15K", "MKG-W", "MKG-Y")
METHODS = ("best_conf", "best_teacher", "best_strategy", "mulde")
LABELS = {
    "best_conf": "ConfTeacher",
    "best_teacher": "BestTeacher",
    "best_strategy": "BestStrategy",
    "mulde": "MulDE-adapted",
}
SOURCES = (
    "src/kd.py",
    "src/kd_learn.py",
    "src/kd_optimizers.py",
    "src/models.py",
    "src/datasets.py",
    "src/regularizers.py",
    "src/utils.py",
    "scripts/run_table2.py",
    "tests/test_table2.py",
    "docs/experiments/table2_paper_targets.json",
    "docs/experiments/TABLE2_REPRODUCTION.md",
)
MATCHED = (
    "rank",
    "batch_size",
    "optimizer",
    "learning_rate",
    "regularizer",
    "reg",
    "scheduler_mode",
    "kd_temp",
    "kd_ce_weight",
    "kd_kl_weight",
    "dkd_alpha",
    "dkd_beta",
    "init",
    "alpha",
    "scale",
    "decay1",
    "decay2",
    "max_epochs",
    "valid",
    "early_stopping",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def reference_path(references, section, key, root=ROOT):
    """Explicit repository-relative result paths; never guess a historical run."""
    entries = json.loads(Path(references).read_text()).get(section, {})
    if key not in entries:
        raise ValueError(f"Missing {section}/{key} in {references}")
    path = Path(entries[key])
    return path if path.is_absolute() else root / path


def baseline(dataset, root=ROOT, references=None):
    paths = (
        [reference_path(references, "baselines", dataset, root)]
        if references
        else list((root / "runs/student" / dataset / "full").rglob("result.json"))
    )
    if len(paths) != 1:
        raise ValueError(f"Expected one original Exp01 result for {dataset}: {paths}")
    result = json.loads(paths[0].read_text())
    assert result["dataset"] == dataset
    assert result["policy_optimizer"] == "source_frozen"
    assert result["max_train_examples"] == 0 and result["eval_queries"] == -1
    assert result["best_epoch"] > 0 and result["final_result"]
    for key in ("teacher_checkpoint", "img_info", "dscp_info"):
        if not Path(result[key]).is_file():
            raise FileNotFoundError(result[key])
    return paths[0], result


def build_command(dataset, method, original, job_dir, mode, commit):
    if method not in METHODS or dataset not in DATASETS:
        raise ValueError((dataset, method))
    cmd = [
        sys.executable,
        "-u",
        "kd_learn.py",
        "--dataset",
        dataset,
        "--model",
        "ComplExMDR",
        "--teacher_strategy",
        method,
        "--kd_loss_type",
        "mulde" if method == "mulde" else "selection_ndkd",
        "--policy_optimizer",
        "same",
    ]
    for key in ("teacher_checkpoint", "img_info", "dscp_info", *MATCHED):
        cmd.extend(["--" + key, str(original[key])])
    # Same hardcoded training seed (2022) as Exp01; no selection-policy updates.
    if mode == "smoke":
        for key, value in {"max_epochs": 1, "valid": 1, "early_stopping": 2}.items():
            cmd[cmd.index("--" + key) + 1] = str(value)
        cmd.extend(["--max_train_examples", "1000", "--eval_queries", "64"])
    else:
        cmd.extend(["--max_train_examples", "0", "--eval_queries", "-1"])
    if method == "mulde":
        cmd.extend(["--mulde_topk", "300", "--mulde_senior_lr", "0.001"])
    cmd.extend(
        [
            "--ckpt_dir",
            str(job_dir / "checkpoints"),
            "--note",
            f"T2_{dataset}_{method}_{mode}_Seed2022",
            "--git_commit",
            commit,
            "--experiment_id",
            "Table2_TeacherSelection",
        ]
    )
    return cmd


def verify_inputs(plan):
    for filename, expected in plan["input_sha256"].items():
        if digest(Path(filename)) != expected:
            raise ValueError(f"Input or original result changed: {filename}")


def completed_result(job, mode):
    folder = Path(job["directory"])
    paths = list((folder / "checkpoints").rglob("result.json"))
    students = list((folder / "checkpoints").rglob("student-*.pth"))
    if len(paths) != 1 or len(students) != 1:
        raise ValueError(f"Missing/ambiguous completed result or checkpoint: {folder}")
    result = json.loads(paths[0].read_text())
    assert result["dataset"] == job["dataset"]
    assert result["teacher_strategy"] == job["method"]
    assert result["kd_loss_type"] == (
        "mulde" if job["method"] == "mulde" else "selection_ndkd"
    )
    assert result["teacher_checkpoint"] == job["teacher_checkpoint"]
    assert result["best_epoch"] > 0
    assert result["max_train_examples"] == (1000 if mode == "smoke" else 0)
    assert result["eval_queries"] == (64 if mode == "smoke" else -1)
    assert result["policy_optimizer"] == "same"
    if job["method"] == "mulde":
        assert result["mulde_topk"] == 300 and result["mulde_senior_lr"] == 0.001
        log = (folder / "train.log").read_text(errors="replace")
        if "MULDE_SENIOR_PARAMS: PASS" not in log:
            raise ValueError("Missing Senior optimizer membership assertion")
        changes = [
            float(v)
            for v in re.findall(r"MULDE_SENIOR epoch=1 weight_change=(\S+)", log)
        ]
        if not changes or not math.isfinite(changes[0]) or changes[0] <= 0:
            raise ValueError("Senior weights did not update in the first epoch")
    options = {
        key[2:]: job["command"][i + 1]
        for i, key in enumerate(job["command"])
        if key.startswith("--")
    }
    for key in MATCHED:
        actual = result[key]
        expected = type(actual)(options[key])
        if actual != expected:
            raise ValueError(
                f"Configuration mismatch: {key}={actual}, expected {expected}"
            )
    scores = [result["final_result"]["MRR"], *result["final_result"]["hits@[1,3,10]"]]
    if len(scores) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in scores):
        raise ValueError("Invalid final_result ranking metrics")
    return paths[0], result


def summarize(directory):
    plan = json.loads((directory / "plan.json").read_text())
    verify_inputs(plan)
    targets = json.loads(
        (ROOT / "docs/experiments/table2_paper_targets.json").read_text()
    )
    rows, failures = [], []
    for dataset, info in plan["baselines"].items():
        original = json.loads(Path(info["result"]).read_text())
        rows.append(
            {
                "dataset": dataset,
                "method": "DSoM-source-frozen",
                "paper_method": "DSoM",
                "best_epoch": original["best_epoch"],
                "test": original["final_result"],
                "result": info["result"],
                "reused_baseline": True,
            }
        )
    for job in plan["jobs"]:
        try:
            path, result = completed_result(job, plan["training_mode"])
            rows.append(
                {
                    "dataset": job["dataset"],
                    "method": LABELS[job["method"]],
                    "paper_method": "MulDE"
                    if job["method"] == "mulde"
                    else LABELS[job["method"]],
                    "best_epoch": result["best_epoch"],
                    "test": result["final_result"],
                    "valid_mrr": result["best_mrr"],
                    "result": str(path),
                    "reused_baseline": False,
                }
            )
        except (ValueError, AssertionError, KeyError, OSError) as exc:
            failures.append(
                {"dataset": job["dataset"], "method": job["method"], "error": str(exc)}
            )
    for row in rows:
        row["paper_test"] = targets[row["dataset"]][row["paper_method"]]
        row["delta_mrr"] = (
            row["test"]["MRR"] - row["paper_test"]["MRR"]
            if plan["training_mode"] == "full"
            else None
        )
    summary = {
        "status": "PASS" if not failures else "INCOMPLETE",
        "training_mode": plan["training_mode"],
        "planned_jobs": len(plan["jobs"]),
        "rows": rows,
        "failures": failures,
        "limitations": plan["limitations"],
        "completed_new_cells": len(rows) - len(plan["baselines"]),
        "expected_full_table_new_cells": 12,
        "full_benchmark_grid_completed": (
            plan["training_mode"] == "full"
            and not failures
            and len(plan["jobs"]) == 12
            and set(plan["baselines"]) == set(DATASETS)
            and all(
                {j["method"] for j in plan["jobs"] if j["dataset"] == d} == set(METHODS)
                for d in DATASETS
            )
        ),
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = [
        "# Table 2 — local reproduction results",
        "",
        "Smoke results are execution checks only. MulDE-adapted has documented unpublished-setting assumptions.",
        f"Completed new cells: {summary['completed_new_cells']}/12; full benchmark grid completed: {summary['full_benchmark_grid_completed']}.",
        "",
        "| Dataset | Method | Epoch | Test MRR | H@1 | H@3 | H@10 | Paper MRR | Delta MRR |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        t, p = row["test"], row["paper_test"]
        h = t["hits@[1,3,10]"]
        delta = (
            "N/A (smoke)" if row["delta_mrr"] is None else f"{row['delta_mrr']:+.6f}"
        )
        text.append(
            f"| {row['dataset']} | {row['method']} | {row['best_epoch']} | {t['MRR']:.6f} | {h[0]:.6f} | {h[1]:.6f} | {h[2]:.6f} | {p['MRR']:.4f} | {delta} |"
        )
    if failures:
        text += [
            "",
            "Incomplete jobs:",
            "",
            *[f"- {f['dataset']}/{f['method']}: {f['error']}" for f in failures],
        ]
    (directory / "table2_results.md").write_text("\n".join(text) + "\n")
    print(
        f"{summary['status']}: {len(rows)} rows; {len(failures)} incomplete jobs; {directory}"
    )
    return not failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("plan", "smoke", "full", "summarize"), default="plan"
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS)
    )
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--references",
        type=Path,
        help="Explicit result registry for a fresh reproduction",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent training processes; default 1",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New run directory; existing directory for summarize",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B only for explicitly requested training",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.mode == "summarize":
        if args.output is None:
            parser.error("summarize requires --output RUN_DIRECTORY")
        raise SystemExit(0 if summarize(args.output.resolve()) else 1)
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if args.mode == "full":
        if dirty:
            parser.error("Commit tracked changes before full training")
        for source in SOURCES:
            git("ls-files", "--error-unmatch", source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    directory = (args.output or ROOT / "runs/table2" / args.mode / stamp).resolve()
    if not directory.is_relative_to(ROOT / "runs/table2"):
        parser.error(
            "--output must be inside runs/table2/ to isolate original artifacts"
        )
    if directory.exists():
        parser.error("Run directory exists; use a new directory or summarize mode")
    training_mode = "smoke" if args.mode == "smoke" else "full"
    plan = {
        "mode": args.mode,
        "training_mode": training_mode,
        "git_commit": commit,
        "tracked_tree_dirty": dirty,
        "training_seed": 2022,
        "environment": {
            "python": sys.version,
            "torch": importlib.metadata.version("torch"),
            "numpy": importlib.metadata.version("numpy"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "source_sha256": {s: digest(ROOT / s) for s in SOURCES},
        "input_sha256": {},
        "baselines": {},
        "jobs": [],
        "limitations": [
            "Single seed; reconstructed BERT/ViT features",
            "DSoM reference is source-frozen RC",
            "DB15K uses original contaminated benchmark validation, not CleanV1",
            "MulDE-adapted: unpublished DSoM settings unavailable; top-K=300, Senior Adam LR=.001, DSoM CE/N3/temperature KL",
        ],
    }
    for dataset in dict.fromkeys(args.datasets):
        path, original = (
            baseline(dataset, references=args.references)
            if args.references
            else baseline(dataset)
        )
        plan["baselines"][dataset] = {
            "result": str(path),
            "teacher_checkpoint": original["teacher_checkpoint"],
        }
        inputs = [
            path,
            *(
                Path(original[k])
                for k in ("teacher_checkpoint", "img_info", "dscp_info")
            ),
        ]
        inputs += [
            ROOT / "data" / dataset / n
            for n in (
                "train.pickle",
                "valid.pickle",
                "test.pickle",
                "to_skip.pickle",
                "ent_id",
                "rel_id",
            )
        ]
        for p in inputs:
            plan["input_sha256"][str(p)] = digest(p)
        for method in dict.fromkeys(args.methods):
            job_dir = directory / dataset / method
            command = build_command(
                dataset, method, original, job_dir, training_mode, commit
            )
            plan["jobs"].append(
                {
                    "dataset": dataset,
                    "method": method,
                    "directory": str(job_dir),
                    "teacher_checkpoint": original["teacher_checkpoint"],
                    "command": command,
                }
            )
    if args.references:
        plan["reference_registry"] = str(args.references.resolve())
        plan["reference_registry_sha256"] = digest(args.references)
        plan["reference_registry_snapshot"] = json.loads(args.references.read_text())
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (directory / "tracked_changes.patch").write_text(
        subprocess.check_output(["git", "diff", "HEAD"], cwd=ROOT, text=True)
    )
    with zipfile.ZipFile(
        directory / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for source in SOURCES:
            archive.write(ROOT / source, source)
    for job in plan["jobs"]:
        job_dir = Path(job["directory"])
        job_dir.mkdir(parents=True)
        (job_dir / "command.sh").write_text(
            "cd "
            + shlex.quote(str(ROOT / "src"))
            + "\n"
            + shlex.join(job["command"])
            + "\n"
        )
    print(f"Plan: {directory / 'plan.json'}", flush=True)
    print(
        f"{len(plan['jobs'])} jobs; mode={args.mode}; workers={args.workers}",
        flush=True,
    )
    if args.mode == "plan":
        print("PLAN ONLY: no training process started; CUDA not imported")
        return
    stop = threading.Event()

    def execute(job):
        if stop.is_set():
            return False
        job_dir = Path(job["directory"])
        env = dict(os.environ)
        env.update(
            DSOM_WANDB="1" if args.wandb else "0",
            WANDB_RUN_GROUP=f"Table2_{training_mode}_{stamp}",
            WANDB_NAME=f"T2_{job['dataset']}_{job['method']}_{training_mode}",
            WANDB_DIR=str(job_dir / "wandb"),
        )
        print(f"START {job['dataset']}/{job['method']}", flush=True)
        with (job_dir / "train.log").open("w") as log:
            code = subprocess.run(
                job["command"],
                cwd=ROOT / "src",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        try:
            if code:
                raise RuntimeError(
                    f"Training exit code {code}; see {job_dir / 'train.log'}"
                )
            completed_result(job, training_mode)
        except (RuntimeError, ValueError, AssertionError, KeyError, OSError) as exc:
            stop.set()
            print(f"FAILED {job['dataset']}/{job['method']}: {exc}", flush=True)
            return False
        print(f"PASS {job['dataset']}/{job['method']}", flush=True)
        return True

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(execute, plan["jobs"]))
    complete = summarize(directory)
    raise SystemExit(0 if all(outcomes) and complete else 1)


if __name__ == "__main__":
    main()
