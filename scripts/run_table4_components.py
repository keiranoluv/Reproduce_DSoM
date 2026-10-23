#!/usr/bin/env python3
"""Plan or run the two DB15K learned-RC component ablations matched to A5.

Default: plan only, no Torch/CUDA import or training. The user runs smoke/full.
"""

import argparse
import importlib.metadata
import json
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

from run_table2 import MATCHED, digest, git, reference_path, verify_inputs

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = {"nekd": (1.0, 0.0), "nnkd": (0.0, 1.0)}
EXTRA = (
    "dataset",
    "model",
    "teacher_checkpoint",
    "img_info",
    "dscp_info",
    "kd_loss_type",
    "teacher_strategy",
    "policy_optimizer",
    "policy_lr",
    "rl_hs",
    "rl_reward",
    "rl_penalize",
    "rl_weight",
    "policy_temp",
    "policy_input_norm",
    "rc_reward_mode",
    "l5_counterfactual",
    "modal_missing",
    "modality_split",
    "fusion_img",
    "fusion_dscp",
)
SOURCES = (
    "src/kd.py",
    "src/kd_learn.py",
    "src/kd_optimizers.py",
    "src/datasets.py",
    "src/models.py",
    "src/regularizers.py",
    "scripts/run_table2.py",
    "scripts/run_table4_components.py",
    "tests/test_table4_components.py",
    "docs/experiments/TABLE4_COMPONENT_ABLATIONS.md",
)
REFERENCES = {
    "A4_average_NDKD": (
        "fixed_teacher_ndkd_full/*/average/checkpoints/*/result.json",
        [0.4191, 0.3369, 0.4583, 0.5756],
    ),
    "A5_RC_NDKD": (
        "learned_rc_adam_full/*/checkpoints/*/result.json",
        [0.4268, 0.3445, 0.4662, 0.5812],
    ),
    "A9_RC_KD": (
        "table4_learned_rc_a9_a10/full/*/A9_KL/checkpoints/*/result.json",
        [0.4203, 0.3362, 0.4607, 0.5796],
    ),
    "A10_RC_DKD": (
        "table4_learned_rc_a9_a10/full/*/A10_DKD/checkpoints/*/result.json",
        [0.4231, 0.3400, 0.4616, 0.5807],
    ),
}
PAPER = {
    "nekd": [0.4100, 0.3307, 0.4493, 0.5600],
    "nnkd": [0.4233, 0.3393, 0.4639, 0.5808],
}


def load_references(registry=None):
    references = {}
    for label, (pattern, target) in REFERENCES.items():
        paths = (
            [reference_path(registry, "table4", label, ROOT)]
            if registry
            else list((ROOT / "runs/student/MMKB-DB15K").glob(pattern))
        )
        if len(paths) != 1:
            raise ValueError(f"Expected one {label} reference; found {paths}")
        result = json.loads(paths[0].read_text())
        references[label] = {"path": str(paths[0]), "paper": target, "result": result}
    validate_a5(references["A5_RC_NDKD"]["result"])
    teacher = references["A5_RC_NDKD"]["result"]["teacher_checkpoint"]
    for label, reference in references.items():
        r = reference["result"]
        if r["teacher_checkpoint"] != teacher or r["dataset"] != "MMKB-DB15K":
            raise ValueError(f"Mismatched reference teacher/dataset: {label}")
        if r["max_train_examples"] != 0 or r["eval_queries"] != -1:
            raise ValueError(f"Reference is not a full benchmark run: {label}")
    return references


def validate_a5(result):
    required = {
        "dataset": "MMKB-DB15K",
        "kd_loss_type": "kdrl",
        "teacher_strategy": "rc",
        "policy_optimizer": "Adam",
        "policy_lr": 0.00001,
        "rc_reward_mode": "binary",
        "dkd_alpha": 1.0,
        "dkd_beta": 1.0,
        "max_train_examples": 0,
        "eval_queries": -1,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise ValueError(f"A5 {key}={result.get(key)!r}; expected {expected!r}")
    if (result.get("rc_distill_loss") or "ndkd") != "ndkd":
        raise ValueError("A5 must use NDKD")


def build_command(baseline, component, mode, directory, commit):
    alpha, beta = COMPONENTS[component]
    cfg = {key: baseline[key] for key in (*EXTRA, *MATCHED)}
    cfg.update(
        dkd_alpha=alpha,
        dkd_beta=beta,
        rc_distill_loss="ndkd",
        max_train_examples=0,
        eval_queries=-1,
        ckpt_dir=str(directory / "checkpoints"),
        note=f"T4_DB15K_RC_{component.upper()}_{mode}_Seed2022",
        git_commit=commit,
        experiment_id="Table4_DB15K_RC_ComponentAblation",
    )
    if mode == "smoke":
        cfg.update(
            max_epochs=1,
            valid=1,
            early_stopping=2,
            max_train_examples=3000,
            eval_queries=64,
        )
    command = [sys.executable, "-u", "kd_learn.py"]
    for key, value in cfg.items():
        if key in ("policy_input_norm", "l5_counterfactual"):
            if value:
                command.append("--" + key)
            continue
        command.extend(["--" + key, str(value)])
    return command, cfg


def completed_result(job):
    folder = Path(job["directory"])
    paths = list((folder / "checkpoints").rglob("result.json"))
    models = list((folder / "checkpoints").rglob("student-*.pth"))
    if len(paths) != 1 or len(models) != 1:
        raise ValueError(f"Missing or ambiguous result/checkpoint: {folder}")
    result = json.loads(paths[0].read_text())
    for key, expected in job["config"].items():
        if result.get(key) != expected:
            raise ValueError(f"{key}: got {result.get(key)!r}, expected {expected!r}")
    if result["best_epoch"] <= 0:
        raise ValueError("No best-validation checkpoint")
    idx = int(result["best_epoch"] / result["valid"]) - 1
    if result["final_result"] != result["curve"]["test"][idx]:
        raise ValueError(
            "final_result does not match the selected checkpoint's evaluation"
        )
    log = (folder / "train.log").read_text(errors="replace")
    updates = [
        int(v) for v in re.findall(r"Batches with updated policy weights:\s*(\d+)", log)
    ]
    if not updates or max(updates) == 0:
        raise ValueError("No recorded learned-RC policy weight updates")
    return paths[0], result


def summarize(directory):
    plan = json.loads((directory / "plan.json").read_text())
    verify_inputs(plan)
    rows, failures = [], []
    for label, reference in plan["references"].items():
        result = json.loads(Path(reference["path"]).read_text())
        rows.append(
            {
                "method": label,
                "best_epoch": result["best_epoch"],
                "valid_mrr": result["best_mrr"],
                "test": result["final_result"],
                "paper": reference["paper"],
                "result": reference["path"],
                "reused": True,
            }
        )
    for job in plan["jobs"]:
        try:
            path, result = completed_result(job)
            rows.append(
                {
                    "method": "RC_" + job["component"].upper(),
                    "best_epoch": result["best_epoch"],
                    "valid_mrr": result["best_mrr"],
                    "test": result["final_result"],
                    "paper": PAPER[job["component"]],
                    "result": str(path),
                    "reused": False,
                }
            )
        except (ValueError, KeyError, OSError, IndexError) as error:
            failures.append({"component": job["component"], "error": str(error)})
    for row in rows:
        row["delta_mrr_pp"] = (
            (row["test"]["MRR"] - row["paper"][0]) * 100
            if plan["training_mode"] == "full"
            else None
        )
    summary = {
        "status": "PASS" if not failures else "INCOMPLETE",
        "rows": rows,
        "failures": failures,
        "training_mode": plan["training_mode"],
        "component_pair_completed": plan["training_mode"] == "full"
        and not failures
        and len(plan["jobs"]) == 2,
        "whole_table4_completed": False,
        "limitations": plan["limitations"],
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = [
        "# Table 4 — Student component ablations",
        "",
        f"Mode: {plan['training_mode']}; smoke scores are execution checks only.",
        "Original DB15K split; learned RC; released-code neighbor SUM (paper describes MEAN).",
        "Teacher Avg., ML-on-validation, and RC-on-validation remain separate pending rows.",
        "",
        "| Method | Best epoch | Valid MRR | Test MRR | H@1 | H@3 | H@10 | Paper MRR | Delta MRR (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        t = row["test"]
        h = t["hits@[1,3,10]"]
        delta = (
            "N/A (smoke)"
            if row["delta_mrr_pp"] is None
            else f"{row['delta_mrr_pp']:+.4f}"
        )
        text.append(
            f"| {row['method']} | {row['best_epoch']} | {row['valid_mrr']:.6f} | {t['MRR']:.6f} | {h[0]:.6f} | {h[1]:.6f} | {h[2]:.6f} | {row['paper'][0]:.4f} | {delta} |"
        )
    if failures:
        text += [
            "",
            "Incomplete jobs:",
            *[f"- {f['component']}: {f['error']}" for f in failures],
        ]
    (directory / "table4_student_results.md").write_text("\n".join(text) + "\n")
    print(f"{summary['status']}: {directory / 'summary.json'}")
    return not failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("plan", "smoke", "full", "summarize"), default="plan"
    )
    parser.add_argument(
        "--components", nargs="+", choices=tuple(COMPONENTS), default=list(COMPONENTS)
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--references", type=Path)
    args = parser.parse_args()
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
    directory = (
        args.output or ROOT / "runs/table4_components" / args.mode / stamp
    ).resolve()
    if (
        not directory.is_relative_to(ROOT / "runs/table4_components")
        or directory.exists()
    ):
        parser.error("Use a NEW output directory inside runs/table4_components/")
    refs = load_references(args.references) if args.references else load_references()
    baseline = refs["A5_RC_NDKD"]["result"]
    mode = "smoke" if args.mode == "smoke" else "full"
    plan = {
        "mode": args.mode,
        "training_mode": mode,
        "git_commit": commit,
        "tracked_tree_dirty": dirty,
        "seed": 2022,
        "wandb": args.wandb,
        "environment": {
            "python": sys.version,
            "torch": importlib.metadata.version("torch"),
            "numpy": importlib.metadata.version("numpy"),
        },
        "source_sha256": {source: digest(ROOT / source) for source in SOURCES},
        "references": {
            label: {"path": ref["path"], "paper": ref["paper"]}
            for label, ref in refs.items()
        },
        "input_sha256": {},
        "jobs": [],
        "limitations": [
            "Original DB15K validation contamination; not CleanV1",
            "Single training seed 2022 and reconstructed features",
            "A5-matched separate-Adam learned RC, not source-frozen reference",
            "Released NEKD sums neighbor probabilities; paper Eq.12 describes their mean",
            "Two component runs do not complete the three teacher-ensemble rows of Table 4",
        ],
    }
    inputs = [Path(ref["path"]) for ref in refs.values()]
    inputs += [
        Path(baseline[key]) for key in ("teacher_checkpoint", "img_info", "dscp_info")
    ]
    inputs += [
        ROOT / "data/MMKB-DB15K" / name
        for name in (
            "train.pickle",
            "valid.pickle",
            "test.pickle",
            "to_skip.pickle",
            "ent_id",
            "rel_id",
        )
    ]
    plan["input_sha256"] = {str(path): digest(path) for path in inputs}
    if args.references:
        plan["reference_registry"] = str(args.references.resolve())
        plan["reference_registry_sha256"] = digest(args.references)
        plan["reference_registry_snapshot"] = json.loads(args.references.read_text())
    for component in dict.fromkeys(args.components):
        job_dir = directory / ("RC_" + component.upper())
        command, cfg = build_command(baseline, component, mode, job_dir, commit)
        plan["jobs"].append(
            {
                "component": component,
                "directory": str(job_dir),
                "command": command,
                "config": cfg,
            }
        )
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
        folder = Path(job["directory"])
        folder.mkdir()
        (folder / "command.sh").write_text(
            "cd "
            + shlex.quote(str(ROOT / "src"))
            + "\n"
            + shlex.join(job["command"])
            + "\n"
        )
    print(f"Plan: {directory / 'plan.json'}", flush=True)
    if args.mode == "plan":
        print("PLAN ONLY: no training started; Torch/CUDA not imported")
        return
    stop = threading.Event()

    def execute(job):
        if stop.is_set():
            return False
        folder = Path(job["directory"])
        env = dict(
            os.environ,
            DSOM_WANDB="1" if args.wandb else "0",
            WANDB_RUN_GROUP=f"Table4_Components_{mode}_{stamp}",
            WANDB_NAME=f"T4_DB15K_RC_{job['component'].upper()}_{mode}",
            WANDB_DIR=str(folder / "wandb"),
        )
        print(f"START RC_{job['component'].upper()}", flush=True)
        with (folder / "train.log").open("w") as log:
            code = subprocess.run(
                job["command"],
                cwd=ROOT / "src",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            ).returncode
        try:
            if code:
                raise RuntimeError(f"Training exit {code}; see {folder / 'train.log'}")
            completed_result(job)
        except (RuntimeError, ValueError, KeyError, OSError, IndexError) as error:
            stop.set()
            print(f"FAILED {job['component']}: {error}", flush=True)
            return False
        print(f"PASS RC_{job['component'].upper()}", flush=True)
        return True

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        outcomes = list(pool.map(execute, plan["jobs"]))
    complete = summarize(directory)
    raise SystemExit(0 if all(outcomes) and complete else 1)


if __name__ == "__main__":
    main()
