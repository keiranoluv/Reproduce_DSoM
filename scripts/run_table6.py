#!/usr/bin/env python3
"""Plan, run, summarize Table 6. Default plan imports no Torch and starts no jobs."""

import argparse
import json
import math
import os
import shlex
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from run_table2 import MATCHED, baseline, digest, git, verify_inputs

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("MKG-Y", "MKG-W", "MMKB-DB15K")
METHODS = ("full", "structural", "visual", "description")
COPY = (
    "dataset",
    "model",
    "teacher_checkpoint",
    "img_info",
    "dscp_info",
    "modality_split",
    "fusion_img",
    "fusion_dscp",
    "modal_missing",
    "rl_hs",
    "rl_reward",
    "rl_penalize",
    "rl_weight",
    "policy_temp",
    "policy_input_norm",
    "rc_reward_mode",
    "l5_counterfactual",
)
SOURCES = (
    "src/kd.py",
    "src/kd_learn.py",
    "src/kd_optimizers.py",
    "src/models.py",
    "src/datasets.py",
    "src/regularizers.py",
    "src/utils.py",
    "scripts/run_table2.py",
    "scripts/run_table6.py",
    "tests/test_table6.py",
    "docs/experiments/TABLE6_REPRODUCTION.md",
    "docs/experiments/table6_paper_targets.json",
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode", choices=("plan", "smoke", "full", "summarize"), default="plan"
    )
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument(
        "--rerun-existing",
        action="store_true",
        help="Train selected cells fresh, including original full baselines",
    )
    p.add_argument("--output", type=Path)
    p.add_argument("--wandb", action="store_true")
    return p


def validate_metrics(values):
    if not math.isfinite(values["MR"]) or values["MR"] < 1:
        raise ValueError("Invalid MR")
    hits = values["hits@[1,3,10]"]
    if len(hits) != 3 or any(
        not math.isfinite(v) or not 0 <= v <= 1 for v in [values["MRR"], *hits]
    ):
        raise ValueError("Invalid MRR/Hits")
    if hits != sorted(hits):
        raise ValueError("Hits must be monotonic")


def selected_result(path):
    result = json.loads(path.read_text())
    checkpoints = list(path.parent.glob("student-*.pth"))
    if len(checkpoints) != 1 or checkpoints[0].stat().st_size == 0:
        raise ValueError(f"Missing/ambiguous student checkpoint: {path}")
    step = result["best_epoch"] / result["valid"]
    if step < 1 or not float(step).is_integer():
        raise ValueError(f"Invalid selected epoch: {path}")
    idx = int(step) - 1
    if result["final_result"] != result["curve"]["test"][idx]:
        raise ValueError(f"Test metric is not from selected checkpoint: {path}")
    if result["best_mrr"] != result["curve"]["valid"][idx]["MRR"]:
        raise ValueError(f"Validation metric is not from selected checkpoint: {path}")
    validate_metrics(result["final_result"])
    return result


def validate_reuse(result, original, method):
    expected = {key: original[key] for key in (*MATCHED, *COPY)}
    expected.update(
        max_train_examples=0,
        eval_queries=-1,
        teacher_strategy="rc" if method == "full" else method,
        kd_loss_type="kdrl" if method == "full" else "fixed_ndkd",
        policy_optimizer="source_frozen" if method == "full" else "same",
    )
    for key, value in expected.items():
        if result.get(key) != value:
            raise ValueError(
                f"Reference mismatch {key}: {result.get(key)!r} != {value!r}"
            )
    if method == "full" and (result.get("rc_distill_loss") or "ndkd") != "ndkd":
        raise ValueError("Full baseline must use NDKD")


def existing_reference(dataset, method, baseline_path, original):
    if method == "full":
        path = baseline_path
    elif dataset == "MMKB-DB15K" and method in ("structural", "description"):
        paths = list(
            (ROOT / "runs/student/MMKB-DB15K/fixed_teacher_ndkd_full").glob(
                f"*/{method}/checkpoints/*/result.json"
            )
        )
        if len(paths) != 1:
            raise ValueError(
                f"Expected one original DB15K {method} result, found {paths}"
            )
        path = paths[0]
    else:
        return None
    result = selected_result(path)
    validate_reuse(result, original, method)
    return {"path": str(path), "dataset": dataset, "method": method}


def build_command(original, method, mode, directory, commit):
    cfg = {key: original[key] for key in (*MATCHED, *COPY)}
    cfg.update(
        teacher_strategy="rc" if method == "full" else method,
        kd_loss_type="kdrl" if method == "full" else "fixed_ndkd",
        policy_optimizer="source_frozen" if method == "full" else "same",
        policy_lr=None,
        rc_distill_loss="ndkd",
        max_train_examples=0,
        eval_queries=-1,
        ckpt_dir=str(directory / "checkpoints"),
        note=f"T6_{original['dataset']}_{method}_{mode}_Seed2022",
        git_commit=commit,
        experiment_id="Table6_ModalityAblation",
    )
    if mode == "smoke":
        cfg.update(
            max_epochs=1,
            valid=1.0,
            early_stopping=2,
            max_train_examples=1000,
            eval_queries=64,
        )
    command = [str(ROOT / "dsom/bin/python"), "-u", "kd_learn.py"]
    for key, value in cfg.items():
        if value is None:
            continue
        if key in ("policy_input_norm", "l5_counterfactual"):
            if value:
                command.append("--" + key)
        else:
            command.extend(["--" + key, str(value)])
    return command, cfg


def prepare(args):
    if len(args.datasets) != len(set(args.datasets)) or len(args.methods) != len(
        set(args.methods)
    ):
        raise ValueError("Duplicate dataset or method")
    commit = git("rev-parse", "HEAD")
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    if args.mode == "full":
        if dirty:
            raise ValueError("Commit tracked changes before full training")
        for source in SOURCES:
            git("ls-files", "--error-unmatch", source)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    directory = (args.output or ROOT / "runs/table6" / args.mode / stamp).resolve()
    if not directory.is_relative_to(ROOT / "runs/table6") or directory.exists():
        raise ValueError("Use a NEW output directory inside runs/table6/")
    mode = "smoke" if args.mode == "smoke" else "full"
    plan = dict(
        mode=args.mode,
        training_mode=mode,
        git_commit=commit,
        tracked_tree_dirty=dirty,
        seed=2022,
        wandb=args.wandb,
        rerun_existing=args.rerun_existing,
        datasets=args.datasets,
        methods=args.methods,
        input_sha256={},
        source_sha256={name: digest(ROOT / name) for name in SOURCES},
        jobs=[],
        references=[],
        baselines={},
        limitations=[
            "Full-modality rows are source_frozen RC, not learned-RC evidence",
            "Fixed single-modality teacher logits; student always structural ComplEx",
            "Frozen Exp01 multimodal teacher is reused; no new teacher pretraining",
            "Released-code neighbor SUM, whereas paper Eq.12 describes MEAN",
            "Original DB15K validation contamination; not CleanV1",
            "Single seed 2022, reconstructed features, original min scheduler retained",
        ],
    )
    inputs = set()
    for dataset in args.datasets:
        path, original = baseline(dataset, root=ROOT)
        validate_reuse(selected_result(path), original, "full")
        plan["baselines"][dataset] = str(path)
        inputs.add(path)
        inputs.update(
            Path(original[key])
            for key in ("teacher_checkpoint", "img_info", "dscp_info")
        )
        inputs.update(
            ROOT / "data" / dataset / name
            for name in (
                "train.pickle",
                "valid.pickle",
                "test.pickle",
                "to_skip.pickle",
                "ent_id",
                "rel_id",
            )
        )
        for method in args.methods:
            reference = (
                None
                if args.rerun_existing
                else existing_reference(dataset, method, path, original)
            )
            if reference:
                inputs.add(Path(reference["path"]))
                plan["references"].append(reference)
            else:
                folder = directory / dataset / method
                command, config = build_command(original, method, mode, folder, commit)
                plan["jobs"].append(
                    dict(
                        dataset=dataset,
                        method=method,
                        directory=str(folder),
                        command=command,
                        config=config,
                    )
                )
    plan["input_sha256"] = {str(path): digest(path) for path in sorted(inputs)}
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (directory / "tracked_changes.patch").write_text(git("diff", "HEAD") + "\n")
    with zipfile.ZipFile(
        directory / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for source in SOURCES:
            archive.write(ROOT / source, source)
    for job in plan["jobs"]:
        folder = Path(job["directory"])
        folder.mkdir(parents=True)
        (folder / "command.sh").write_text(
            "cd "
            + shlex.quote(str(ROOT / "src"))
            + "\n"
            + shlex.join(job["command"])
            + "\n"
        )
    return directory, plan


def completed_result(job):
    paths = list((Path(job["directory"]) / "checkpoints").rglob("result.json"))
    if len(paths) != 1:
        raise ValueError(f"Missing/ambiguous result: {job['directory']}")
    result = selected_result(paths[0])
    for key, expected in job["config"].items():
        if result.get(key) != expected:
            raise ValueError(f"{key}: {result.get(key)!r} != {expected!r}")
    return paths[0], result


def summarize(directory):
    plan = json.loads((directory / "plan.json").read_text())
    verify_inputs(plan)
    # Use snapshotted targets so a future documentation change cannot alter old scores.
    with zipfile.ZipFile(directory / "source_snapshot.zip") as archive:
        paper = json.loads(archive.read("docs/experiments/table6_paper_targets.json"))
    rows, failures = [], []
    for cell, reused in [(cell, True) for cell in plan["references"]] + [
        (cell, False) for cell in plan["jobs"]
    ]:
        try:
            if reused:
                path = Path(cell["path"])
                result = selected_result(path)
                original = json.loads(
                    Path(plan["baselines"][cell["dataset"]]).read_text()
                )
                validate_reuse(result, original, cell["method"])
            else:
                path, result = completed_result(cell)
            target = paper[cell["dataset"]][cell["method"]]
            row = dict(
                dataset=cell["dataset"],
                method=cell["method"],
                reused=reused,
                best_epoch=result["best_epoch"],
                valid_mrr=result["best_mrr"],
                test=result["final_result"],
                paper=target,
                result=str(path),
                protocol="source_frozen RC"
                if cell["method"] == "full"
                else "fixed NDKD",
            )
            row["delta_mrr_pp"] = (
                (row["test"]["MRR"] - target["MRR"]) * 100
                if plan["training_mode"] == "full"
                else None
            )
            rows.append(row)
        except (ValueError, KeyError, OSError, IndexError) as error:
            failures.append(
                dict(dataset=cell["dataset"], method=cell["method"], error=str(error))
            )
    rows.sort(
        key=lambda r: (
            plan["datasets"].index(r["dataset"]),
            plan["methods"].index(r["method"]),
        )
    )
    wanted = {(d, m) for d in plan["datasets"] for m in plan["methods"]}
    actual = {(r["dataset"], r["method"]) for r in rows}
    if actual != wanted or len(rows) != len(actual):
        failures.append({"error": "Incomplete or duplicate table cells"})
    summary = dict(
        status="PASS" if not failures else "INCOMPLETE",
        training_mode=plan["training_mode"],
        rows=rows,
        failures=failures,
        selected_grid_completed=not failures,
        full_table6_grid_completed=not failures
        and plan["training_mode"] == "full"
        and actual == {(d, m) for d in DATASETS for m in METHODS},
        exact_author_protocol_reproduced=False,
        limitations=plan["limitations"],
    )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = [
        "# Table 6 — modality ablation",
        "",
        f"Mode: {plan['training_mode']}; smoke scores are execution checks only.",
        "Full-modality rows use source_frozen RC. Single-modality rows use fixed NDKD.",
        "Original benchmark splits; seed 2022; released SUM loss; reconstructed features.",
        "",
        "| Dataset | Method | Reused | Best epoch | Test MRR | H@1 | H@3 | H@10 | Paper MRR | Delta MRR (pp) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        delta = "n/a" if row["delta_mrr_pp"] is None else f"{row['delta_mrr_pp']:+.3f}"
        values = row["test"]
        text.append(
            f"| {row['dataset']} | {row['method']} | {row['reused']} | {row['best_epoch']} | {values['MRR']:.6f} | "
            + " | ".join(f"{value:.6f}" for value in values["hits@[1,3,10]"])
            + f" | {row['paper']['MRR']:.4f} | {delta} |"
        )
    (directory / "table6_results.md").write_text("\n".join(text) + "\n")
    print(f"{summary['status']}: {directory / 'summary.json'}", flush=True)
    return not failures


def execute_jobs(plan, stamp):
    """User-only smoke/full entry: sequential jobs stop on first failure."""
    for job in plan["jobs"]:
        folder = Path(job["directory"])
        env = dict(
            os.environ,
            DSOM_WANDB="1" if plan["wandb"] else "0",
            DSOM_RUN_DIR=str(folder),
            WANDB_RUN_GROUP=f"Table6_{plan['training_mode']}_{stamp}",
            WANDB_NAME=f"T6_{job['dataset']}_{job['method']}_{plan['training_mode']}",
            WANDB_DIR=str(folder / "wandb"),
        )
        (folder / "wandb").mkdir()
        print(
            f"START {job['dataset']} {job['method']}; log: {folder / 'train.log'}",
            flush=True,
        )
        try:
            with (folder / "train.log").open("w") as log:
                code = subprocess.run(
                    job["command"],
                    cwd=ROOT / "src",
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ).returncode
            if code:
                raise RuntimeError(f"Training exited {code}")
            completed_result(job)
        except (RuntimeError, ValueError, KeyError, OSError, IndexError) as error:
            (folder / "failure.json").write_text(
                json.dumps({"error": str(error)}, indent=2) + "\n"
            )
            print(f"FAILED {job['dataset']} {job['method']}: {error}", flush=True)
            return False
        print(f"PASS {job['dataset']} {job['method']}", flush=True)
    return True


def main():
    args = parser().parse_args()
    if args.mode == "summarize":
        if args.output is None:
            raise ValueError("summarize requires --output EXISTING_SUITE")
        raise SystemExit(0 if summarize(args.output.resolve()) else 1)
    directory, plan = prepare(args)
    print(f"Plan: {directory / 'plan.json'}", flush=True)
    print(
        f"{len(plan['jobs'])} new jobs; {len(plan['references'])} reused cells",
        flush=True,
    )
    if args.mode == "plan":
        print("PLAN ONLY: no training, Torch import, or GPU initialization")
        return
    success = execute_jobs(plan, directory.name)
    complete = summarize(directory)
    raise SystemExit(0 if success and complete else 1)


if __name__ == "__main__":
    main()
