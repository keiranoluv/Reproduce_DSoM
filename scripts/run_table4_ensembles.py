#!/usr/bin/env python3
"""Prepare the remaining Table 4 teacher rows. Default plan never imports Torch.

Only the user invokes --mode smoke/full. No student or teacher retraining.
"""

import argparse
import json
import os
import pickle
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from run_table2 import digest, git, verify_inputs
from run_table4_components import load_references

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "scripts/run_table4_ensembles.py",
    "src/table4_ensembles.py",
    "tests/test_table4_ensembles.py",
    "docs/experiments/TABLE4_TEACHER_ENSEMBLES.md",
    "docs/experiments/table4_ensemble_sources.json",
    "scripts/run_table2.py",
    "scripts/run_table4_components.py",
    "src/models.py",
    "src/datasets.py",
)
PAPER = {"teacher_mean_metrics": 0.3764, "ml": 0.4062, "rc": 0.4108}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("plan", "smoke", "full"), default="plan")
    p.add_argument(
        "--methods", nargs="+", choices=("avg", "ml", "rc"), default=["avg", "ml", "rc"]
    )
    p.add_argument("--output", type=Path)
    p.add_argument(
        "--student-suite",
        type=Path,
        default=ROOT / "runs/table4_components/full/20261023_075850_052725",
    )
    p.add_argument("--ml-epochs", type=int, default=20)
    p.add_argument("--rc-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=2022)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--references", type=Path)
    return p


def prepare(args):
    for value in (
        args.ml_epochs,
        args.rc_epochs,
        args.batch_size,
        args.eval_batch_size,
    ):
        if value <= 0:
            raise ValueError("Epoch counts and batch sizes must be positive")
    if len(args.methods) != len(set(args.methods)):
        raise ValueError("Duplicate methods")
    if args.mode == "full":
        if git("status", "--porcelain", "--untracked-files=no"):
            raise ValueError(
                "Full run requires a clean tracked working tree; commit first"
            )
        for name in SOURCES:
            git("ls-files", "--error-unmatch", name)
    references = (
        load_references(args.references) if args.references else load_references()
    )
    a5 = references["A5_RC_NDKD"]["result"]
    student_suite = args.student_suite.resolve()
    student_plan = json.loads((student_suite / "plan.json").read_text())
    verify_inputs(student_plan)
    student_summary = json.loads((student_suite / "summary.json").read_text())
    if (
        student_summary["status"] != "PASS"
        or not student_summary["component_pair_completed"]
    ):
        raise ValueError("Student component suite is not complete")
    inputs = dict(student_plan["input_sha256"])
    for path in (
        student_suite / "summary.json",
        student_suite / "plan.json",
        *(Path(row["result"]) for row in student_summary["rows"]),
    ):
        inputs[str(path)] = digest(path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    directory = (
        args.output or ROOT / "runs/table4_ensembles" / args.mode / timestamp
    ).resolve()
    if not directory.is_relative_to(ROOT / "runs/table4_ensembles"):
        raise ValueError("Output must be inside runs/table4_ensembles")
    directory.mkdir(parents=True, exist_ok=False)
    config = dict(
        methods=args.methods,
        seed=args.seed,
        teacher_checkpoint=a5["teacher_checkpoint"],
        dataset="MMKB-DB15K",
        ml_epochs=1 if args.mode == "smoke" else args.ml_epochs,
        rc_epochs=1 if args.mode == "smoke" else args.rc_epochs,
        batch_size=16 if args.mode == "smoke" else args.batch_size,
        eval_batch_size=16 if args.mode == "smoke" else args.eval_batch_size,
        queries_per_direction=64 if args.mode == "smoke" else 0,
        ml_hidden=32,
        ml_optimizer="Adagrad",
        ml_lr=0.1,
        rc_hidden=1024,
        rc_optimizer="Adam",
        rc_lr=1e-5,
        rc_reference="mean teacher logits",
        rc_reward=1.0,
        rc_penalty=-10.0,
        rc_inference="argmax, dropout disabled",
        checkpoint_selection="fixed final epoch",
        train_split="valid only, RHS then LHS, duplicates retained",
        teacher_avg_primary="mean of three modality metrics; mean-logit diagnostic also saved",
    )
    plan = dict(
        mode=args.mode,
        git_commit=git("rev-parse", "HEAD"),
        config=config,
        student_suite=str(student_suite),
        input_sha256=inputs,
        source_sha256={name: digest(ROOT / name) for name in SOURCES},
        limitations=[
            "Original contaminated DB15K validation, single seed, reconstructed features",
            "Teacher Avg wording ambiguous: report both mean metrics and mean logits",
            "ML architecture/optimizer follow MoSE; fixed final epoch replaces test selection",
            "RC-on-val is an explicit adaptation; original standalone training code unavailable",
            "RC binary advantage uses average-teacher CE reference, not a student",
        ],
    )
    if args.references:
        plan.update(
            reference_registry=str(args.references.resolve()),
            reference_registry_sha256=digest(args.references),
            reference_registry_snapshot=json.loads(args.references.read_text()),
        )
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    with zipfile.ZipFile(directory / "source_snapshot.zip", "w") as archive:
        for name in SOURCES:
            archive.write(ROOT / name, name)
    return directory, plan


def load_queries(config):
    """Load the frozen benchmark mapping; reciprocal relations use half-offset."""
    import numpy as np
    import torch

    folder = ROOT / "data" / config["dataset"]
    with (folder / "train.pickle").open("rb") as stream:
        train = pickle.load(stream)
    entities = int(max(train[:, 0].max(), train[:, 2].max()) + 1)
    relations = int(train[:, 1].max() + 1)
    with (folder / "to_skip.pickle").open("rb") as stream:
        filters = pickle.load(stream)
    queries = {}
    for split in ("valid", "test"):
        with (folder / f"{split}.pickle").open("rb") as stream:
            values = pickle.load(stream)
        limit = config["queries_per_direction"]
        if limit:
            rng = np.random.default_rng(config["seed"])
            values = values[rng.permutation(len(values))[:limit]]
        rhs = torch.as_tensor(values.astype("int64"))
        lhs = rhs[:, [2, 1, 0]].clone()
        lhs[:, 1] += relations
        queries[split] = {"rhs": rhs, "lhs": lhs}
    return (entities, 2 * relations, entities), queries, filters


def seed_all(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_head(teacher, method, config, queries, device, directory, tracker=None):
    import torch

    from table4_ensembles import MetaLearner, ValidationRC, fit_epoch

    seed_all(config["seed"])
    if method == "ml":
        model = MetaLearner(config["ml_hidden"]).to(device)
        optimizer = torch.optim.Adagrad(model.parameters(), lr=config["ml_lr"])
    else:
        model = ValidationRC(teacher.sizes[0], config["rc_hidden"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["rc_lr"])
    history = []
    for epoch in range(1, config[method + "_epochs"] + 1):
        entry = dict(
            epoch=epoch,
            **fit_epoch(
                teacher, model, optimizer, queries, device, config["batch_size"]
            ),
        )
        history.append(entry)
        (directory / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(method, json.dumps(entry), flush=True)
        if tracker:
            tracker.log(
                {
                    "epoch": epoch,
                    **{"train/" + k: v for k, v in entry.items() if k != "epoch"},
                }
            )
    if not any(row["batches_with_nonzero_gradient"] for row in history):
        raise ValueError(f"{method}: no nonzero gradients; head did not learn")
    # State dict and optimizer state only; no copy of the large frozen teacher.
    checkpoint = directory / "final_head.pth"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": config[method + "_epochs"],
            "method": method,
            "config": config,
        },
        checkpoint,
    )
    # Evaluate exactly the saved artifact rather than an unsaved in-memory state.
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def write_summary(directory, plan, results):
    verify_inputs(plan)
    student = json.loads((Path(plan["student_suite"]) / "summary.json").read_text())
    rows = []
    for method in plan["config"]["methods"]:
        keys = (
            ("teacher_mean_metrics", "teacher_mean_logits")
            if method == "avg"
            else (method,)
        )
        for name in keys:
            values = results[method]["test"][name]
            target = PAPER.get(name)
            rows.append(
                dict(
                    method=name,
                    test=values,
                    paper_mrr=target,
                    delta_mrr_pp=(values["MRR"] - target) * 100
                    if target is not None and plan["mode"] == "full"
                    else None,
                )
            )
    complete = plan["mode"] == "full" and set(plan["config"]["methods"]) == {
        "avg",
        "ml",
        "rc",
    }
    report = dict(
        status="PASS",
        mode=plan["mode"],
        teacher_rows_completed=complete,
        all_nine_rows_have_results=complete and student["component_pair_completed"],
        exact_author_protocol_reproduced=False,
        teacher_rows=rows,
        student_rows=student["rows"],
        limitations=plan["limitations"],
    )
    (directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Table 4 — teacher ensemble adaptation and existing student results",
        "",
        f"Mode: {plan['mode']}. Smoke scores are not benchmark results.",
        "",
        "ML uses a fixed final epoch; RC-on-val is adapted. Teacher Avg is ambiguous.",
        "",
        "| Method | MRR | H@1 | H@3 | H@10 | Paper MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows + student["rows"]:
        values = row["test"]
        target = row.get("paper_mrr", row.get("paper", [None])[0])
        text_target = f"{target:.6f}" if target is not None else "diagnostic"
        lines.append(
            f"| {row['method']} | {values['MRR']:.6f} | "
            + " | ".join(f"{v:.6f}" for v in values["hits@[1,3,10]"])
            + f" | {text_target} |"
        )
    (directory / "table4_results.md").write_text("\n".join(lines) + "\n")


def execute(directory, plan, wandb_enabled):
    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from models import ComplExMDR
    from table4_ensembles import evaluate

    config = plan["config"]
    device = torch.device("cuda")
    # This path is entered only with explicit user smoke/full mode.
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU execution requires CUDA; use --mode plan for CPU review"
        )
    shape, queries, filters = load_queries(config)
    teacher = torch.load(
        config["teacher_checkpoint"], map_location=device, weights_only=False
    )
    if not isinstance(teacher, ComplExMDR) or tuple(teacher.sizes) != shape:
        raise ValueError("Unexpected teacher class or benchmark shape")
    if not (teacher.modality_split and teacher.fusion_img and teacher.fusion_dscp):
        raise ValueError("All three teacher modalities are required")
    teacher.to(device).requires_grad_(False).eval()
    # Feature tensors in old full-object checkpoints are plain attributes, not buffers.
    for name in ("img_vec", "dscp_vec"):
        setattr(teacher, name, getattr(teacher, name).to(device))
    results = {}
    for method in config["methods"]:
        job = directory / method
        job.mkdir()
        tracker = None
        try:
            if wandb_enabled:
                import wandb

                tracker = wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "DSoM-Reproduction"),
                    name=f"T4_DB15K_Teacher_{method}_{plan['mode']}",
                    group=directory.name,
                    dir=str(job),
                    config=config,
                    tags=["table4", plan["mode"], "teacher-ensemble-adaptation"],
                )
            head = (
                None
                if method == "avg"
                else train_head(
                    teacher, method, config, queries["valid"], device, job, tracker
                )
            )
            heads = {} if head is None else {method: head}
            result = {
                split: evaluate(
                    teacher,
                    heads,
                    queries[split],
                    filters,
                    device,
                    config["eval_batch_size"],
                )
                for split in ("valid", "test")
            }
            result.update(method=method, config=config, selection="fixed final epoch")
            (job / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            if tracker:
                key = "teacher_mean_metrics" if method == "avg" else method
                for split in ("valid", "test"):
                    tracker.summary[split + "/MRR"] = result[split][key]["MRR"]
                    for k, value in zip(
                        (1, 3, 10), result[split][key]["hits@[1,3,10]"]
                    ):
                        tracker.summary[f"{split}/Hits@{k}"] = value
            results[method] = result
            print(f"{method} test:", result["test"], flush=True)
            del head, heads
        finally:
            if tracker:
                tracker.finish(exit_code=0 if method in results else 1)
    write_summary(directory, plan, results)


def main():
    args = parser().parse_args()
    directory, plan = prepare(args)
    print("Plan:", directory / "plan.json", flush=True)
    if args.mode == "plan":
        print("Plan only. No Torch import, training, or GPU initialization.")
        return
    try:
        execute(directory, plan, args.wandb)
    except Exception as error:
        (directory / "failure.json").write_text(
            json.dumps({"error": repr(error)}, indent=2) + "\n"
        )
        raise
    print("PASS:", directory / "summary.json")


if __name__ == "__main__":
    main()
