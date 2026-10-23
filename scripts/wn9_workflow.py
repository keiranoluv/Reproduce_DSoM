"""Shared metadata/orchestration for separately runnable WN9 teacher and student.

No Torch import: GPU execution happens only in explicit smoke/full subprocesses.
"""

import argparse
import json
import math
import os
import re
import shlex
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from prepare_wn9_from_mose import verify_prepared
from run_table2 import digest, git, verify_inputs

ROOT = Path(__file__).resolve().parents[1]
PAPER = {"MRR": 0.9213, "hits@[1,3,10]": [0.9052, None, 0.9556]}
SOURCES = (
    "scripts/run_wn9_teacher.py",
    "scripts/run_wn9_student.py",
    "scripts/run_table7.py",
    "scripts/wn9_workflow.py",
    "scripts/eval_wn9_student.py",
    "src/wn9_evaluation.py",
    "scripts/prepare_wn9_from_mose.py",
    "scripts/run_table2.py",
    "src/learn.py",
    "src/kd_learn.py",
    "src/kd.py",
    "src/kd_optimizers.py",
    "src/models.py",
    "src/datasets.py",
    "src/optimizers.py",
    "src/regularizers.py",
    "src/utils.py",
    "tests/test_table7.py",
    "docs/experiments/TABLE7_WN9_REPRODUCTION.md",
)
LIMITATIONS = [
    "Original MoSE WN9 benchmark with inverse facts across splits; unchanged",
    "Reconstructed WN18 BERT/ViT features reused by synset ID, including 8 imputed images",
    "Released SUM NDKD and original min scheduler retained, seed 2022 only",
    "Teacher selected by filtered oracle validation Hits@10; not a final deployable score",
    "Student selected by filtered validation MRR; raw metrics use that same checkpoint",
    "Table 7 exact author raw/filtered protocol unresolved; both settings reported",
]


def parser(kind):
    p = argparse.ArgumentParser(
        description=f"WN9 {kind}; default plan starts no training"
    )
    modes = (
        ("plan", "smoke", "full", "summarize")
        if kind == "e2e"
        else ("plan", "smoke", "full")
    )
    p.add_argument("--mode", choices=modes, default="plan")
    p.add_argument("--output", type=Path)
    p.add_argument("--wandb", action="store_true")
    p.add_argument(
        "--policy-optimizer", choices=("source_frozen", "Adam"), default="source_frozen"
    )
    p.add_argument("--policy-lr", type=float, default=1e-5)
    if kind == "student":
        p.add_argument(
            "--teacher-run",
            type=Path,
            required=True,
            help="Completed teacher stage directory, not just a checkpoint path",
        )
    return p


def policy_limit(policy):
    return (
        "source_frozen policy is initialized/sampled but never optimized; no learned-RC claim"
        if policy == "source_frozen"
        else "Separate-Adam learned RC at configured LR; adaptation, not verified author optimizer"
    )


def source_guard(mode):
    if mode == "full":
        if git("status", "--porcelain", "--untracked-files=no"):
            raise ValueError("Commit tracked changes before full training")
        for source in SOURCES:
            git("ls-files", "--error-unmatch", source)


def inputs():
    manifest = verify_prepared(ROOT / "data/WN9")
    result = {**manifest["input_sha256"], **manifest["output_sha256"]}
    path = ROOT / "data/WN9/preparation_manifest.json"
    result[str(path)] = digest(path)
    return result


def new_directory(kind, mode, output):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    base = ROOT / "runs/table7"
    default = base / mode / stamp if kind == "e2e" else base / kind / mode / stamp
    directory = (output or default).resolve()
    if not directory.is_relative_to(base) or directory.exists():
        raise ValueError("Use a NEW directory within runs/table7/")
    directory.mkdir(parents=True)
    return directory


def base_config(mode):
    return dict(
        dataset="WN9",
        model="ComplExMDR",
        modality_split=True,
        fusion_img=True,
        fusion_dscp=True,
        modal_missing=0.0,
        alpha=1.0,
        scale=20.0,
        img_info=str(ROOT / "data/WN9/img_vec.pickle"),
        dscp_info=str(ROOT / "data/WN9/dscp_vec.pickle"),
        rank=2000,
        batch_size=1000,
        optimizer="Adagrad",
        learning_rate=0.1,
        regularizer="N3",
        reg=0.01,
        init=0.001,
        decay1=0.9,
        decay2=0.999,
        scheduler_mode="min",
        max_epochs=1 if mode == "smoke" else 200,
        valid=1.0 if mode == "smoke" else 3.0,
        early_stopping=2 if mode == "smoke" else 10,
        max_train_examples=1000 if mode == "smoke" else 0,
        eval_queries=64 if mode == "smoke" else -1,
    )


def build_command(
    stage,
    mode,
    directory,
    commit,
    policy="source_frozen",
    policy_lr=1e-5,
    teacher_checkpoint=None,
):
    config = base_config(mode)
    config.update(
        ckpt_dir=str(directory / "checkpoints"),
        git_commit=commit,
        experiment_id="Table7_WN9",
        note=f"T7_WN9_{stage}_{mode}_Seed2022",
    )
    if stage == "student":
        config.update(
            teacher_checkpoint=teacher_checkpoint or "<PENDING_SELECTED_WN9_TEACHER>",
            kd_loss_type="kdrl",
            teacher_strategy="rc",
            rc_distill_loss="ndkd",
            kd_temp=4.0,
            kd_ce_weight=1.0,
            kd_kl_weight=2.0,
            dkd_alpha=1.0,
            dkd_beta=1.0,
            policy_optimizer=policy,
            policy_lr=policy_lr if policy == "Adam" else None,
            rl_hs=1024,
            rl_reward=1.0,
            rl_penalize=10.0,
            rl_weight=1.0,
            policy_temp=1.0,
            policy_input_norm=False,
            rc_reward_mode="binary",
            l5_counterfactual=False,
        )
        config["note"] += "_" + policy
    script = "learn.py" if stage == "teacher" else "kd_learn.py"
    command = [sys.executable, "-u", script]
    for key, value in config.items():
        if value is None:
            continue
        if key in ("policy_input_norm", "l5_counterfactual"):
            if value:
                command.append("--" + key)
        else:
            command.extend(["--" + key, str(value)])
    return command, config


def snapshot(directory, plan):
    plan["source_sha256"] = {name: digest(ROOT / name) for name in SOURCES}
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    with zipfile.ZipFile(
        directory / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in SOURCES:
            archive.write(ROOT / name, name)


def training_result(directory, stage, config):
    paths = list((directory / "checkpoints").rglob("result.json"))
    pattern = "m-*.pth" if stage == "teacher" else "student-*.pth"
    checkpoints = list((directory / "checkpoints").rglob(pattern))
    if len(paths) != 1 or len(checkpoints) != 1 or checkpoints[0].stat().st_size == 0:
        raise ValueError(
            f"Missing/ambiguous {stage} result or checkpoint in {directory}"
        )
    result = json.loads(paths[0].read_text())
    for key, expected in config.items():
        if result.get(key) != expected:
            raise ValueError(
                f"{stage} config mismatch: {key}: {result.get(key)!r} != {expected!r}"
            )
    scores = [
        v["hits@[1,3,10]"][2] if stage == "teacher" else v["MRR"]
        for v in result["curve"]["valid"]
    ]
    step = result["best_epoch"] / result["valid"]
    if not float(step).is_integer() or result["best_epoch"] > result["max_epochs"]:
        raise ValueError(f"{stage}: invalid checkpoint epoch")
    idx = int(step) - 1
    if idx < 0 or not scores or idx != scores.index(max(scores)):
        raise ValueError(f"{stage}: wrong validation-selected checkpoint epoch")
    if result["final_result"] != result["curve"]["test"][idx]:
        raise ValueError(
            f"{stage}: final_result differs from selected test curve entry"
        )
    if result["best_mrr"] != result["curve"]["valid"][idx]["MRR"]:
        raise ValueError(f"{stage}: selected validation MRR mismatch")
    if not result["curve_loss"] or not all(
        math.isfinite(v) for v in result["curve_loss"]
    ):
        raise ValueError(f"{stage}: invalid training loss history")
    return paths[0], checkpoints[0], result


def teacher_reference(directory, mode, hashes):
    directory = directory.resolve()
    plan = json.loads((directory / "plan.json").read_text())
    if plan["stage"] != "teacher" or (mode != "smoke" and plan["mode"] != "full"):
        raise ValueError(
            "Full student requires a full WN9 teacher stage, never smoke/WN18"
        )
    verify_inputs(plan)
    # WN9 dataset/features and preprocessing source must be identical across stages.
    for name, expected in hashes.items():
        if plan["input_sha256"].get(name) != expected:
            raise ValueError(f"Teacher data provenance mismatch: {name}")
    cfg = plan["config"]
    if cfg["dataset"] != "WN9" or cfg["max_train_examples"] != (
        1000 if plan["mode"] == "smoke" else 0
    ):
        raise ValueError("Teacher is not a compatible WN9 run")
    result_path, checkpoint, _ = training_result(directory, "teacher", cfg)
    completed = json.loads((directory / "stage_result.json").read_text())
    if (
        completed["status"] != "PASS"
        or completed["checkpoint_sha256"] != digest(checkpoint)
        or completed["result_sha256"] != digest(result_path)
    ):
        raise ValueError("Teacher stage did not pass or checkpoint has changed")
    return checkpoint


def run_process(command, directory, log_name, env):
    print(f"RUN: {shlex.join(command)}\nLOG: {directory / log_name}", flush=True)
    with (directory / log_name).open("w") as stream:
        code = subprocess.run(
            command, cwd=ROOT / "src", env=env, stdout=stream, stderr=subprocess.STDOUT
        ).returncode
    if code:
        raise RuntimeError(f"Exit {code}; see {directory / log_name}")


def verify_evaluation(directory, config, training, checkpoint):
    report = json.loads((directory / "evaluation.json").read_text())
    if report["dataset"] != "WN9" or Path(report["checkpoint"]) != checkpoint.resolve():
        raise ValueError("Evaluation checkpoint/dataset mismatch")
    if report["query_limit"] != (64 if config["max_train_examples"] else 0):
        raise ValueError("Evaluation query limit mismatch")
    for split, count in (("valid", 1337), ("test", 1319)):
        if report["results"][split]["queries_per_direction"] != (
            64 if config["max_train_examples"] else count
        ):
            raise ValueError("Incomplete evaluation coverage")
        raw = report["results"][split]["raw"]["both"]
        filtered = report["results"][split]["filtered"]["both"]
        for values in (raw, filtered):
            if (
                not math.isfinite(values["MR"])
                or values["MR"] < 1
                or any(
                    not math.isfinite(v) or not 0 <= v <= 1
                    for v in (values["MRR"], *values["hits@[1,3,10]"])
                )
            ):
                raise ValueError("Invalid evaluation metrics")
        if raw["MRR"] > filtered["MRR"] + 1e-9 or raw["MR"] < filtered["MR"] - 1e-9:
            raise ValueError("Raw/filtered ranking ordering violated")
    if not config["max_train_examples"]:
        # With full data, check independently reloaded checkpoint against training metrics.
        for split, expected in (
            ("valid", training["best_mrr"]),
            ("test", training["final_result"]["MRR"]),
        ):
            if not math.isclose(
                report["results"][split]["filtered"]["both"]["MRR"],
                expected,
                rel_tol=2e-6,
                abs_tol=2e-6,
            ):
                raise ValueError(
                    f"{split}: reloaded filtered MRR differs from training checkpoint"
                )
        values = report["results"]["test"]["filtered"]["both"]
        for actual, expected in zip(
            values["hits@[1,3,10]"], training["final_result"]["hits@[1,3,10]"]
        ):
            if not math.isclose(actual, expected, abs_tol=2e-6):
                raise ValueError(
                    "Reloaded filtered Hits differ from training checkpoint"
                )
    return report


def run_stage(stage, args, output=None, teacher_run=None, group=None):
    if not math.isfinite(args.policy_lr) or args.policy_lr <= 0:
        raise ValueError("policy-lr must be positive and finite")
    source_guard(args.mode)
    hashes = inputs()
    teacher_checkpoint = None
    if stage == "student":
        teacher_run = teacher_run or args.teacher_run
        teacher_checkpoint = teacher_reference(teacher_run, args.mode, hashes)
        hashes = dict(hashes, **{str(teacher_checkpoint): digest(teacher_checkpoint)})
        for name in ("plan.json", "stage_result.json"):
            path = teacher_run.resolve() / name
            hashes[str(path)] = digest(path)
    directory = new_directory(stage, args.mode, output or args.output)
    command, cfg = build_command(
        stage,
        args.mode,
        directory,
        git("rev-parse", "HEAD"),
        args.policy_optimizer,
        args.policy_lr,
        str(teacher_checkpoint) if teacher_checkpoint else None,
    )
    plan = dict(
        stage=stage,
        mode=args.mode,
        config=cfg,
        command=command,
        input_sha256=hashes,
        limitations=[*LIMITATIONS, policy_limit(args.policy_optimizer)],
    )
    if stage == "student":
        plan["teacher_run"] = str(teacher_run.resolve())
    snapshot(directory, plan)
    (directory / "command.sh").write_text(
        "cd " + shlex.quote(str(ROOT / "src")) + "\n" + shlex.join(command) + "\n"
    )
    print(f"{stage.upper()} plan: {directory / 'plan.json'}", flush=True)
    if args.mode == "plan":
        return directory
    (directory / "checkpoints").mkdir()
    (directory / "wandb").mkdir()
    env = dict(
        os.environ,
        DSOM_WANDB="1" if args.wandb else "0",
        DSOM_RUN_DIR=str(directory),
        WANDB_NAME=f"T7_WN9_{stage}_{args.mode}_{args.policy_optimizer}",
        WANDB_RUN_GROUP=group or f"Table7_{args.mode}_{directory.name}",
        WANDB_DIR=str(directory / "wandb"),
    )
    policy_updates = None
    try:
        run_process(command, directory, "train.log", env)
        result_path, checkpoint, result = training_result(directory, stage, cfg)
        if stage == "student":
            if cfg["policy_optimizer"] == "Adam":
                updates = re.findall(
                    r"Batches with updated policy weights:\s*(\d+)",
                    (directory / "train.log").read_text(errors="replace"),
                )
                policy_updates = sum(int(v) for v in updates)
                if not updates or (args.mode == "full" and not policy_updates):
                    raise ValueError("Learned RC has no recorded policy updates")
            evaluation_command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts/eval_wn9_student.py"),
                "--checkpoint",
                str(checkpoint),
                "--output",
                str(directory / "evaluation.json"),
                "--device",
                "cuda",
                "--queries",
                "64" if args.mode == "smoke" else "0",
            ]
            plan["evaluation_command"] = evaluation_command
            (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
            run_process(evaluation_command, directory, "evaluation.log", env)
            verify_evaluation(directory, cfg, result, checkpoint)
        verify_inputs(plan)
        payload = dict(
            status="PASS",
            stage=stage,
            mode=args.mode,
            checkpoint=str(checkpoint),
            checkpoint_sha256=digest(checkpoint),
            result=str(result_path),
            result_sha256=digest(result_path),
            best_epoch=result["best_epoch"],
            validation_selection="oracle Hits@10"
            if stage == "teacher"
            else "filtered MRR",
        )
        if stage == "student":
            payload["evaluation_sha256"] = digest(directory / "evaluation.json")
            payload["policy_update_batches"] = policy_updates
        (directory / "stage_result.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )
    except Exception as error:
        (directory / "failure.json").write_text(
            json.dumps({"error": repr(error)}, indent=2) + "\n"
        )
        raise
    print(f"PASS {stage}: {directory / 'stage_result.json'}", flush=True)
    return directory


def summarize(directory):
    root_plan = json.loads((directory / "plan.json").read_text())
    verify_inputs(root_plan)
    teacher_dir, student_dir = directory / "teacher", directory / "student"
    student_plan = json.loads((student_dir / "plan.json").read_text())
    if student_plan["stage"] != "student" or student_plan["mode"] != root_plan["mode"]:
        raise ValueError("Student stage/mode differs from the E2E plan")
    for name, expected in root_plan["input_sha256"].items():
        if student_plan["input_sha256"].get(name) != expected:
            raise ValueError(f"Student data provenance mismatch: {name}")
    teacher_checkpoint = teacher_reference(
        teacher_dir, root_plan["mode"], root_plan["input_sha256"]
    )
    if student_plan["config"]["teacher_checkpoint"] != str(teacher_checkpoint):
        raise ValueError("Student does not use this E2E teacher checkpoint")
    verify_inputs(student_plan)
    if student_plan["config"]["policy_optimizer"] != root_plan["policy_optimizer"]:
        raise ValueError("Student policy optimizer differs from the E2E plan")
    result_path, checkpoint, trained = training_result(
        student_dir, "student", student_plan["config"]
    )
    stage = json.loads((student_dir / "stage_result.json").read_text())
    if (
        stage["status"] != "PASS"
        or stage["checkpoint_sha256"] != digest(checkpoint)
        or stage["result_sha256"] != digest(result_path)
    ):
        raise ValueError("Student checkpoint failed or changed")
    if stage["evaluation_sha256"] != digest(student_dir / "evaluation.json"):
        raise ValueError("Evaluation changed after student stage passed")
    evaluated = verify_evaluation(
        student_dir, student_plan["config"], trained, checkpoint
    )
    paper = root_plan["paper"]
    rows = []
    for protocol in ("raw", "filtered"):
        values = evaluated["results"]["test"][protocol]["both"]
        rows.append(
            dict(
                protocol=protocol,
                test=values,
                delta_mrr_pp=(values["MRR"] - paper["MRR"]) * 100
                if root_plan["mode"] == "full"
                else None,
            )
        )
    summary = dict(
        status="PASS",
        mode=root_plan["mode"],
        full_table7_dsom_row_completed=root_plan["mode"] == "full",
        exact_author_protocol_reproduced=False,
        policy_optimizer=student_plan["config"]["policy_optimizer"],
        teacher_checkpoint=str(teacher_checkpoint),
        student_checkpoint=str(checkpoint),
        best_student_epoch=trained["best_epoch"],
        paper=paper,
        rows=rows,
        limitations=root_plan["limitations"],
    )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = [
        "# Table 7 — WN9 DSoM row",
        "",
        f"Mode: {root_plan['mode']}; policy optimizer: {summary['policy_optimizer']}.",
        "Student chosen by filtered validation MRR; raw/filtered share one checkpoint.",
        "Paper evaluation protocol unresolved. Smoke metrics are not benchmark results.",
        "",
        "| Setting | MRR | H@1 | H@3 | H@10 | Delta paper MRR (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        v = row["test"]
        delta = "n/a" if row["delta_mrr_pp"] is None else f"{row['delta_mrr_pp']:+.3f}"
        text.append(
            f"| {row['protocol']} | {v['MRR']:.6f} | "
            + " | ".join(f"{h:.6f}" for h in v["hits@[1,3,10]"])
            + f" | {delta} |"
        )
    text.append("| Paper DSoM | .9213 | .9052 | — | .9556 | reference |")
    (directory / "table7_results.md").write_text("\n".join(text) + "\n")
    print(f"PASS E2E: {directory / 'summary.json'}", flush=True)
    return summary


def e2e(args):
    if args.mode == "summarize":
        if args.output is None:
            raise ValueError("summarize requires --output EXISTING_SUITE")
        return summarize(args.output.resolve())
    if not math.isfinite(args.policy_lr) or args.policy_lr <= 0:
        raise ValueError("policy-lr must be positive and finite")
    source_guard(args.mode)
    directory = new_directory("e2e", args.mode, args.output)
    plan = dict(
        mode=args.mode,
        input_sha256=inputs(),
        policy_optimizer=args.policy_optimizer,
        paper=PAPER,
        limitations=[*LIMITATIONS, policy_limit(args.policy_optimizer)],
        stages={},
    )
    for stage in ("teacher", "student"):
        command, config = build_command(
            stage,
            args.mode,
            directory / stage,
            git("rev-parse", "HEAD"),
            args.policy_optimizer,
            args.policy_lr,
        )
        plan["stages"][stage] = dict(command=command, config=config)
    snapshot(directory, plan)
    print(f"E2E plan: {directory / 'plan.json'}", flush=True)
    if args.mode == "plan":
        print(
            "PLAN ONLY: teacher -> selected checkpoint -> student -> raw/filtered evaluation; no Torch import"
        )
        return directory
    try:
        teacher = run_stage(
            "teacher", args, output=directory / "teacher", group=directory.name
        )
        run_stage(
            "student",
            args,
            output=directory / "student",
            teacher_run=teacher,
            group=directory.name,
        )
        summarize(directory)
    except Exception as error:
        (directory / "failure.json").write_text(
            json.dumps({"error": repr(error)}, indent=2) + "\n"
        )
        raise
    return directory
