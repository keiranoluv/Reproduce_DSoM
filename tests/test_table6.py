"""CPU checks for modality selection, real optimizer branch, and Table 6 orchestration."""

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_table6 as runner  # noqa: E402
from test_table2 import Regularizer, Student, Teacher, examples  # noqa: E402
from test_table4_components import a5, argument_parser  # noqa: E402

from kd import KD_FixedStrategy  # noqa: E402
from kd_optimizers import KDOptimizer  # noqa: E402


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("CUDA initialization forbidden")

    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


@pytest.mark.parametrize(
    "strategy,indices",
    [
        ("structural", [0]),
        ("visual", [1]),
        ("description", [2]),
        ("structural_description", [0, 2]),
        ("average", [0, 1, 2]),
    ],
)
def test_fixed_modality_selection_and_actual_ndkd_update(
    strategy, indices, monkeypatch
):
    # Redirect the legacy GPU data transfer for this tiny synthetic CPU test only.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    teacher, student = Teacher(), Student()
    teacher_before = copy.deepcopy(teacher.state_dict())
    student_before = copy.deepcopy(student.state_dict())
    distiller = KD_FixedStrategy(student, teacher, strategy)
    distiller.train()
    assert not teacher.training
    _, _, selected = distiller.forward_train(examples(), 0)
    expected = teacher.scores.detach()[indices].mean(0)
    torch.testing.assert_close(selected, expected)
    assert not selected.requires_grad
    assert {id(p) for p in distiller.get_learnable_parameters()} == {
        id(p) for p in student.parameters()
    }
    optimizer = KDOptimizer(
        distiller,
        Regularizer(),
        torch.optim.Adagrad(student.parameters(), lr=0.1),
        batch_size=2,
        temperature=4.0,
        kd_loss="fixed_ndkd",
        verbose=False,
        args=SimpleNamespace(policy_optimizer="same", dkd_alpha=1.0, dkd_beta=1.0),
        to_skip={(0, 0): [0, 2], (1, 1): [1]},
    )
    loss = optimizer.epoch(examples(), 0)
    assert torch.isfinite(loss)
    assert any(
        not torch.equal(v, student_before[k]) for k, v in student.state_dict().items()
    )
    assert all(
        torch.equal(v, teacher_before[k]) for k, v in teacher.state_dict().items()
    )
    assert all(p.grad is None for p in teacher.parameters())


def original():
    value = a5()
    value.update(policy_optimizer="source_frozen", policy_lr=None, rc_distill_loss=None)
    return value


@pytest.mark.parametrize("method", runner.METHODS)
@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_generated_commands_parse_with_actual_training_cli(method, mode, tmp_path):
    base = original()
    command, config = runner.build_command(base, method, mode, tmp_path, "commit")
    parsed = vars(argument_parser().parse_args(command[3:]))
    for key, value in config.items():
        assert parsed[key] == value, key
    assert parsed["kd_loss_type"] == ("kdrl" if method == "full" else "fixed_ndkd")
    assert parsed["teacher_strategy"] == ("rc" if method == "full" else method)
    for key in (*runner.MATCHED, *runner.COPY):
        if mode == "smoke" and key in ("max_epochs", "valid", "early_stopping"):
            continue
        assert parsed[key] == base[key], key
    assert parsed["max_train_examples"] == (1000 if mode == "smoke" else 0)
    assert parsed["eval_queries"] == (64 if mode == "smoke" else -1)


def metrics():
    return {"MR": 20.0, "MRR": 0.4, "hits@[1,3,10]": [0.3, 0.45, 0.6]}


def save_result(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(
        cfg,
        best_epoch=int(cfg["valid"]),
        best_mrr=0.8,
        final_result=metrics(),
        curve={"test": [metrics()], "valid": [{"MRR": 0.8}]},
    )
    path.write_text(json.dumps(value))
    (path.parent / "student-fixture.pth").write_bytes(
        b"not a real model; file-check fixture"
    )
    return value


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "git", lambda *args: "commit" if args[0] == "rev-parse" else ""
    )
    for source in runner.SOURCES:
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            (ROOT / source).read_bytes()
            if source.endswith("table6_paper_targets.json")
            else b"CPU fixture"
        )
        path.write_bytes(content)
    for dataset in runner.DATASETS:
        cfg = original()
        cfg["dataset"] = dataset
        for key in ("teacher_checkpoint", "img_info", "dscp_info"):
            path = tmp_path / "inputs" / dataset / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
            cfg[key] = str(path)
        data = tmp_path / "data" / dataset
        data.mkdir(parents=True)
        for name in (
            "train.pickle",
            "valid.pickle",
            "test.pickle",
            "to_skip.pickle",
            "ent_id",
            "rel_id",
        ):
            (data / name).write_bytes(b"fixture")
        path = (
            tmp_path
            / "runs/student"
            / dataset
            / "full/original/checkpoints/0/result.json"
        )
        save_result(path, cfg)
        if dataset == "MMKB-DB15K":
            for method in ("structural", "description"):
                other = dict(
                    cfg,
                    teacher_strategy=method,
                    kd_loss_type="fixed_ndkd",
                    policy_optimizer="same",
                )
                path = (
                    tmp_path
                    / "runs/student"
                    / dataset
                    / "fixed_teacher_ndkd_full/original"
                    / method
                    / "checkpoints/0/result.json"
                )
                save_result(path, other)
    return tmp_path


def test_default_plan_has_seven_jobs_five_references_and_never_runs(
    workspace, monkeypatch
):
    def fail(*args, **kwargs):
        pytest.fail("Plan must not start a subprocess")

    monkeypatch.setattr(runner.subprocess, "run", fail)
    monkeypatch.setattr(sys, "argv", ["run_table6.py"])
    runner.main()
    plan = json.loads(
        next((workspace / "runs/table6/plan").glob("*/plan.json")).read_text()
    )
    assert len(plan["jobs"]) == 7
    assert len(plan["references"]) == 5
    assert [(j["dataset"], j["method"]) for j in plan["jobs"]] == [
        ("MKG-Y", "structural"),
        ("MKG-Y", "visual"),
        ("MKG-Y", "description"),
        ("MKG-W", "structural"),
        ("MKG-W", "visual"),
        ("MKG-W", "description"),
        ("MMKB-DB15K", "visual"),
    ]


def test_rerun_existing_trains_all_twelve_cells(workspace):
    _, plan = runner.prepare(runner.parser().parse_args(["--rerun-existing"]))
    assert len(plan["jobs"]) == 12
    assert not plan["references"]
    assert (
        sum(j["config"]["policy_optimizer"] == "source_frozen" for j in plan["jobs"])
        == 3
    )


def test_complete_summary_uses_selected_metrics_and_snapshotted_targets(workspace):
    directory, plan = runner.prepare(runner.parser().parse_args([]))
    for job in plan["jobs"]:
        save_result(Path(job["directory"]) / "checkpoints/0/result.json", job["config"])
    (workspace / "docs/experiments/table6_paper_targets.json").write_text("{}")
    assert runner.summarize(directory)
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["full_table6_grid_completed"]
    assert not summary["exact_author_protocol_reproduced"]
    assert len(summary["rows"]) == 12
    assert sum(r["reused"] for r in summary["rows"]) == 5
    assert all(r["test"]["MRR"] == 0.4 for r in summary["rows"])
    assert summary["rows"][0]["paper"]["MRR"] == 0.4002


def test_missing_output_is_incomplete_not_pass(workspace):
    directory, _ = runner.prepare(runner.parser().parse_args([]))
    assert not runner.summarize(directory)
    summary = json.loads((directory / "summary.json").read_text())
    assert not summary["full_table6_grid_completed"]
    assert summary["status"] == "INCOMPLETE"


def test_changed_original_input_is_rejected(workspace):
    directory, plan = runner.prepare(runner.parser().parse_args([]))
    Path(next(iter(plan["input_sha256"]))).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        runner.summarize(directory)


def test_smoke_and_subset_never_claim_full_benchmark(workspace):
    args = runner.parser().parse_args(
        ["--mode", "smoke", "--datasets", "MKG-Y", "--methods", "visual"]
    )
    directory, plan = runner.prepare(args)
    assert len(plan["jobs"]) == 1 and not plan["references"]
    job = plan["jobs"][0]
    save_result(Path(job["directory"]) / "checkpoints/0/result.json", job["config"])
    assert runner.summarize(directory)
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["selected_grid_completed"]
    assert not summary["full_table6_grid_completed"]
    assert summary["rows"][0]["delta_mrr_pp"] is None


def test_config_and_checkpoint_selection_mismatch_rejected(workspace):
    _, plan = runner.prepare(runner.parser().parse_args([]))
    job = plan["jobs"][0]
    path = Path(job["directory"]) / "checkpoints/0/result.json"
    result = save_result(path, job["config"])
    result["teacher_strategy"] = "average"
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="teacher_strategy"):
        runner.completed_result(job)
    result["teacher_strategy"] = job["method"]
    result["final_result"]["MRR"] = 0.9
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="selected checkpoint"):
        runner.completed_result(job)


def test_failed_job_stops_later_jobs(workspace, monkeypatch):
    _, plan = runner.prepare(runner.parser().parse_args([]))
    calls = []

    def fail(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    assert not runner.execute_jobs(plan, "fixture")
    assert len(calls) == 1
    assert calls[0][1]["env"]["DSOM_WANDB"] == "0"
    assert (Path(plan["jobs"][0]["directory"]) / "failure.json").exists()
    assert not (Path(plan["jobs"][1]["directory"]) / "train.log").exists()


def test_dirty_full_guard_precedes_work(monkeypatch):
    monkeypatch.setattr(runner, "git", lambda *args: "dirty")
    with pytest.raises(ValueError, match="Commit tracked"):
        runner.prepare(runner.parser().parse_args(["--mode", "full"]))


def test_reused_result_must_match_teacher_and_training_protocol():
    value = original()
    value.update(
        teacher_strategy="structural",
        kd_loss_type="fixed_ndkd",
        policy_optimizer="same",
    )
    runner.validate_reuse(value, original(), "structural")
    value["kd_kl_weight"] = 1.0
    with pytest.raises(ValueError, match="kd_kl_weight"):
        runner.validate_reuse(value, original(), "structural")
