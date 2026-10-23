"""CPU scoring/integration checks; orchestration tests never launch real GPU jobs."""

import argparse
import ast
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import wn9_workflow as flow  # noqa: E402

from kd import KD_RL  # noqa: E402
from kd_optimizers import KDOptimizer  # noqa: E402
from models import ComplExMDR  # noqa: E402
from optimizers import KBCOptimizer  # noqa: E402
from regularizers import N3  # noqa: E402
from wn9_evaluation import evaluate_student, ranking_pair  # noqa: E402


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "_lazy_init",
        lambda *a, **k: pytest.fail("CUDA initialization forbidden"),
    )


def training_parser(stage):
    source = ROOT / "src" / ("learn.py" if stage == "teacher" else "kd_learn.py")
    names = {
        "big_datasets",
        "datasets",
        "models",
        "regularizers",
        "optimizers",
        "parser",
    }
    body = []
    for node in ast.parse(source.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            body.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            f = node.value.func
            if (
                isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Name)
                and f.value.id == "parser"
                and f.attr == "add_argument"
            ):
                body.append(node)
    env = {"argparse": argparse, "ast": ast}
    exec(
        compile(ast.Module(body=body, type_ignores=[]), "CPU CLI definitions", "exec"),
        env,
    )
    return env["parser"]


@pytest.mark.parametrize("stage", ["teacher", "student"])
@pytest.mark.parametrize("mode", ["plan", "smoke", "full"])
@pytest.mark.parametrize("policy", ["source_frozen", "Adam"])
def test_stage_commands_parse_using_actual_training_cli(stage, mode, policy, tmp_path):
    command, cfg = flow.build_command(
        stage,
        mode,
        tmp_path,
        "commit",
        policy,
        teacher_checkpoint="/fixture/WN9-teacher.pth",
    )
    parsed = vars(training_parser(stage).parse_args(command[3:]))
    for key, value in cfg.items():
        assert parsed[key] == value, key
    assert parsed["dataset"] == "WN9" and parsed["rank"] == 2000
    assert parsed["max_train_examples"] == (1000 if mode == "smoke" else 0)
    assert parsed["max_epochs"] == (1 if mode == "smoke" else 200)


@pytest.mark.parametrize("id_type", [int, np.uint64])
def test_raw_filtered_known_answers_and_pessimistic_ties(id_type):
    scores = torch.tensor([[8.0, 4.0, 4.0, 7.0, 0.0]])
    targets = torch.tensor([[4.0]])
    query = torch.tensor([[0, 0, 1]])
    filters = {(0, 0): [id_type(0), id_type(1), id_type(3)]}
    before = copy.deepcopy(filters)
    raw, filtered = ranking_pair(scores, targets, query, filters)
    assert raw.item() == 4 and filtered.item() == 2
    assert filters == before
    assert scores[0, 1] == 4


def test_actual_tiny_teacher_to_student_and_checkpoint_scoring(tmp_path, monkeypatch):
    # Actual legacy modules, small synthetic data, all .cuda transfers redirected
    # to CPU in this test; global CUDA initialization is forbidden independently.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    import pickle

    for name in ("image", "text"):
        with (tmp_path / name).open("wb") as f:
            pickle.dump(np.arange(12, dtype=np.float32).reshape(4, 3) / 100, f)
    torch.manual_seed(2022)
    args = SimpleNamespace(
        modal_missing=0.0,
        fusion_img=True,
        fusion_dscp=True,
        policy_optimizer="source_frozen",
        rl_weight=1.0,
        dkd_alpha=1.0,
        dkd_beta=1.0,
        rc_distill_loss="ndkd",
        rc_reward_mode="binary",
        policy_input_norm=False,
        l5_counterfactual=False,
    )
    teacher = ComplExMDR(
        (4, 4, 4),
        2,
        img_info=str(tmp_path / "image"),
        dscp_info=str(tmp_path / "text"),
        dataset="WN9",
        args=args,
    )
    triples = torch.tensor([[0, 0, 1], [1, 1, 2], [1, 2, 0], [2, 3, 1]])
    initial = copy.deepcopy(teacher.state_dict())
    teacher_optim = KBCOptimizer(
        teacher,
        N3(0.01),
        torch.optim.Adagrad(teacher.parameters(), lr=0.1),
        batch_size=2,
        modality_split=True,
        fusion_img=True,
        fusion_dscp=True,
    )
    teacher_optim.epoch(triples)
    assert any(not torch.equal(v, initial[k]) for k, v in teacher.state_dict().items())
    teacher_path = tmp_path / "teacher.pth"
    torch.save(teacher, teacher_path)
    teacher = (
        torch.load(teacher_path, map_location="cpu", weights_only=False)
        .requires_grad_(False)
        .eval()
    )
    frozen = copy.deepcopy(teacher.state_dict())
    student = ComplExMDR(
        (4, 4, 4),
        2,
        modality_split=False,
        fusion_img=False,
        fusion_dscp=False,
        dataset="WN9",
        args=args,
    )
    distiller = KD_RL(student, teacher, 4, rl_hs=8, args=args)
    policy_before = copy.deepcopy(distiller.rewarder.state_dict())
    train_masks = {(h, r): [t] for h, r, t in triples.tolist()}
    student_optim = KDOptimizer(
        distiller,
        N3(0.01),
        torch.optim.Adagrad(student.parameters(), lr=0.1),
        batch_size=2,
        kd_loss="kdrl",
        args=args,
        to_skip=train_masks,
        verbose=False,
    )
    before = copy.deepcopy(student.state_dict())
    assert torch.isfinite(student_optim.epoch(triples, 0))
    assert any(not torch.equal(v, before[k]) for k, v in student.state_dict().items())
    assert all(torch.equal(v, frozen[k]) for k, v in teacher.state_dict().items())
    assert all(
        torch.equal(v, policy_before[k])
        for k, v in distiller.rewarder.state_dict().items()
    )
    path = tmp_path / "student.pth"
    torch.save(student, path)
    student = torch.load(path, map_location="cpu", weights_only=False)
    examples = np.array([[0, 0, 1], [1, 1, 2]], dtype=np.uint64)
    filters = {
        "rhs": {(0, 0): [np.uint64(1)], (1, 1): [np.uint64(2)]},
        "lhs": {(1, 2): [np.uint64(0)], (2, 3): [np.uint64(1)]},
    }
    values = evaluate_student(student, examples, filters, "cpu", batch_size=2)
    assert values["queries_per_direction"] == 2
    for side in ("rhs", "lhs"):
        q = torch.tensor(examples.astype("int64"))
        if side == "lhs":
            q = q[:, [2, 1, 0]]
            q[:, 1] += 2
        reference = student.get_ranking(q, filters[side], batch_size=2)
        assert values["filtered"][side]["MRR"] == pytest.approx(
            reference.double().reciprocal().mean().item()
        )
    assert values["raw"]["both"]["MRR"] <= values["filtered"]["both"]["MRR"]
    with pytest.raises(ValueError, match="structural student"):
        evaluate_student(teacher, examples, filters, "cpu")


def metric(mrr=0.6):
    return {"MR": 4.0, "MRR": mrr, "hits@[1,3,10]": [0.4, 0.65, 0.8]}


def mock_training(directory):
    plan = json.loads((directory / "plan.json").read_text())
    cfg = plan["config"]
    stage = plan["stage"]
    count = 1 if plan["mode"] == "smoke" else 2
    valid = (
        [metric(0.9), metric(0.85)][:count]
        if stage == "teacher"
        else [metric(0.7), metric(0.75)][:count]
    )
    if count == 2:
        valid[0]["hits@[1,3,10]"][2] = 0.7
    test = [metric() for _ in range(count)]
    best = count - 1
    result = dict(
        cfg,
        best_epoch=int((best + 1) * cfg["valid"]),
        best_mrr=valid[best]["MRR"],
        curve={"valid": valid, "test": test},
        final_result=test[best],
        curve_loss=[0.9, 0.8],
    )
    folder = directory / "checkpoints/0"
    folder.mkdir()
    (folder / "result.json").write_text(json.dumps(result))
    (
        folder / ("m-fixture.pth" if stage == "teacher" else "student-fixture.pth")
    ).write_bytes(b"CPU orchestration fixture")
    (directory / "train.log").write_text("Batches with updated policy weights: 1\n")


def mock_evaluation(directory):
    plan = json.loads((directory / "plan.json").read_text())
    cfg = plan["config"]
    results = {}
    for split, size in (("valid", 1337), ("test", 1319)):
        filtered = metric(0.75 if split == "valid" else 0.6)
        raw = dict(metric(0.5), MR=5.0)
        results[split] = {
            "queries_per_direction": 64 if plan["mode"] == "smoke" else size,
            "raw": {"both": raw},
            "filtered": {"both": filtered},
        }
    report = dict(
        dataset="WN9",
        checkpoint=str(directory / "checkpoints/0/student-fixture.pth"),
        query_limit=64 if cfg["max_train_examples"] else 0,
        results=results,
    )
    (directory / "evaluation.json").write_text(json.dumps(report))


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "ROOT", tmp_path)
    monkeypatch.setattr(flow, "SOURCES", ())
    monkeypatch.setattr(
        flow, "git", lambda *args: "commit" if args[0] == "rev-parse" else ""
    )
    data = tmp_path / "frozen-data"
    data.write_bytes(b"WN9 fixture")
    monkeypatch.setattr(flow, "inputs", lambda: {str(data): flow.digest(data)})

    def process(command, directory, log_name, env):
        if log_name == "train.log":
            mock_training(directory)
        else:
            mock_evaluation(directory)

    monkeypatch.setattr(flow, "run_process", process)
    return tmp_path


def test_e2e_plan_does_not_launch_processes(workspace, monkeypatch):
    monkeypatch.setattr(
        flow, "run_process", lambda *a: pytest.fail("Plan must not run")
    )
    directory = flow.e2e(flow.parser("e2e").parse_args([]))
    plan = json.loads((directory / "plan.json").read_text())
    assert "<PENDING_" in plan["stages"]["student"]["config"]["teacher_checkpoint"]
    assert not (directory / "teacher").exists()


@pytest.mark.parametrize("policy", ["source_frozen", "Adam"])
def test_e2e_full_chain_and_summarize_with_mocked_training(workspace, policy):
    directory = flow.e2e(
        flow.parser("e2e").parse_args(["--mode", "full", "--policy-optimizer", policy])
    )
    result = flow.summarize(directory)
    assert result["status"] == "PASS" and result["full_table7_dsom_row_completed"]
    assert result["policy_optimizer"] == policy
    assert not result["exact_author_protocol_reproduced"]
    teacher_plan = json.loads((directory / "teacher/plan.json").read_text())
    assert (
        flow.training_result(directory / "teacher", "teacher", teacher_plan["config"])[
            2
        ]["best_mrr"]
        == 0.85
    )
    assert len(result["rows"]) == 2


def test_smoke_chain_never_claims_table7_reproduction(workspace):
    directory = flow.e2e(flow.parser("e2e").parse_args(["--mode", "smoke"]))
    summary = json.loads((directory / "summary.json").read_text())
    assert not summary["full_table7_dsom_row_completed"]
    assert all(row["delta_mrr_pp"] is None for row in summary["rows"])


def test_independent_teacher_then_student_stages(workspace):
    teacher_args = flow.parser("teacher").parse_args(["--mode", "full"])
    teacher = flow.run_stage("teacher", teacher_args)
    args = flow.parser("student").parse_args(
        ["--mode", "full", "--teacher-run", str(teacher)]
    )
    student = flow.run_stage("student", args)
    assert json.loads((student / "stage_result.json").read_text())["status"] == "PASS"


def test_full_student_rejects_smoke_teacher(workspace):
    teacher = flow.run_stage(
        "teacher", flow.parser("teacher").parse_args(["--mode", "smoke"])
    )
    args = flow.parser("student").parse_args(
        ["--mode", "full", "--teacher-run", str(teacher)]
    )
    with pytest.raises(ValueError, match="full WN9 teacher"):
        flow.run_stage("student", args)


def test_teacher_failure_stops_student(workspace, monkeypatch):
    calls = []

    def fail(command, directory, *args):
        calls.append(directory)
        raise RuntimeError("mock training failed")

    monkeypatch.setattr(flow, "run_process", fail)
    with pytest.raises(RuntimeError, match="failed"):
        flow.e2e(flow.parser("e2e").parse_args(["--mode", "smoke"]))
    assert len(calls) == 1 and calls[0].name == "teacher"
    assert not (calls[0].parent / "student").exists()
    assert (calls[0].parent / "failure.json").exists()


def test_checkpoint_tampering_is_rejected(workspace):
    teacher = flow.run_stage(
        "teacher", flow.parser("teacher").parse_args(["--mode", "full"])
    )
    next((teacher / "checkpoints").rglob("m-*.pth")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint has changed"):
        flow.teacher_reference(teacher, "full", flow.inputs())


def test_evaluation_coverage_and_metric_mismatch_rejected(workspace):
    directory = flow.e2e(flow.parser("e2e").parse_args(["--mode", "full"]))
    student = directory / "student"
    cfg = json.loads((student / "plan.json").read_text())["config"]
    _, checkpoint, trained = flow.training_result(student, "student", cfg)
    p = student / "evaluation.json"
    value = json.loads(p.read_text())
    value["results"]["test"]["queries_per_direction"] = 64
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="coverage"):
        flow.verify_evaluation(student, cfg, trained, checkpoint)
    value["results"]["test"]["queries_per_direction"] = 1319
    value["results"]["test"]["filtered"]["both"]["MRR"] = 0.7
    p.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="reloaded filtered MRR"):
        flow.verify_evaluation(student, cfg, trained, checkpoint)


def test_full_guard_rejects_dirty_tree(monkeypatch):
    monkeypatch.setattr(flow, "git", lambda *args: "dirty")
    with pytest.raises(ValueError, match="Commit tracked"):
        flow.source_guard("full")


def test_full_summary_rejects_student_stage_mode_mismatch(workspace):
    directory = flow.e2e(flow.parser("e2e").parse_args(["--mode", "full"]))
    path = directory / "student/plan.json"
    plan = json.loads(path.read_text())
    plan["mode"] = "smoke"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="stage/mode"):
        flow.summarize(directory)


def test_cpu_checkpoint_evaluation_entrypoint(tmp_path, monkeypatch):
    import pickle

    import eval_wn9_student as worker

    model = ComplExMDR(
        (6555, 18, 6555),
        1,
        modality_split=False,
        fusion_img=False,
        fusion_dscp=False,
        dataset="WN9",
    )
    checkpoint = tmp_path / "synthetic-untrained-student.pth"
    torch.save(model, checkpoint)
    data = tmp_path / "data/WN9"
    data.mkdir(parents=True)
    examples = np.array([[0, 0, 1], [1, 0, 2]], dtype=np.uint64)
    for split in ("valid", "test"):
        with (data / f"{split}.pickle").open("wb") as stream:
            pickle.dump(examples, stream)
    filters = {
        "rhs": {(0, 0): [np.uint64(1)], (1, 0): [np.uint64(2)]},
        "lhs": {(1, 9): [np.uint64(0)], (2, 9): [np.uint64(1)]},
    }
    with (data / "to_skip.pickle").open("wb") as stream:
        pickle.dump(filters, stream)
    monkeypatch.setattr(worker, "ROOT", tmp_path)
    output = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "eval_wn9_student.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(output),
            "--device",
            "cpu",
            "--queries",
            "2",
        ],
    )
    worker.main()
    report = json.loads(output.read_text())
    assert report["device"] == "cpu"
    assert report["results"]["test"]["queries_per_direction"] == 2
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_zero_learned_policy_updates_only_allowed_for_smoke(
    workspace, monkeypatch, mode
):
    original = flow.run_process

    def zero_updates(command, directory, log_name, env):
        original(command, directory, log_name, env)
        if log_name == "train.log":
            (directory / log_name).write_text(
                "Batches with updated policy weights: 0\n"
            )

    monkeypatch.setattr(flow, "run_process", zero_updates)
    args = flow.parser("e2e").parse_args(["--mode", mode, "--policy-optimizer", "Adam"])
    if mode == "full":
        with pytest.raises(ValueError, match="policy updates"):
            flow.e2e(args)
    else:
        directory = flow.e2e(args)
        stage = json.loads((directory / "student/stage_result.json").read_text())
        assert stage["policy_update_batches"] == 0
