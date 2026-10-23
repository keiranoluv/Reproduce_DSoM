"""Small synthetic CPU tests; no GPU or real training jobs required."""

import argparse
import ast
import importlib.util
import io
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
from kd import KD_BestConf, KD_BestStrategy, KD_BestTeacher, KD_MulDE, MulDESenior  # noqa: E402
from kd_optimizers import KDOptimizer  # noqa: E402


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("These tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(
            torch.tensor([[0.2, -0.1, 0.4, 0.0, -0.3], [0.1, 0.5, -0.2, 0.0, 0.3]])
        )

    def forward(self, x):
        p = self.scores[x[:, 0]]
        return p, p


class Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scores = torch.nn.Parameter(
            torch.tensor(
                [
                    [[3.0, 0.0, 1.0, -1.0, -2.0], [4.0, 1.0, 0.0, -1.0, -2.0]],
                    [[0.0, 5.0, 1.0, -1.0, -2.0], [0.0, 3.0, 1.0, -1.0, -2.0]],
                    [[2.0, 1.0, 0.0, -1.0, -2.0], [1.0, 2.0, 0.0, -1.0, -2.0]],
                ]
            )
        )

    def forward(self, x):
        p = self.scores[:, x[:, 0]]
        return p[0], None, p[1], None, p[2], None


def examples():
    return torch.tensor([[0, 0, 0], [1, 1, 1]])


@pytest.mark.parametrize("kind", ["confidence", "teacher", "strategy"])
def test_selection_matches_independent_definition(kind):
    x, teacher, student = examples(), Teacher(), Student()
    cls = {
        "confidence": KD_BestConf,
        "teacher": KD_BestTeacher,
        "strategy": KD_BestStrategy,
    }[kind]
    distiller = cls(student, teacher)
    _, _, selected = distiller.forward_train(x, 0)
    logits = teacher.scores.detach().transpose(0, 1)  # B, M, E
    candidates = logits
    if kind == "strategy":
        w = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5],
                [1 / 3, 1 / 3, 1 / 3],
            ]
        )
        candidates = torch.einsum("sm,bme->bse", w, logits)
    if kind == "confidence":
        choices = candidates.softmax(-1).amax(-1).argmax(-1)
    else:
        costs = torch.stack(
            [
                F.cross_entropy(candidates[:, i], x[:, 2], reduction="none")
                for i in range(candidates.shape[1])
            ],
            dim=1,
        )
        choices = costs.argmin(1)
    expected = candidates[torch.arange(len(x)), choices]
    torch.testing.assert_close(selected, expected)
    assert not selected.requires_grad
    assert {id(p) for p in distiller.get_learnable_parameters()} == {
        id(p) for p in student.parameters()
    }


def test_best_teacher_averages_exact_ties():
    teacher = Teacher()
    with torch.no_grad():
        teacher.scores.copy_(
            torch.ones_like(teacher.scores) * torch.arange(3.0)[:, None, None]
        )
    _, _, selected = KD_BestTeacher(Student(), teacher).forward_train(examples(), 0)
    torch.testing.assert_close(selected, torch.ones_like(selected))


def test_best_strategy_can_select_pair_better_than_each_single_teacher():
    teacher = Teacher()
    with torch.no_grad():
        teacher.scores[:, 0].copy_(
            torch.tensor(
                [
                    [2.0, 3.0, -5.0, -5.0, -5.0],
                    [2.0, -5.0, 3.0, -5.0, -5.0],
                    [0.0, 0.0, 0.0, 5.0, 0.0],
                ]
            )
        )
    _, _, selected = KD_BestStrategy(Student(), teacher).forward_train(
        examples()[:1], 0
    )
    expected = torch.tensor([[2.0, -1.0, -1.0, -5.0, -5.0]])
    torch.testing.assert_close(selected, expected)


def test_senior_formula_temperature_and_missing_target():
    senior = MulDESenior(4)
    teachers = torch.tensor([[[1.0, 2.0], [2.0, -1.0], [-1.0, 3.0]]])
    junior = torch.tensor([[0.3, 0.7]], requires_grad=True)
    relations = torch.tensor([1])
    candidates, truth = torch.tensor([[0, 2]]), torch.tensor([1])
    labels, loss, attention = senior(teachers, junior, relations, candidates, truth, 0)
    p = teachers.softmax(-1)
    divergence = (p * (p.log() - junior.detach().log_softmax(-1)[:, None])).mean(-1)
    expected_attention = (-divergence).softmax(1)
    torch.testing.assert_close(attention, expected_attention)
    torch.testing.assert_close(
        labels, (teachers * 0.5 * (3 * attention[:, :, None])).sum(1)
    )
    expected_loss = F.softplus((teachers * 0.5).sum(1)).sum(-1).mean()
    torch.testing.assert_close(loss, expected_loss)
    late = senior(teachers, junior, relations, candidates, truth, 100)[2]
    torch.testing.assert_close(late, torch.full_like(late, 1 / 3), atol=1e-6, rtol=0)
    assert math.isfinite(loss.item())
    loss.backward()
    assert junior.grad is None
    assert senior.relation_weights.weight.grad[1].abs().sum() > 0
    assert senior.relation_weights.weight.grad[0].abs().sum() == 0


class Regularizer:
    def forward(self, factors):
        return 0.01 * factors.square().mean()


@pytest.mark.parametrize(
    "method", ["best_conf", "best_teacher", "best_strategy", "mulde"]
)
def test_actual_optimizer_branch_updates_student_and_senior_on_cpu(method, monkeypatch):
    # Redirect the legacy loader's .cuda() inside this test only. CUDA initialization
    # is independently forbidden by the autouse fixture above.
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *a, **kw: self)
    student, teacher = Student(), Teacher()
    teacher_before = teacher.scores.detach().clone()
    student_before = student.scores.detach().clone()
    args = SimpleNamespace(policy_optimizer="same", dkd_alpha=1.0, dkd_beta=1.0)
    senior_optimizer = None
    if method == "mulde":
        distiller = KD_MulDE(student, teacher, 4, topk=3)
        senior_before = distiller.senior.relation_weights.weight.detach().clone()
        senior_optimizer = torch.optim.Adam(distiller.senior.parameters(), lr=0.001)
        loss_type = "mulde"
        epochs = []
        original = distiller.forward_train

        def capture(x, epoch):
            epochs.append(epoch)
            return original(x, epoch)

        monkeypatch.setattr(distiller, "forward_train", capture)
    else:
        cls = {
            "best_conf": KD_BestConf,
            "best_teacher": KD_BestTeacher,
            "best_strategy": KD_BestStrategy,
        }[method]
        distiller, loss_type = cls(student, teacher), "selection_ndkd"
    optimizer = KDOptimizer(
        distiller,
        Regularizer(),
        torch.optim.Adagrad(student.parameters(), lr=0.1),
        batch_size=2,
        temperature=4.0,
        kd_loss=loss_type,
        args=args,
        to_skip={(0, 0): [0, 2], (1, 1): [1]},
        policy_optimizer=senior_optimizer,
        verbose=False,
    )
    loss = optimizer.epoch(examples(), 7)
    assert torch.isfinite(loss)
    assert not torch.equal(student_before, student.scores)
    torch.testing.assert_close(teacher_before, teacher.scores)
    assert teacher.scores.grad is None
    if method == "mulde":
        assert not torch.equal(senior_before, distiller.senior.relation_weights.weight)
        assert epochs == [7]  # The attention schedule uses epoch, not batch index.
        assert optimizer.senior_weight_change > 0


def test_invalid_topk_is_rejected():
    with pytest.raises(ValueError):
        KD_MulDE(Student(), Teacher(), 4, topk=1)


@pytest.mark.parametrize("kind", ["strategy", "mulde"])
def test_checkpoint_roundtrip_after_backward_on_cpu(kind):
    teacher, student = Teacher().requires_grad_(False), Student()
    distiller = (
        KD_MulDE(student, teacher, 4, topk=3)
        if kind == "mulde"
        else KD_BestStrategy(student, teacher)
    )
    prediction, _, soft = distiller.forward_train(examples(), 0)
    loss = prediction.square().mean() + soft.square().mean()
    if kind == "mulde":
        loss = loss + distiller.last_senior_loss
    loss.backward()
    checkpoint = io.BytesIO()
    torch.save(distiller, checkpoint)
    checkpoint.seek(0)
    restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for name, tensor in distiller.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor)
    torch.testing.assert_close(restored.forward_test(examples())[0], prediction)


def runner():
    path = SRC.parent / "scripts/run_table2.py"
    spec = importlib.util.spec_from_file_location("table2_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_settings():
    return dict(
        teacher_checkpoint="/fixture/teacher.pth",
        img_info="/fixture/img.pickle",
        dscp_info="/fixture/text.pickle",
        rank=2000,
        batch_size=1000,
        optimizer="Adagrad",
        learning_rate=0.1,
        regularizer="N3",
        reg=0.01,
        scheduler_mode="min",
        kd_temp=4.0,
        kd_ce_weight=1.0,
        kd_kl_weight=2.0,
        dkd_alpha=1.0,
        dkd_beta=1.0,
        init=0.001,
        alpha=1,
        scale=20,
        decay1=0.9,
        decay2=0.999,
        max_epochs=200,
        valid=3.0,
        early_stopping=10,
    )


def training_argument_parser_without_loading_cuda():
    # Compile only literal enum assignments and argparse setup, never imports,
    # seeding, checkpoint loading, CUDA setup, or the training entry point.
    tree = ast.parse((SRC / "kd_learn.py").read_text())
    enums = {
        "big_datasets",
        "datasets",
        "models",
        "regularizers",
        "optimizers",
        "parser",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in enums for t in node.targets
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
    namespace = {"argparse": argparse, "ast": ast}
    exec(
        compile(
            ast.Module(body=body, type_ignores=[]),
            "CPU-only argument definitions",
            "exec",
        ),
        namespace,
    )
    return namespace["parser"]


@pytest.mark.parametrize(
    "method", ["best_conf", "best_teacher", "best_strategy", "mulde"]
)
@pytest.mark.parametrize("dataset", ["MMKB-DB15K", "MKG-W", "MKG-Y"])
def test_planned_commands_match_exp01_and_select_correct_objective(
    method, dataset, tmp_path
):
    module = runner()
    original = original_settings()
    full = module.build_command(dataset, method, original, tmp_path, "full", "commit")

    def value(key):
        return full[full.index("--" + key) + 1]

    for key in module.MATCHED:
        assert value(key) == str(original[key])
    assert value("teacher_checkpoint") == original["teacher_checkpoint"]
    assert value("teacher_strategy") == method
    assert value("kd_loss_type") == ("mulde" if method == "mulde" else "selection_ndkd")
    assert value("policy_optimizer") == "same"
    assert value("max_train_examples") == "0" and value("eval_queries") == "-1"
    smoke = module.build_command(dataset, method, original, tmp_path, "smoke", "commit")
    assert smoke[smoke.index("--max_epochs") + 1] == "1"
    assert smoke[smoke.index("--eval_queries") + 1] == "64"
    assert smoke.count("--max_epochs") == 1
    parser = training_argument_parser_without_loading_cuda()
    for command in (full, smoke):
        parsed = parser.parse_args(command[3:])
        assert parsed.dataset == dataset and parsed.teacher_strategy == method


def test_default_plan_cannot_start_subprocess_training(tmp_path, monkeypatch, capsys):
    import json

    module = runner()
    module.ROOT, module.SOURCES = tmp_path, ()
    original = original_settings()
    original_path = tmp_path / "baseline.json"
    original_path.write_text(json.dumps(original))
    for key in ("teacher_checkpoint", "img_info", "dscp_info"):
        p = tmp_path / (key + ".fixture")
        p.write_bytes(b"fixture")
        original[key] = str(p)
    data = tmp_path / "data/MKG-Y"
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
    monkeypatch.setattr(module, "baseline", lambda dataset: (original_path, original))
    monkeypatch.setattr(
        module, "git", lambda *args: "commit" if args[0] == "rev-parse" else ""
    )
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "")

    def forbid_process(*args, **kwargs):
        raise AssertionError("Plan mode must not start training")

    monkeypatch.setattr(module.subprocess, "run", forbid_process)
    out = tmp_path / "runs/table2/plan/test"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_table2.py",
            "--datasets",
            "MKG-Y",
            "--methods",
            "best_conf",
            "--output",
            str(out),
        ],
    )
    module.main()
    plan = json.loads((out / "plan.json").read_text())
    assert plan["mode"] == "plan" and len(plan["jobs"]) == 1
    assert not list(out.rglob("result.json"))
    assert "PLAN ONLY" in capsys.readouterr().out


def test_full_mode_requires_commit_before_training(monkeypatch):
    module = runner()
    monkeypatch.setattr(
        module, "git", lambda *args: " M src/kd.py" if args[0] == "status" else "commit"
    )
    monkeypatch.setattr(sys, "argv", ["run_table2.py", "--mode", "full"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2


@pytest.mark.parametrize("has_checkpoint", [True, False])
def test_summary_uses_best_checkpoint_metrics_and_marks_missing_jobs(
    tmp_path, has_checkpoint
):
    import json

    module = runner()
    module.ROOT = tmp_path
    targets = tmp_path / "docs/experiments/table2_paper_targets.json"
    targets.parent.mkdir(parents=True)
    targets.write_bytes(
        (SRC.parent / "docs/experiments/table2_paper_targets.json").read_bytes()
    )
    original_path = tmp_path / "baseline.json"
    metrics = {"MR": 100.0, "MRR": 0.4, "hits@[1,3,10]": [0.3, 0.4, 0.5]}
    original_path.write_text(json.dumps({"best_epoch": 9, "final_result": metrics}))
    original = original_settings()
    out = tmp_path / "runs/table2/full/test"
    job_dir = out / "MKG-Y/best_conf"
    checkpoint_dir = job_dir / "checkpoints/0"
    checkpoint_dir.mkdir(parents=True)
    command = module.build_command(
        "MKG-Y", "best_conf", original, job_dir, "full", "commit"
    )
    job = {
        "dataset": "MKG-Y",
        "method": "best_conf",
        "directory": str(job_dir),
        "teacher_checkpoint": original["teacher_checkpoint"],
        "command": command,
    }
    result = dict(
        original,
        dataset="MKG-Y",
        teacher_strategy="best_conf",
        kd_loss_type="selection_ndkd",
        policy_optimizer="same",
        max_train_examples=0,
        eval_queries=-1,
        best_epoch=3,
        best_mrr=0.38,
        final_result=dict(metrics, MRR=0.37),
        test=dict(metrics, MRR=0.99),
    )
    (checkpoint_dir / "result.json").write_text(json.dumps(result))
    if has_checkpoint:
        (checkpoint_dir / "student-test.pth").write_bytes(b"CPU fixture; not loaded")
    plan = {
        "training_mode": "full",
        "baselines": {"MKG-Y": {"result": str(original_path)}},
        "jobs": [job],
        "input_sha256": {str(original_path): module.digest(original_path)},
        "limitations": [],
    }
    (out / "plan.json").write_text(json.dumps(plan))
    assert module.summarize(out) == has_checkpoint
    summary = json.loads((out / "summary.json").read_text())
    assert not summary[
        "full_benchmark_grid_completed"
    ]  # This fixture only covers one cell.
    if has_checkpoint:
        assert summary["rows"][1]["test"]["MRR"] == 0.37
        assert summary["status"] == "PASS"
    else:
        assert summary["status"] == "INCOMPLETE" and len(summary["failures"]) == 1
    # An altered original result must be detected before any result can be summarized.
    original_path.write_text("changed")
    with pytest.raises(ValueError, match="Input or original result changed"):
        module.summarize(out)
