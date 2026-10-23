"""Small CPU mathematical/integration checks, without loading real checkpoints."""

import copy
import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_table4_ensembles as launcher  # noqa: E402

from table4_ensembles import (  # noqa: E402
    MetaLearner,
    ValidationRC,
    average_metrics,
    evaluate,
    filtered_ranks,
    fit_epoch,
)


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("CUDA forbidden in CPU tests")

    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.sizes = (5, 4, 5)
        self.weight = nn.Parameter(torch.arange(75.0).reshape(5, 5, 3) / 13)
        self.seen = []

    def forward(self, query):
        self.seen.extend(query.tolist())
        values = self.weight[query[:, 0]]
        return values[..., 0], (), values[..., 1], (), values[..., 2], ()


def fixture():
    teacher = Teacher().requires_grad_(False).eval()
    queries = {
        "rhs": torch.tensor([[0, 0, 1], [1, 1, 2], [2, 0, 3]]),
        "lhs": torch.tensor([[1, 2, 0], [2, 3, 1], [3, 2, 2]]),
    }
    filters = {
        side: {(h, r): [t] for h, r, t in rows.tolist()}
        for side, rows in queries.items()
    }
    return teacher, queries, filters


@pytest.mark.parametrize("id_type", [int, np.uint64, np.uint32, np.int64])
def test_filtered_pessimistic_ties_and_no_filter_mutation(id_type):
    scores = torch.tensor([[9.0, 4.0, 4.0, 8.0, 3.0]])
    queries = torch.tensor([[0, 0, 1]])
    filters = {(id_type(0), id_type(0)): [id_type(v) for v in (0, 1, 3)]}
    original = copy.deepcopy(filters)
    assert filtered_ranks(scores, queries, filters).item() == 2
    assert filters == original
    assert scores[0, 0] == 9


def test_mean_metric_differs_from_ranking_mean_logits():
    q = torch.tensor([[0, 0, 0]])
    logits = torch.tensor([[[3.0, 0.0, 0.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]])
    ranks = [filtered_ranks(logits[..., i], q, {}) for i in range(3)]
    mean_mrr = sum(1 / r.item() for r in ranks) / 3
    fused_mrr = 1 / filtered_ranks(logits.mean(-1), q, {}).item()
    assert mean_mrr != fused_mrr


def test_mose_architecture_preserves_single_batch_and_entity_axes():
    model = MetaLearner()
    x = torch.randn(1, 5, 3)
    reference = F.relu(
        F.linear(
            F.relu(F.linear(x, model.fc1.weight, model.fc1.bias)),
            model.fc2.weight,
            model.fc2.bias,
        )
    ).squeeze(-1)
    assert model(x).shape == (1, 5)
    torch.testing.assert_close(model(x), reference)
    assert model.fc1.weight.shape == (32, 3)


def test_rc_actions_are_selected_without_labels():
    model = ValidationRC(5, hidden=8).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.fc2.bias[4] = 20
    x = torch.randn(2, 5, 3)
    torch.testing.assert_close(model(x), (x[..., 0] + x[..., 2]) / 2)


def test_rc_binary_advantage_reference(monkeypatch):
    model = ValidationRC(5, hidden=8)
    x = torch.zeros(2, 5, 3)
    x[0, 0, 0] = 9  # Structural beats average for first target.
    x[1, 0, 1] = 9  # Structural loses to average for second target.
    targets = torch.zeros(2, dtype=torch.long)
    monkeypatch.setattr(
        torch.distributions.Categorical,
        "sample",
        lambda self: torch.zeros(2, dtype=torch.long),
    )
    model.eval()
    expected = -11 * F.log_softmax(model.policy_logits(x), dim=-1)[0, 0] / 2
    loss = model.reinforce_loss(x, targets)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert model.fc2.weight.grad.abs().sum() > 0


def test_evaluation_matches_independent_ranks_and_is_target_agnostic():
    teacher, queries, filters = fixture()
    model = MetaLearner()
    result = evaluate(teacher, {"ml": model}, queries, filters, "cpu", 2)
    expected = []
    for side, values in queries.items():
        out = teacher(values)
        scores = (out[0] + out[2] + out[4]) / 3
        for i, (h, r, t) in enumerate(values.tolist()):
            excluded = set(filters[side][h, r]) | {t}
            expected.append(
                1
                + sum(
                    scores[i, e] >= scores[i, t] for e in range(5) if e not in excluded
                )
            )
    assert result["teacher_mean_logits"]["MRR"] == pytest.approx(
        torch.tensor(expected).double().reciprocal().mean().item()
    )
    assert result["teacher_mean_metrics"] == average_metrics(
        [result[name] for name in ("structural", "visual", "description")]
    )
    assert not model.training


def test_fit_updates_head_only_and_uses_only_supplied_validation_queries():
    torch.manual_seed(2022)
    teacher, queries, _ = fixture()
    before = copy.deepcopy(teacher.state_dict())
    model = MetaLearner()
    # Positive final bias ensures this fixture exercises CE gradients.
    with torch.no_grad():
        model.fc2.bias.fill_(2)
    original = copy.deepcopy(model.state_dict())
    report = fit_epoch(
        teacher,
        model,
        torch.optim.Adagrad(model.parameters(), lr=0.1),
        queries,
        "cpu",
        2,
    )
    assert report["batches_with_nonzero_gradient"] > 0
    assert any(not torch.equal(original[k], v) for k, v in model.state_dict().items())
    assert all(torch.equal(before[k], v) for k, v in teacher.state_dict().items())
    assert all(p.grad is None for p in teacher.parameters())
    assert sorted(teacher.seen) == sorted(
        sum((q.tolist() for q in queries.values()), [])
    )


@pytest.mark.parametrize("method", ["ml", "rc"])
def test_saved_head_reload_in_cpu_training_integration(tmp_path, method):
    teacher, queries, _ = fixture()
    # Avoid equal modality offsets: allow genuinely better mixed actions for RC.
    with torch.no_grad():
        teacher.weight.copy_(torch.randn_like(teacher.weight))
    cfg = dict(
        seed=2022,
        ml_hidden=8,
        rc_hidden=8,
        ml_lr=0.1,
        rc_lr=0.001,
        ml_epochs=2,
        rc_epochs=2,
        batch_size=2,
    )
    head = launcher.train_head(teacher, method, cfg, queries, "cpu", tmp_path)
    saved = torch.load(tmp_path / "final_head.pth", weights_only=True)
    assert saved["epoch"] == 2
    assert saved["method"] == method
    assert all(
        torch.equal(v.cpu(), saved["model"][k]) for k, v in head.state_dict().items()
    )
    assert len(json.loads((tmp_path / "history.json").read_text())) == 2


def test_query_loader_retains_duplicates_and_reciprocal_offsets(tmp_path, monkeypatch):
    folder = tmp_path / "data/toy"
    folder.mkdir(parents=True)
    values = np.array([[0, 0, 2], [0, 0, 2], [1, 1, 0]])
    for split in ("train", "valid", "test"):
        with (folder / f"{split}.pickle").open("wb") as stream:
            pickle.dump(values, stream)
    with (folder / "to_skip.pickle").open("wb") as stream:
        pickle.dump({"rhs": {}, "lhs": {}}, stream)
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    shape, queries, _ = launcher.load_queries(
        dict(dataset="toy", queries_per_direction=0, seed=2022)
    )
    assert shape == (3, 4, 3)
    assert queries["valid"]["rhs"].tolist() == values.tolist()
    assert queries["valid"]["lhs"].tolist() == [[2, 2, 0], [2, 2, 0], [0, 3, 1]]


def test_default_plan_does_not_execute(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_table4_ensembles.py"])
    monkeypatch.setattr(launcher, "prepare", lambda args: (Path("/fixture"), {}))
    monkeypatch.setattr(
        launcher, "execute", lambda *args: pytest.fail("Must not execute")
    )
    launcher.main()


def test_full_requires_commit_before_any_work(monkeypatch):
    monkeypatch.setattr(launcher, "git", lambda *args: " M src/models.py")
    with pytest.raises(ValueError, match="clean tracked"):
        launcher.prepare(launcher.parser().parse_args(["--mode", "full"]))


def test_summary_includes_adaptation_flags_and_all_student_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "verify_inputs", lambda plan: None)
    suite = tmp_path / "student"
    suite.mkdir()
    values = dict(MR=2.0, MRR=0.5, **{"hits@[1,3,10]": [0.3, 0.8, 1.0]})
    rows = [dict(method=f"student{i}", test=values, paper=[0.4]) for i in range(6)]
    (suite / "summary.json").write_text(
        json.dumps(dict(rows=rows, component_pair_completed=True))
    )
    plan = dict(
        config={"methods": ["avg", "ml", "rc"]},
        mode="full",
        student_suite=str(suite),
        limitations=["adapted"],
    )
    results = {
        method: {
            "test": {
                key: values
                for key in ("teacher_mean_metrics", "teacher_mean_logits", "ml", "rc")
            }
        }
        for method in ("avg", "ml", "rc")
    }
    launcher.write_summary(tmp_path, plan, results)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["all_nine_rows_have_results"]
    assert not summary["exact_author_protocol_reproduced"]
    assert len(summary["student_rows"]) == 6
    assert len(summary["teacher_rows"]) == 4  # includes mean-logit diagnostic


def test_module_import_does_not_initialize_cuda():
    importlib.reload(launcher)
    assert not torch.cuda.is_initialized()
