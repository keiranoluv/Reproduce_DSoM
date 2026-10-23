"""CPU mathematical audit and launcher checks for A5-matched component runs."""

import argparse
import ast
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import datasets as dataset_module  # noqa: E402
from kd_optimizers import dkd_loss, multi_dkd_loss  # noqa: E402


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("GPU initialization forbidden in these tests")

    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


def runner():
    spec = importlib.util.spec_from_file_location(
        "table4_runner_test", ROOT / "scripts/run_table4_components.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def a5():
    return dict(
        dataset="MMKB-DB15K",
        model="ComplExMDR",
        teacher_checkpoint="/fixture/teacher.pth",
        img_info="/fixture/image.pickle",
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
        kd_loss_type="kdrl",
        teacher_strategy="rc",
        policy_optimizer="Adam",
        policy_lr=0.00001,
        rl_hs=1024,
        rl_reward=1.0,
        rl_penalize=10.0,
        rl_weight=1.0,
        policy_temp=1.0,
        policy_input_norm=False,
        rc_reward_mode="binary",
        l5_counterfactual=False,
        modal_missing=0.0,
        modality_split=True,
        fusion_img=True,
        fusion_dscp=True,
        max_train_examples=0,
        eval_queries=-1,
    )


def logits():
    student = torch.tensor(
        [[0.2, -0.4, 1.1, -0.8, 0.6], [0.1, 0.9, -0.3, 0.4, -0.2]],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher = torch.tensor(
        [[1.2, 0.7, -0.1, 0.3, -0.4], [-0.7, 0.1, 1.4, -0.8, 0.2]], dtype=torch.float64
    )
    return student, teacher, torch.tensor([0, 2]), [[0, 1], [2]]


def independent_terms(
    student, teacher, target, neighbors, temperature, paper_mean=False
):
    student, teacher = (x / temperature for x in (student, teacher))
    ps, pt = student.softmax(1), teacher.softmax(1)
    first, second = [], []
    for i, ids in enumerate(neighbors):
        ids = sorted(set(ids) | {int(target[i])})
        others = [e for e in range(ps.shape[1]) if e not in ids]
        count = len(ids) if paper_mean else 1
        sn, tn = ps[i, ids].sum() / count, pt[i, ids].sum() / count
        sb, tb = torch.stack([sn, 1 - sn]), torch.stack([tn, 1 - tn])
        first.append((tb * (tb.log() - sb.log())).sum())
        # Gather only non-neighbor logits before softmax, independently of the
        # released implementation's -1000 mask and its grouped-probability code.
        sl, tl = student[i, others].log_softmax(0), teacher[i, others].log_softmax(0)
        second.append((tl.exp() * (tl - sl)).sum())
    return tuple(torch.stack(v).mean() * temperature**2 for v in (first, second))


@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
def test_component_values_and_gradients_match_independent_kl(alpha, beta):
    s, t, y, n = logits()
    actual = multi_dkd_loss(s, t, y, alpha, beta, 4.0, n)
    ne, nn = independent_terms(s, t, y, n, 4.0)
    expected = alpha * ne + beta * nn
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    ga = torch.autograd.grad(actual, s, retain_graph=True)[0]
    ge = torch.autograd.grad(expected, s)[0]
    torch.testing.assert_close(ga, ge, atol=1e-12, rtol=1e-10)
    assert torch.isfinite(ga).all()


def test_components_reconstruct_default_ndkd_loss_and_gradient():
    s, t, y, n = logits()
    ne = multi_dkd_loss(s, t, y, 1.0, 0.0, 4.0, n)
    nn = multi_dkd_loss(s, t, y, 0.0, 1.0, 4.0, n)
    both = multi_dkd_loss(s, t, y, 1.0, 1.0, 4.0, n)
    torch.testing.assert_close(ne + nn, both, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(
        torch.autograd.grad(ne + nn, s, retain_graph=True)[0],
        torch.autograd.grad(both, s)[0],
        atol=1e-12,
        rtol=1e-10,
    )


def test_nnekd_is_conditional_distribution_and_excludes_neighbor_gradients():
    s, t, y, n = logits()
    loss = multi_dkd_loss(s, t, y, 0.0, 1.0, 4.0, n)
    gradient = torch.autograd.grad(loss, s)[0]
    for i, ids in enumerate(n):
        torch.testing.assert_close(
            gradient[i, ids], torch.zeros_like(gradient[i, ids]), atol=1e-12, rtol=0
        )
    # Shift every non-neighbor logit by the same constant. Conditional
    # distributions match although neighbor-versus-other probability mass differs.
    other = t.clone()
    for i, ids in enumerate(n):
        other[i, [j for j in range(t.shape[1]) if j not in ids]] += 2.0
    torch.testing.assert_close(
        multi_dkd_loss(other, t, y, 0.0, 1.0, 4.0, n),
        torch.tensor(0.0, dtype=t.dtype),
        atol=1e-12,
        rtol=0,
    )
    assert multi_dkd_loss(other, t, y, 1.0, 0.0, 4.0, n) > 0


def test_neighbor_term_is_group_mass_not_within_neighbor_distribution():
    p = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float64)
    q = torch.tensor([[0.2, 0.1, 0.3, 0.4]], dtype=torch.float64)
    loss = multi_dkd_loss(
        4 * p.log(), 4 * q.log(), torch.tensor([0]), 1.0, 0.0, 4.0, [[0, 1]]
    )
    torch.testing.assert_close(
        loss, torch.tensor(0.0, dtype=p.dtype), atol=1e-12, rtol=0
    )


def test_released_sum_differs_from_paper_mean_with_multiple_neighbors():
    s, t, y, n = logits()
    code = multi_dkd_loss(s, t, y, 1.0, 0.0, 4.0, n)
    paper, _ = independent_terms(s, t, y, n, 4.0, paper_mean=True)
    assert abs(float((code - paper).detach())) > 1e-6


@pytest.mark.parametrize("alpha,beta", [(1.0, 0.0), (0.0, 1.0), (1.0, 1.0)])
def test_single_neighbor_reduces_to_original_dkd(alpha, beta):
    s, t, y, _ = logits()
    n = [[int(v)] for v in y]
    torch.testing.assert_close(
        multi_dkd_loss(s, t, y, alpha, beta, 4.0, n),
        dkd_loss(s, t, y, alpha, beta, 4.0),
        atol=1e-12,
        rtol=1e-10,
    )


def test_actual_dataset_neighbors_use_train_and_reciprocals_only(tmp_path, monkeypatch):
    data = tmp_path / "toy"
    data.mkdir()
    arrays = {
        "train": np.array([[0, 0, 1], [0, 0, 2], [1, 1, 2]], dtype=np.uint64),
        "valid": np.array([[0, 0, 0]], dtype=np.uint64),
        "test": np.array([[1, 1, 1]], dtype=np.uint64),
    }
    for split, array in arrays.items():
        with (data / f"{split}.pickle").open("wb") as f:
            pickle.dump(array, f)
    filters = {"rhs": {(0, 0): [0, 1, 2], (1, 1): [1, 2]}, "lhs": {}}
    with (data / "to_skip.pickle").open("wb") as f:
        pickle.dump(filters, f)
    monkeypatch.setattr(dataset_module, "DATA_PATH", tmp_path)
    dataset = dataset_module.Dataset("toy")
    assert dataset.train_to_skip[(0, 0)] == [1, 2]
    assert dataset.train_to_skip[(1, 1)] == [2]
    assert dataset.train_to_skip[(1, 2)] == [0]
    assert dataset.train_to_skip[(2, 2)] == [0]
    assert dataset.train_to_skip[(2, 3)] == [1]


def argument_parser():
    tree = ast.parse((ROOT / "src/kd_learn.py").read_text())
    names = {
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
            isinstance(n, ast.Name) and n.id in names for n in node.targets
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
        compile(
            ast.Module(body=body, type_ignores=[]), "CPU argument definitions", "exec"
        ),
        env,
    )
    return env["parser"]


@pytest.mark.parametrize("component", ["nekd", "nnkd"])
@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_commands_change_only_component_and_smoke_limits(component, mode, tmp_path):
    module = runner()
    original = a5()
    module.validate_a5(original)
    command, cfg = module.build_command(original, component, mode, tmp_path, "commit")
    parsed = vars(argument_parser().parse_args(command[3:]))
    for key, value in cfg.items():
        assert parsed[key] == value, key
    assert (parsed["dkd_alpha"], parsed["dkd_beta"]) == module.COMPONENTS[component]
    for key, value in original.items():
        if key in ("dkd_alpha", "dkd_beta") or (
            mode == "smoke"
            and key
            in (
                "max_epochs",
                "valid",
                "early_stopping",
                "max_train_examples",
                "eval_queries",
            )
        ):
            continue
        assert parsed[key] == value, key
    assert "--policy_input_norm" not in command and "--l5_counterfactual" not in command
    assert parsed["rc_distill_loss"] == "ndkd" and parsed["policy_optimizer"] == "Adam"


def test_true_boolean_switches_are_serialized_without_value(tmp_path):
    original = a5()
    original.update(policy_input_norm=True, l5_counterfactual=True)
    command, cfg = runner().build_command(original, "nekd", "full", tmp_path, "commit")
    parsed = argument_parser().parse_args(command[3:])
    assert parsed.policy_input_norm and parsed.l5_counterfactual


def test_source_frozen_baseline_is_rejected():
    original = a5()
    original["policy_optimizer"] = "source_frozen"
    with pytest.raises(ValueError, match="policy_optimizer"):
        runner().validate_a5(original)


def test_plan_never_starts_training(tmp_path, monkeypatch, capsys):
    module = runner()
    module.ROOT = tmp_path
    module.SOURCES = ()
    original = a5()
    reference = tmp_path / "a5.json"
    reference.write_text("{}")
    for key in ("teacher_checkpoint", "img_info", "dscp_info"):
        p = tmp_path / key
        p.write_bytes(b"CPU fixture")
        original[key] = str(p)
    data = tmp_path / "data/MMKB-DB15K"
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
    monkeypatch.setattr(
        module,
        "load_references",
        lambda: {
            "A5_RC_NDKD": {
                "result": original,
                "path": str(reference),
                "paper": [0.4268, 0.3445, 0.4662, 0.5812],
            }
        },
    )
    monkeypatch.setattr(
        module, "git", lambda *args: "commit" if args[0] == "rev-parse" else ""
    )
    monkeypatch.setattr(module.subprocess, "check_output", lambda *args, **kwargs: "")

    def fail(*args, **kwargs):
        raise AssertionError("Plan must never start training")

    monkeypatch.setattr(module.subprocess, "run", fail)
    out = tmp_path / "runs/table4_components/plan/test"
    monkeypatch.setattr(sys, "argv", ["run_table4_components.py", "--output", str(out)])
    module.main()
    plan = json.loads((out / "plan.json").read_text())
    assert len(plan["jobs"]) == 2 and plan["mode"] == "plan"
    assert "PLAN ONLY" in capsys.readouterr().out


def test_full_mode_requires_committed_code(monkeypatch):
    module = runner()
    monkeypatch.setattr(
        module, "git", lambda *args: " M src/kd.py" if args[0] == "status" else "commit"
    )
    monkeypatch.setattr(sys, "argv", ["run_table4_components.py", "--mode", "full"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
