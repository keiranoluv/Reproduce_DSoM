"""CPU checks for fresh-checkout setup, explicit references, and delegated CLIs."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import build_features as feature_cli  # noqa: E402
import prepare_dataset as dataset_cli  # noqa: E402
import run_reproduction as fresh  # noqa: E402
import run_table2  # noqa: E402
import run_table4_components  # noqa: E402
import run_table6  # noqa: E402
from test_table4_components import a5  # noqa: E402
from test_table7 import training_parser  # noqa: E402


@pytest.mark.parametrize("pre_staged", [False, True])
def test_structural_prepare_and_verify_with_permuted_entities(
    tmp_path, monkeypatch, pre_staged
):
    monkeypatch.setattr(dataset_cli, "ROOT", tmp_path)
    monkeypatch.setitem(dataset_cli.EXPECTED, "toy", (2, 1, (1, 1, 1)))
    source = tmp_path / "raw source"
    source.mkdir()
    for split, row in [("train", "b\tr\ta"), ("valid", "a\tr\tb"), ("test", "a\tr\ta")]:
        (source / (split + ".txt")).write_text(row + "\n")
    if pre_staged:
        raw = tmp_path / "src_data/toy"
        raw.mkdir(parents=True)
        for split in ("train", "valid", "test"):
            (raw / split).write_bytes(
                (source / (split + ".txt")).read_bytes().replace(b"\n", b"\r\n")
            )
        before = {
            split: (raw / split).read_bytes() for split in ("train", "valid", "test")
        }
    monkeypatch.setattr(
        sys,
        "argv",
        ["prepare_dataset.py", "--dataset", "toy", "--source-dir", str(source)],
    )
    dataset_cli.main()
    if pre_staged:
        assert all(
            (raw / split).read_bytes() == content for split, content in before.items()
        )
    report = dataset_cli.verify("toy", tmp_path / "src_data/toy", tmp_path / "data/toy")
    assert report["entities"] == 2
    assert (tmp_path / "data/toy/ent_id").read_text() == "a\t0\nb\t1\n"
    with pytest.raises(FileExistsError):
        dataset_cli.main()


@pytest.mark.parametrize("profile", fresh.PROFILES)
@pytest.mark.parametrize("mode", ["smoke", "full"])
def test_portable_student_profiles_parse_on_actual_cli(
    profile, mode, tmp_path, monkeypatch
):
    monkeypatch.setattr(fresh, "git", lambda *args: "commit")
    command, cfg = fresh.command(
        "student", "MMKB-DB15K", profile, mode, tmp_path, "/teacher.pth"
    )
    assert command[0] == sys.executable
    parsed = vars(training_parser("student").parse_args(command[3:]))
    assert all(parsed[k] == v for k, v in cfg.items())
    if profile != "baseline":
        baseline = a5()
        for key in (*run_table2.MATCHED, *run_table4_components.EXTRA):
            if key in ("kd_loss_type", "teacher_strategy", "policy_optimizer"):
                continue
            if key in ("teacher_checkpoint", "img_info", "dscp_info"):
                continue
            if mode == "smoke" and key in ("max_epochs", "valid", "early_stopping"):
                continue
            # Fixed average has no policy learning; this difference is intentional.
            if key == "policy_lr" and profile == "table4-average":
                continue
            assert parsed[key] == baseline[key], key


def test_reference_registry_resolves_repository_relative_paths(tmp_path):
    registry = tmp_path / "references.json"
    registry.write_text(json.dumps({"baselines": {"MKG-Y": "runs/new/result.json"}}))
    assert (
        run_table2.reference_path(registry, "baselines", "MKG-Y", tmp_path)
        == tmp_path / "runs/new/result.json"
    )
    with pytest.raises(ValueError, match="Missing baselines"):
        run_table2.reference_path(registry, "baselines", "MKG-W", tmp_path)


@pytest.mark.parametrize("dataset", list(fresh.EXPECTED))
@pytest.mark.parametrize("stage", ["teacher", "student"])
def test_all_supported_baseline_dataset_commands_parse(
    dataset, stage, tmp_path, monkeypatch
):
    monkeypatch.setattr(fresh, "git", lambda *args: "commit")
    cmd, cfg = fresh.command(
        stage, dataset, "baseline", "full", tmp_path, "/teacher.pth"
    )
    parsed = vars(training_parser(stage).parse_args(cmd[3:]))
    assert parsed["dataset"] == dataset
    assert all(parsed[key] == value for key, value in cfg.items())


def test_registry_table6_does_not_require_historical_db15k_runs(tmp_path, monkeypatch):
    registry = tmp_path / "references.json"
    registry.write_text(json.dumps({"table6": {}}))
    monkeypatch.setattr(run_table6, "ROOT", tmp_path)
    assert (
        run_table6.existing_reference("MMKB-DB15K", "structural", None, {}, registry)
        is None
    )


def test_empty_registry_slot_is_checked_before_training(tmp_path, monkeypatch):
    registry = tmp_path / "references.json"
    registry.write_text(json.dumps({"baselines": {"MKG-Y": "runs/old/result.json"}}))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_reproduction.py",
            "--dataset",
            "MKG-Y",
            "--mode",
            "full",
            "--registry",
            str(registry),
        ],
    )
    monkeypatch.setattr(
        fresh, "data_inputs", lambda *a: pytest.fail("Must reject before data/training")
    )
    with pytest.raises(SystemExit):
        fresh.main()


def test_default_baseline_plan_never_launches_training(tmp_path, monkeypatch):
    monkeypatch.setattr(fresh, "ROOT", tmp_path)
    monkeypatch.setattr(fresh, "git", lambda *args: "commit")
    monkeypatch.setattr(fresh, "data_inputs", lambda dataset: {})
    monkeypatch.setattr(
        fresh.subprocess, "run", lambda *a, **k: pytest.fail("Plan must not launch")
    )
    monkeypatch.setattr(sys, "argv", ["run_reproduction.py", "--dataset", "MKG-Y"])
    fresh.main()
    plan = json.loads(
        next((tmp_path / "runs/reproduction").rglob("plan.json")).read_text()
    )
    assert set(plan["jobs"]) == {"teacher", "student"}


def test_feature_adapter_uses_explicit_paths_and_existing_backend(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(feature_cli, "ROOT", tmp_path)
    monkeypatch.setitem(feature_cli.EXPECTED, "FB15K-237", (2, 1, (1, 1, 1)))
    data = tmp_path / "data/FB15K-237"
    data.mkdir(parents=True)
    (data / "ent_id").write_text("/m/a\t0\n/m/b\t1\n")
    text = tmp_path / "descriptions.txt"
    text.write_text("/m/a\ttext\n")
    backend = SimpleNamespace()

    def main():
        assert backend.TEXT_FILE == text
        assert backend.ENT_ID_FILE == data / "ent_id"
        assert backend.OUT_FILE == data / "dscp_vec.pickle"
        assert backend.BATCH_SIZE == 2
        backend.OUT_FILE.write_bytes(b"CPU adapter fixture, not model output")

    backend.main = main
    monkeypatch.setattr(feature_cli, "module_for", lambda *args: backend)
    # Fake implementation file only for provenance hashing after delegated extraction.
    code = tmp_path / "src/build_text_features.py"
    code.parent.mkdir()
    code.write_text("CPU fixture")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_features.py",
            "--dataset",
            "FB15K-237",
            "--modality",
            "text",
            "--input",
            str(text),
            "--mode",
            "extract",
            "--device",
            "cpu",
            "--batch-size",
            "2",
        ],
    )
    feature_cli.main()
    assert (data / "text_portable_manifest.json").exists()


def test_feature_audit_never_imports_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(feature_cli, "ROOT", tmp_path)
    monkeypatch.setitem(feature_cli.EXPECTED, "WN18", (2, 1, (1, 1, 1)))
    data = tmp_path / "data/WN18"
    data.mkdir(parents=True)
    (data / "ent_id").write_text("00000001\t0\n00000002\t1\n")
    text = tmp_path / "text.txt"
    text.write_text("00000001\tfirst\n")
    monkeypatch.setattr(
        feature_cli, "module_for", lambda *a: pytest.fail("No encoder import in audit")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_features.py",
            "--dataset",
            "WN18",
            "--modality",
            "text",
            "--input",
            str(text),
        ],
    )
    feature_cli.main()


def test_fresh_full_profiles_register_compatible_table_references(
    tmp_path, monkeypatch
):
    from test_table7 import mock_training

    monkeypatch.setattr(fresh, "ROOT", tmp_path)
    monkeypatch.setattr(fresh, "SOURCES", ())
    monkeypatch.setattr(
        fresh, "git", lambda *args: "commit" if args[0] == "rev-parse" else ""
    )
    marker = tmp_path / "data-marker"
    marker.write_bytes(b"unchanged fixture")
    monkeypatch.setattr(
        fresh, "data_inputs", lambda *a: {str(marker): run_table2.digest(marker)}
    )
    for name in ("img_vec.pickle", "dscp_vec.pickle"):
        path = tmp_path / "data/MMKB-DB15K" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    def run(cmd, cwd, env, stdout, stderr):
        mock_training(Path(stdout.name).parent)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(fresh.subprocess, "run", run)
    registry = tmp_path / "registry.json"
    base = [
        "run_reproduction.py",
        "--dataset",
        "MMKB-DB15K",
        "--mode",
        "full",
        "--registry",
        str(registry),
    ]
    monkeypatch.setattr(sys, "argv", base)
    fresh.main()
    path, baseline = run_table2.baseline(
        "MMKB-DB15K", root=tmp_path, references=registry
    )
    assert path.is_file()
    teacher = Path(baseline["teacher_checkpoint"]).parents[2]
    for profile in fresh.LABELS:
        monkeypatch.setattr(
            sys,
            "argv",
            base
            + [
                "--stage",
                "student",
                "--teacher-run",
                str(teacher),
                "--profile",
                profile,
            ],
        )
        fresh.main()
    monkeypatch.setattr(run_table4_components, "ROOT", tmp_path)
    references = run_table4_components.load_references(registry)
    assert len(references) == 4
    assert (
        len({ref["result"]["teacher_checkpoint"] for ref in references.values()}) == 1
    )


def test_generated_interpreter_comes_from_active_environment(tmp_path, monkeypatch):
    # No attempt is made to execute this fake path; commands must retain it verbatim.
    interpreter = "/opt/a different environment/bin/python"
    monkeypatch.setattr(sys, "executable", interpreter)
    original = a5()
    command, _ = run_table4_components.build_command(
        original, "nekd", "full", tmp_path, "commit"
    )
    assert command[0] == interpreter
    command, _ = run_table6.build_command(
        original, "visual", "full", tmp_path, "commit"
    )
    assert command[0] == interpreter


def test_baseline_preflight_rejects_invalid_split_before_gpu(tmp_path, monkeypatch):
    import pickle

    import numpy as np

    monkeypatch.setattr(fresh, "ROOT", tmp_path)
    monkeypatch.setitem(fresh.EXPECTED, "toy", (2, 1, (1, 1, 1)))
    folder = tmp_path / "data/toy"
    folder.mkdir(parents=True)
    (folder / "ent_id").write_text("a\t0\nb\t1\n")
    (folder / "rel_id").write_text("r\t0\n")
    for name, value in {
        "train.pickle": np.array([[0, 0, 1]], dtype=np.uint64),
        "valid.pickle": np.array([[1, 0, 0]], dtype=np.uint64),
        "test.pickle": np.array([[0, 0, 1]], dtype=np.uint64),
        "img_vec.pickle": np.zeros((2, 1000), dtype=np.float32),
        "dscp_vec.pickle": np.zeros((2, 768), dtype=np.float32),
        "to_skip.pickle": {},
        "probas.pickle": {},
    }.items():
        with (folder / name).open("wb") as stream:
            pickle.dump(value, stream)
    assert len(fresh.data_inputs("toy")) == 9
    with (folder / "test.pickle").open("wb") as stream:
        pickle.dump(np.array([[0, 0, 2]], dtype=np.uint64), stream)
    with pytest.raises(ValueError, match="Invalid processed split"):
        fresh.data_inputs("toy")
