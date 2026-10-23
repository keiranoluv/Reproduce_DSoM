"""CPU-only source/mapping/feature preparation checks with tiny synthetic data."""

import copy
import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_wn9_from_mose as prep  # noqa: E402


def test_split_audit_distinguishes_exact_and_inverse_overlap():
    a, b, c = "n00000001", "n00000002", "n00000003"
    splits = {
        "train": [(a, "_hypernym", b), (b, "_part_of", c)],
        "valid": [(b, "_hyponym", a)],
        "test": [(a, "_hypernym", b), (c, "_part_of", b)],
    }
    before = copy.deepcopy(splits)
    _, _, report = prep.split_audit(splits)
    assert report["overlaps"]["train/valid"]["exact"] == 0
    assert report["overlaps"]["train/valid"]["named_inverse_relation"] == 1
    assert report["overlaps"]["train/test"]["exact"] == 1
    assert report["overlaps"]["train/test"]["same_relation_reverse"] == 1
    assert report["overlaps"]["train/test"]["named_inverse_relation"] == 0
    assert report["cold_entities_vs_train"] == 0
    assert splits == before


def test_non_synset_identifiers_rejected():
    with pytest.raises(ValueError, match="synset"):
        prep.split_audit({s: [("0", "_hypernym", "1")] for s in prep.SPLITS})


def test_text_normalization_and_duplicate_rejection(tmp_path):
    path = tmp_path / "text"
    path.write_text("00000001\tdog\\n animal  \n")
    assert prep.texts(path) == {"00000001": "dog  animal"}
    path.write_text("00000001\tdog\n00000001\tcat\n")
    with pytest.raises(ValueError, match="Duplicate"):
        prep.texts(path)


def test_mapping_requires_contiguous_unique_ids(tmp_path):
    path = tmp_path / "map"
    path.write_text("00000001\t0\n00000002\t0\n")
    with pytest.raises(ValueError, match="contiguous"):
        prep.mapping(path)


@pytest.fixture
def tiny_source(tmp_path):
    source = tmp_path / "source"
    raw = source / "src_data/WN9"
    raw.mkdir(parents=True)
    a, b = "n00000001", "n00000002"
    splits = {
        "train": [(a, "_hypernym", b), (b, "_hyponym", a)],
        "valid": [(a, "_hyponym", b)],
        "test": [(b, "_hypernym", a)],
    }
    for split, triples in splits.items():
        (raw / split).write_text("".join("\t".join(row) + "\n" for row in triples))
    (raw / "ent_dscp.txt").write_text("00000001\tfirst\n00000002\tsecond\n")
    (raw / "ent_name.txt").write_text(a + "\tfirst\n" + b + "\tsecond\n")
    parent = tmp_path / "parent"
    parent.mkdir()
    # Deliberately permuted IDs; selecting rows by position must fail the test.
    (parent / "ent_id").write_text("00000001\t2\n00000002\t0\n00000003\t1\n")
    for name, dim in (("img_vec.pickle", 4), ("dscp_vec.pickle", 3)):
        with (parent / name).open("wb") as stream:
            pickle.dump(np.arange(3 * dim, dtype=np.float32).reshape(3, dim), stream)
    inputs = {str(p): prep.sha(p) for folder in (raw, parent) for p in folder.iterdir()}
    rows = [
        dict(wn9_id=i, wn9_entity=e, wn18_entity=e[1:], wn18_id=idx, image_source="")
        for i, (e, idx) in enumerate(((a, 2), (b, 0)))
    ]
    report = dict(input_sha256=inputs, feature_reuse={"parent": str(parent)})
    return source, parent, splits, rows, report


def prepare_fixture(tmp_path, tiny_source):
    source, parent, splits, rows, report = tiny_source
    target, raw_target = tmp_path / "data/WN9", tmp_path / "raw/WN9"
    result = prep.prepare(report, splits, rows, source, parent, target, raw_target)
    return target, raw_target, result


def test_preparation_preserves_raw_triples_and_gathers_correct_feature_rows(
    tmp_path, tiny_source
):
    target, raw, report = prepare_fixture(tmp_path, tiny_source)
    parent = tiny_source[1]
    for name in ("img_vec.pickle", "dscp_vec.pickle"):
        np.testing.assert_array_equal(
            prep.load_pickle(target / name), prep.load_pickle(parent / name)[[2, 0]]
        )
    for split in prep.SPLITS:
        assert (raw / split).read_bytes() == (
            tiny_source[0] / "src_data/WN9" / split
        ).read_bytes()
    assert prep.mapping(target / "ent_id") == {"n00000001": 0, "n00000002": 1}
    assert prep.verify_prepared(target) == report
    prep.verify_hashes(tiny_source[-1]["input_sha256"])


def test_preparation_matches_existing_preprocessor_without_changing_global(
    tmp_path, tiny_source
):
    import process_datasets

    original_path = process_datasets.DATA_PATH
    target, _, _ = prepare_fixture(tmp_path, tiny_source)
    assert process_datasets.DATA_PATH == original_path
    filters = prep.load_pickle(target / "to_skip.pickle")
    assert filters["rhs"][0, 0] == [1]
    assert filters["rhs"][0, 1] == [1]
    assert filters["lhs"][1, 2] == [0]
    assert filters["lhs"][1, 3] == [0]


def test_prepare_refuses_existing_destinations(tmp_path, tiny_source):
    source, parent, splits, rows, report = tiny_source
    target, raw = tmp_path / "data/WN9", tmp_path / "raw/WN9"
    raw.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already exists"):
        prep.prepare(report, splits, rows, source, parent, target, raw)
    assert not target.exists()


def test_verify_detects_changed_parent_feature(tmp_path, tiny_source):
    target, _, _ = prepare_fixture(tmp_path, tiny_source)
    (tiny_source[1] / "img_vec.pickle").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed file"):
        prep.verify_prepared(target)


def test_verify_reconstructs_filters_even_if_manifest_hash_is_rewritten(
    tmp_path, tiny_source
):
    target, _, _ = prepare_fixture(tmp_path, tiny_source)
    filters_path = target / "to_skip.pickle"
    with filters_path.open("wb") as stream:
        pickle.dump({"lhs": {}, "rhs": {}}, stream)
    manifest_path = target / "preparation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"][str(filters_path)] = prep.sha(filters_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="filters differ"):
        prep.verify_prepared(target)


def test_default_audit_does_not_prepare_or_import_torch(monkeypatch, tmp_path):
    module = importlib.reload(prep)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["prepare_wn9_from_mose.py"])
    report = {"structure": {}, "feature_reuse": {}}
    monkeypatch.setattr(module, "audit", lambda *args: (report, {}, []))
    monkeypatch.setattr(
        module, "prepare", lambda *args: pytest.fail("Audit must not prepare")
    )
    before = "torch" in sys.modules
    module.main()
    assert ("torch" in sys.modules) == before
