#!/usr/bin/env python3
"""Audit WN9 and optionally prepare it using existing frozen WN18 features.

NumPy/CPU only: no Torch, encoders, teacher checkpoints, or training imports.
Default audit writes evidence only; prepare refuses existing output directories.
"""

import argparse
import hashlib
import json
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "192d9ff2882b5be6ef8239a37cbd90c5dd102b5a"
RAW_FILES = ("train", "valid", "test", "ent_dscp.txt", "ent_name.txt")
SPLITS = ("train", "valid", "test")
INVERSE = {
    "_hypernym": "_hyponym",
    "_hyponym": "_hypernym",
    "_has_part": "_part_of",
    "_part_of": "_has_part",
    "_member_holonym": "_member_meronym",
    "_member_meronym": "_member_holonym",
    "_member_of_domain_topic": "_synset_domain_topic_of",
    "_synset_domain_topic_of": "_member_of_domain_topic",
    "_derivationally_related_form": "_derivationally_related_form",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pickle(path):
    # Inputs are locally generated features or our own prepared data, not web pickles.
    with path.open("rb") as stream:
        return pickle.load(stream)


def texts(path):
    result = {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        key, value = line.split("\t", 1)
        if key in result:
            raise ValueError(f"Duplicate key {key} in {path}:{number}")
        result[key] = value.replace("\\n", " ").strip()
    return result


def mapping(path):
    result = {key: int(value) for key, value in texts(path).items()}
    if sorted(result.values()) != list(range(len(result))):
        raise ValueError(f"Mapping IDs must be contiguous: {path}")
    return result


def read_triples(path):
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = tuple(line.split("\t"))
        if len(parts) != 3:
            raise ValueError(f"Malformed triple {path}:{number}")
        rows.append(parts)
    return rows


def split_audit(splits):
    entities = sorted(
        {e for rows in splits.values() for h, _, t in rows for e in (h, t)}
    )
    relations = sorted({r for rows in splits.values() for _, r, _ in rows})
    if any(not re.fullmatch(r"n\d{8}", e) for e in entities):
        raise ValueError(
            "Expected noun synset IDs n########; do not infer IDs by row order"
        )
    train_entities = {e for h, _, t in splits["train"] for e in (h, t)}
    overlaps = {}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        left, right = set(splits[a]), set(splits[b])
        inverse = {(t, INVERSE[r], h) for h, r, t in left if r in INVERSE}
        found = inverse & right
        overlaps[a + "/" + b] = {
            "exact": len(left & right),
            "same_relation_reverse": len({(t, r, h) for h, r, t in left} & right),
            "named_inverse_relation": len(found),
            "named_inverse_by_target_relation": dict(
                sorted(Counter(r for _, r, _ in found).items())
            ),
        }
    return (
        entities,
        relations,
        dict(
            entities=len(entities),
            relations=len(relations),
            relation_names=relations,
            splits={
                s: {"rows": len(v), "unique": len(set(v))} for s, v in splits.items()
            },
            cold_entities_vs_train=len(set(entities) - train_entities),
            overlaps=overlaps,
            inverse_relation_definition=INVERSE,
        ),
    )


def audit(source_repo, parent, kg_text):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(source_repo), *args], text=True
        ).strip()

    if git("rev-parse", "HEAD") != SOURCE_COMMIT:
        raise ValueError(f"Use the audited MoSE source commit {SOURCE_COMMIT}")
    raw = source_repo / "src_data/WN9"
    for name in RAW_FILES:
        committed = subprocess.check_output(
            ["git", "-C", str(source_repo), "show", f"HEAD:src_data/WN9/{name}"]
        )
        if (raw / name).read_bytes() != committed:
            raise ValueError(f"Source differs from committed MoSE file: {name}")
    splits = {s: read_triples(raw / s) for s in SPLITS}
    entities, relations, structural = split_audit(splits)
    expected = {"train": 11741, "valid": 1337, "test": 1319}
    if (
        len(entities) != 6555
        or len(relations) != 9
        or any(
            len(splits[s]) != expected[s] or len(set(splits[s])) != expected[s]
            for s in SPLITS
        )
    ):
        raise ValueError("WN9 source statistics differ from DSoM Table 5")
    if structural["cold_entities_vs_train"]:
        raise ValueError("Unexpected entities absent from WN9 training")
    ent_map, rel_map = mapping(parent / "ent_id"), mapping(parent / "rel_id")
    if any(e[1:] not in ent_map for e in entities):
        raise ValueError("Some WN9 synsets are absent from WN18")
    if any(r not in rel_map for r in relations):
        raise ValueError("Some WN9 relation names are absent from WN18")
    source_text, parent_text = texts(raw / "ent_dscp.txt"), texts(kg_text)
    if any(
        not source_text.get(e[1:]) or source_text[e[1:]] != parent_text.get(e[1:])
        for e in entities
    ):
        raise ValueError(
            "WN9 descriptions do not match the text used for WN18 feature extraction"
        )
    image_manifest = json.loads((parent / "img_features_manifest.json").read_text())
    image_sources = image_manifest["sources_by_internal_id"]
    if len(image_sources) != len(ent_map):
        raise ValueError("Image manifest and entity mapping differ in size")
    indices = np.array([ent_map[e[1:]] for e in entities], dtype=np.int64)
    rows = []
    for idx, (entity, parent_id) in enumerate(zip(entities, indices.tolist())):
        path = image_sources[parent_id]
        if path and Path(path).parts[1] != entity:
            raise ValueError(f"Image manifest synset mismatch: {entity}: {path}")
        rows.append(
            dict(
                wn9_id=idx,
                wn9_entity=entity,
                wn18_entity=entity[1:],
                wn18_id=parent_id,
                image_source=path,
            )
        )
    dimensions = {"img_vec.pickle": 1000, "dscp_vec.pickle": 768}
    for name, dim in dimensions.items():
        features = load_pickle(parent / name)
        if not isinstance(features, np.ndarray) or features.shape != (
            len(ent_map),
            dim,
        ):
            raise ValueError(f"Unexpected parent feature shape: {name}")
        if features.dtype != np.float32 or not np.isfinite(features[indices]).all():
            raise ValueError(f"Invalid selected feature values: {name}")
        del features
    # Describe graph provenance too; graph-trained WN18 models are never reused.
    id_to_entity = {v: k for k, v in ent_map.items()}
    id_to_relation = {v: k for k, v in rel_map.items()}
    parent_splits = {}
    for s in SPLITS:
        parent_splits[s] = {
            (id_to_entity[int(h)], id_to_relation[int(r)], id_to_entity[int(t)])
            for h, r, t in load_pickle(parent / f"{s}.pickle")
        }
    membership = {}
    for s, triples in splits.items():
        normalized = {(h[1:], r, t[1:]) for h, r, t in triples}
        membership[s] = {
            p: len(normalized & values) for p, values in parent_splits.items()
        }
        membership[s]["not_in_parent_union"] = len(
            normalized - set.union(*parent_splits.values())
        )
    paths = [raw / name for name in RAW_FILES] + [kg_text]
    paths += [
        parent / name
        for name in (
            "ent_id",
            "rel_id",
            "img_vec.pickle",
            "dscp_vec.pickle",
            "img_features_manifest.json",
            "train.pickle",
            "valid.pickle",
            "test.pickle",
            "to_skip.pickle",
            "probas.pickle",
        )
    ]
    paths += [
        ROOT / "scripts" / name
        for name in ("build_wn18_text_features.py", "build_wn18_image_features.py")
    ]
    report = dict(
        schema_version=1,
        status="PASS",
        audited_utc=datetime.now(timezone.utc).isoformat(),
        source=dict(
            repository="https://github.com/OreOZhao/MoSE4MKGC",
            commit=SOURCE_COMMIT,
            folder="src_data/WN9",
            local_repository=str(source_repo),
        ),
        references=dict(
            ikrl="https://github.com/thunlp/IKRL",
            dsom="https://arxiv.org/html/2507.20738v1#A3",
            raw_vs_filtered_baseline="https://openreview.net/pdf?id=Gf94NMabeh",
        ),
        structure=structural,
        wn18_graph_membership=membership,
        feature_reuse=dict(
            parent=str(parent),
            mapping="strip exactly one leading n, then lookup ent_id",
            matched_entities=len(entities),
            identical_nonempty_descriptions=len(entities),
            real_images=sum(bool(row["image_source"]) for row in rows),
            inherited_imputed_images=sum(not row["image_source"] for row in rows),
            inherited_imputed_entities=[
                row["wn9_entity"] for row in rows if not row["image_source"]
            ],
            image_encoder=image_manifest["encoder"],
            image_preprocessing=image_manifest["preprocessing"],
            text_encoder="bert-base-uncased; frozen WN18 feature rows, no re-encoding",
            dimensions=dimensions,
        ),
        numpy_version=np.__version__,
        preparation_code_sha256={
            name: sha(ROOT / name)
            for name in ("scripts/prepare_wn9_from_mose.py", "src/process_datasets.py")
        },
        input_sha256={str(path.resolve()): sha(path) for path in paths},
        limitations=[
            "MoSE source snapshot verified; IKRL archive not independently compared",
            "Existing reconstructed WN18 frozen-encoder features, not author-released DSoM features",
            "Eight missing-image vectors inherit WN18 seed/order, not a fresh WN9 RNG draw",
            "Named inverse facts cross splits; benchmark left unchanged, not a clean split",
            "WN9 graph-trained teacher/student checkpoints must be trained fresh",
            "Table 7 raw versus filtered evaluation protocol still requires reconciliation",
        ],
    )
    return report, splits, rows


def verify_hashes(hashes):
    for name, expected in hashes.items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Changed file: {name}")


def verify_prepared(target):
    manifest = json.loads((target / "preparation_manifest.json").read_text())
    verify_hashes(manifest["input_sha256"])
    verify_hashes(manifest["output_sha256"])
    raw = Path(manifest["raw_output"])
    entities, relations, _ = split_audit({s: read_triples(raw / s) for s in SPLITS})
    expected_e = {entity: i for i, entity in enumerate(entities)}
    expected_r = {relation: i for i, relation in enumerate(relations)}
    if (
        mapping(target / "ent_id") != expected_e
        or mapping(target / "rel_id") != expected_r
    ):
        raise ValueError("Prepared mappings differ from sorted source identifiers")
    filters = {"lhs": defaultdict(set), "rhs": defaultdict(set)}
    for s in SPLITS:
        expected = np.array(
            [
                (expected_e[h], expected_r[r], expected_e[t])
                for h, r, t in read_triples(raw / s)
            ],
            dtype=np.uint64,
        )
        np.testing.assert_array_equal(load_pickle(target / f"{s}.pickle"), expected)
        for h, r, t in expected.tolist():
            filters["rhs"][h, r].add(t)
            filters["lhs"][t, r + len(relations)].add(h)
    expected_filters = {
        side: {k: sorted(v) for k, v in values.items()}
        for side, values in filters.items()
    }
    if load_pickle(target / "to_skip.pickle") != expected_filters:
        raise ValueError("Prepared filters differ from the exact split union")
    train = load_pickle(target / "train.pickle").astype(np.int64)
    lhs = np.bincount(train[:, 0], minlength=len(entities)).astype(float)
    rhs = np.bincount(train[:, 2], minlength=len(entities)).astype(float)
    for key, counts in (("lhs", lhs), ("rhs", rhs), ("both", lhs + rhs)):
        np.testing.assert_array_equal(
            load_pickle(target / "probas.pickle")[key], counts / counts.sum()
        )
    parent = Path(manifest["feature_reuse"]["parent"])
    parent_map = mapping(parent / "ent_id")
    indices = [parent_map[e[1:]] for e in entities]
    for name in ("img_vec.pickle", "dscp_vec.pickle"):
        np.testing.assert_array_equal(
            load_pickle(target / name), load_pickle(parent / name)[indices]
        )
    return manifest


def prepare(report, splits, rows, source, parent, target, raw_target):
    if target.exists() or raw_target.exists():
        raise FileExistsError(
            "WN9 output already exists; use --mode verify, do not overwrite"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    raw_target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".wn9_prepare_", dir=target.parent) as tmp:
        staging = Path(tmp)
        staged_raw = staging / "raw"
        staged_raw.mkdir()
        for name in RAW_FILES:
            shutil.copyfile(source / "src_data/WN9" / name, staged_raw / name)
        sys.path.insert(0, str(ROOT / "src"))
        import process_datasets

        old_data_path = process_datasets.DATA_PATH
        try:
            process_datasets.DATA_PATH = staging / "data"
            process_datasets.prepare_dataset(str(staged_raw), "WN9")
        finally:
            process_datasets.DATA_PATH = old_data_path
        staged_target = staging / "data/WN9"
        indices = [row["wn18_id"] for row in rows]
        for name in ("img_vec.pickle", "dscp_vec.pickle"):
            selected = load_pickle(parent / name)[indices].copy()
            with (staged_target / name).open("wb") as stream:
                pickle.dump(selected, stream)
        fields = ("wn9_id", "wn9_entity", "wn18_entity", "wn18_id", "image_source")
        (staged_target / "wn18_feature_row_map.tsv").write_text(
            "\t".join(fields)
            + "\n"
            + "".join("\t".join(str(row[key]) for key in fields) + "\n" for row in rows)
        )
        verify_hashes(report["input_sha256"])
        # Both destinations are new. No benchmark or WN18 file is overwritten.
        staged_raw.rename(raw_target)
        staged_target.rename(target)
    result = dict(report, raw_output=str(raw_target), processed_output=str(target))
    result["output_sha256"] = {
        str(path): sha(path)
        for directory in (target, raw_target)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }
    (target / "preparation_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    verify_prepared(target)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("audit", "prepare", "verify"), default="audit")
    p.add_argument(
        "--source-repo", type=Path, default=ROOT.parent / "dataset_audit/MoSE4MKGC_WN9"
    )
    p.add_argument("--parent-features", type=Path, default=ROOT / "data/WN18")
    p.add_argument(
        "--parent-text",
        type=Path,
        default=ROOT.parent / "kg-bert/data/WN18/entity2text.txt",
    )
    args = p.parse_args()
    target, raw_target = ROOT / "data/WN9", ROOT / "src_data/WN9"
    if args.mode == "verify":
        verify_prepared(target)
        print(
            "PASS: WN9 hashes, mappings, split triples, filters, and exact inherited feature rows"
        )
        return
    report, splits, rows = audit(
        args.source_repo.resolve(),
        args.parent_features.resolve(),
        args.parent_text.resolve(),
    )
    if args.mode == "prepare":
        report = prepare(
            report,
            splits,
            rows,
            args.source_repo.resolve(),
            args.parent_features.resolve(),
            target,
            raw_target,
        )
    output = (
        ROOT
        / "runs/audit"
        / datetime.now(timezone.utc).strftime("wn9_sources_%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS:", output / "report.json")
    print(
        json.dumps(
            {"structure": report["structure"], "features": report["feature_reuse"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
