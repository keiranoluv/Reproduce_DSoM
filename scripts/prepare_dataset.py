#!/usr/bin/env python3
"""Portable structural preparation/verification. Never download or overwrite data."""

import argparse
import json
import pickle
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from run_table2 import digest

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "MMKB-DB15K": (12842, 279, (79222, 9904, 9902)),
    "MKG-W": (15000, 169, (34196, 4276, 4274)),
    "MKG-Y": (15000, 28, (21310, 2665, 2663)),
    "FB15K-237": (14541, 237, (272115, 17535, 20466)),
    "WN18": (40943, 18, (141442, 5000, 5000)),
}


def read(path):
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        parts = tuple(line.split("\t"))
        if len(parts) != 3 or not all(parts):
            raise ValueError(f"Invalid triple {path}:{number}")
        rows.append(parts)
    return rows


def verify(dataset, raw, target):
    counts = EXPECTED[dataset]
    splits = {s: read(raw / s) for s in ("train", "valid", "test")}
    entities = sorted(
        {e for rows in splits.values() for h, _, t in rows for e in (h, t)}
    )
    relations = sorted({r for rows in splits.values() for _, r, _ in rows})
    if (
        len(entities),
        len(relations),
        tuple(len(v) for v in splits.values()),
    ) != counts:
        raise ValueError(
            f"Unexpected {dataset} statistics; do not swap/deduplicate splits to force a match"
        )
    e = {name: i for i, name in enumerate(entities)}
    r = {name: i for i, name in enumerate(relations)}

    def load(path):
        with path.open("rb") as f:
            return pickle.load(f)

    for filename, expected in (("ent_id", e), ("rel_id", r)):
        actual = {
            k: int(v)
            for k, v in (
                row.split("\t") for row in (target / filename).read_text().splitlines()
            )
        }
        if actual != expected:
            raise ValueError(f"Mapping mismatch: {filename}")
    filters = {"rhs": defaultdict(set), "lhs": defaultdict(set)}
    for split, rows in splits.items():
        mapped = np.array([(e[h], r[rel], e[t]) for h, rel, t in rows], dtype=np.uint64)
        np.testing.assert_array_equal(load(target / f"{split}.pickle"), mapped)
        for h, rel, t in mapped.tolist():
            filters["rhs"][h, rel].add(t)
            filters["lhs"][t, rel + len(r)].add(h)
    actual_filters = load(target / "to_skip.pickle")
    expected_filters = {
        side: {key: sorted(values) for key, values in mapping.items()}
        for side, mapping in filters.items()
    }
    if actual_filters != expected_filters:
        raise ValueError("Filter mismatch")
    train = np.array([(e[h], r[rel], e[t]) for h, rel, t in splits["train"]])
    if (int(train[:, [0, 2]].max()) + 1, int(train[:, 1].max()) + 1) != counts[:2]:
        raise ValueError(
            "Dataset train-max shape inference differs from complete mapping"
        )
    return {
        "dataset": dataset,
        "entities": len(e),
        "relations": len(r),
        "splits": {s: len(v) for s, v in splits.items()},
        "duplicates": {s: len(v) - len(set(v)) for s, v in splits.items()},
        "exact_overlap": {
            a + "/" + b: len(set(splits[a]) & set(splits[b]))
            for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=EXPECTED, required=True)
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--train-file", default="train.txt")
    p.add_argument("--valid-file", default="valid.txt")
    p.add_argument("--test-file", default="test.txt")
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    raw, target = ROOT / "src_data" / args.dataset, ROOT / "data" / args.dataset
    if args.verify_only:
        print(json.dumps(verify(args.dataset, raw, target), indent=2))
        return
    if args.source_dir is None:
        p.error("--source-dir is required for preparation")
    if target.exists():
        raise FileExistsError("Processed dataset already exists; use --verify-only")
    source = {
        s: args.source_dir / getattr(args, s + "_file")
        for s in ("train", "valid", "test")
    }
    for path in source.values():
        read(path)
    if raw.exists():
        for split, path in source.items():
            if read(raw / split) != read(path):
                raise ValueError(
                    f"Existing staged {split} differs from source; no files overwritten"
                )
        if args.dataset == "MKG-W" and digest(raw / "ent_links") != digest(
            args.source_dir / "ent_links"
        ):
            raise ValueError("Existing ent_links differs from source")
    target.parent.mkdir(parents=True, exist_ok=True)
    raw.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".prepare_") as temp:
        staging = Path(temp)
        staged_raw = staging / "raw"
        staged_raw.mkdir()
        for split, path in source.items():
            shutil.copyfile(path, staged_raw / split)
        if args.dataset == "MKG-W":
            shutil.copyfile(args.source_dir / "ent_links", staged_raw / "ent_links")
        sys.path.insert(0, str(ROOT / "src"))
        import process_datasets

        previous = process_datasets.DATA_PATH
        try:
            process_datasets.DATA_PATH = staging / "processed"
            process_datasets.prepare_dataset(str(staged_raw), args.dataset)
        finally:
            process_datasets.DATA_PATH = previous
        staged_target = staging / "processed" / args.dataset
        report = verify(args.dataset, staged_raw, staged_target)
        report["raw_source_sha256"] = {
            str(path.resolve()): digest(path) for path in source.values()
        }
        report["staged_raw_sha256"] = {
            split: digest((raw if raw.exists() else staged_raw) / split)
            for split in source
        }
        (staged_target / "structural_manifest.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if not raw.exists():
            staged_raw.rename(raw)
        staged_target.rename(target)
    print("PASS:", target)


if __name__ == "__main__":
    main()
