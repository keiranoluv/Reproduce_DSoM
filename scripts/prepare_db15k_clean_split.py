#!/usr/bin/env python3
"""Trace DB15K provenance and prepare an isolated, reproducible clean holdout.

Only reads the original benchmark. Existing output directories are never replaced.
Run --verify-only to validate the saved hashes and protocol after training.
"""

import argparse
import hashlib
import json
import pickle
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "data/MMKB-DB15K"
NAME = "MMKB-DB15K-CleanV1"
OUTPUT = ROOT / "data" / NAME
RAW_OUTPUT = ROOT / "src_data" / NAME
REPORT = ROOT / "docs/audits/db15k_source_trace.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def dump_pickle(path, obj):
    with path.open("wb") as stream:
        pickle.dump(obj, stream, protocol=4)


def mapping(path):
    return {
        key: int(value)
        for key, value in (
            line.rsplit("\t", 1) for line in path.read_text().splitlines()
        )
    }


def triples(array):
    return [tuple(map(int, row)) for row in array]


def group_key(row):
    h, r, t = row
    return min(h, t), r, max(h, t)


def split_stats(arrays):
    sets = {s: set(triples(a)) for s, a in arrays.items()}
    groups = {s: {group_key(row) for row in rows} for s, rows in sets.items()}
    return {
        "splits": {
            s: {
                "rows": len(arrays[s]),
                "unique": len(rows),
                "duplicates": len(arrays[s]) - len(rows),
                "entities": len({e for h, r, t in rows for e in (h, t)}),
                "relations": len({r for h, r, t in rows}),
            }
            for s, rows in sets.items()
        },
        "exact_overlaps": {
            f"{a}_{b}": len(sets[a] & sets[b])
            for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]
        },
        "same_relation_undirected_group_overlaps": {
            f"{a}_{b}": len(groups[a] & groups[b])
            for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]
        },
        "validation_exclusive_unique": len(
            sets["valid"] - sets["train"] - sets["test"]
        ),
        "test_entities_absent_from_train": len(
            {e for h, r, t in sets["test"] for e in (h, t)}
            - {e for h, r, t in sets["train"] for e in (h, t)}
        ),
    }


def make_holdout(train, test, size, seed):
    """Hold out whole same-relation reversal groups, retaining train coverage."""
    test_groups = {group_key(row) for row in triples(test)}
    groups = defaultdict(list)
    excluded = []
    for row in triples(train):
        if group_key(row) in test_groups:
            excluded.append(row)
        else:
            groups[group_key(row)].append(row)
    if not 0 < size < sum(map(len, groups.values())):
        raise ValueError("Validation size must leave a nonempty training split")
    entity_count = Counter(
        e for rows in groups.values() for h, r, t in rows for e in (h, t)
    )
    relation_count = Counter(r for rows in groups.values() for h, r, t in rows)
    total = sum(relation_count.values())
    quotas = {r: size * n // total for r, n in relation_count.items()}
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    keys = [keys[int(i)] for i in rng.permutation(len(keys))]
    selected, selected_rel = set(), Counter()
    selected_n = 0

    # Approximate relation stratification first; fill the remainder in a second pass.
    for use_quota in (True, False):
        for key in keys:
            rows = groups[key]
            n = len(rows)
            if key in selected or selected_n + n > size:
                continue
            r = key[1]
            if use_quota and selected_rel[r] + n > quotas[r]:
                continue
            ec = Counter(e for h, r, t in rows for e in (h, t))
            if relation_count[r] <= n or any(
                entity_count[e] <= c for e, c in ec.items()
            ):
                continue
            selected.add(key)
            selected_n += n
            selected_rel[r] += n
            entity_count.subtract(ec)
            relation_count[r] -= n
            if selected_n == size:
                break
    if selected_n != size:
        raise ValueError(
            f"Could only hold out {selected_n}/{size} triples with coverage constraints"
        )
    kept, valid = [], []
    for row in triples(train):
        key = group_key(row)
        if key not in test_groups:
            (valid if key in selected else kept).append(row)
    return (
        np.asarray(kept, dtype=train.dtype),
        np.asarray(valid, dtype=train.dtype),
        np.asarray(excluded, dtype=train.dtype).reshape(-1, 3),
    )


def verify():
    manifest = json.loads((OUTPUT / "split_manifest.json").read_text())
    for filename, expected in manifest["preserved_original_sha256"].items():
        path = Path(filename)
        if not path.is_absolute():
            path = ROOT / path
        if sha256(path) != expected:
            raise ValueError(f"Original artifact changed: {path}")
    for filename, expected in manifest["created_sha256"].items():
        if sha256(ROOT / filename) != expected:
            raise ValueError(f"Clean artifact changed: {filename}")
    arrays = {
        s: load_pickle(OUTPUT / f"{s}.pickle") for s in ("train", "valid", "test")
    }
    stats = split_stats(arrays)
    assert stats == manifest["clean_statistics"]
    assert not any(stats["exact_overlaps"].values())
    assert not any(stats["same_relation_undirected_group_overlaps"].values())
    assert all(v["duplicates"] == 0 for v in stats["splits"].values())
    assert sha256(OUTPUT / "test.pickle") == sha256(ORIGINAL / "test.pickle")
    rebuilt_train, rebuilt_valid, rebuilt_excluded = make_holdout(
        load_pickle(ORIGINAL / "train.pickle"),
        arrays["test"],
        manifest["validation_size"],
        manifest["split_seed"],
    )
    assert np.array_equal(arrays["train"], rebuilt_train)
    assert np.array_equal(arrays["valid"], rebuilt_valid)
    assert np.array_equal(
        load_pickle(OUTPUT / "excluded_test_reversals.pickle"), rebuilt_excluded
    )
    assert len(rebuilt_train) + len(rebuilt_valid) + len(rebuilt_excluded) == len(
        load_pickle(ORIGINAL / "train.pickle")
    )
    # Filtering may use all clean splits; training masks are built from train only by Dataset.
    n_rel = len(mapping(OUTPUT / "rel_id"))
    expected = {"lhs": defaultdict(set), "rhs": defaultdict(set)}
    for array in arrays.values():
        for h, r, t in triples(array):
            expected["rhs"][(h, r)].add(t)
            expected["lhs"][(t, r + n_rel)].add(h)
    expected = {k: {q: sorted(v) for q, v in d.items()} for k, d in expected.items()}
    assert load_pickle(OUTPUT / "to_skip.pickle") == expected
    print(
        "PASS: source artifacts preserved; clean split disjoint; official test unchanged; filters correct"
    )
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT.parent / "dataset_audit/MMRNS/MMKB-DB15K"
    )
    parser.add_argument("--upstream-record", type=Path)
    parser.add_argument("--valid-size", type=int, default=9904)
    parser.add_argument("--split-seed", type=int, default=2022)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
        return
    if OUTPUT.exists() or RAW_OUTPUT.exists():
        raise FileExistsError("Existing clean dataset; use --verify-only")
    if args.upstream_record is None:
        parser.error(
            "--upstream-record is required to establish fresh upstream provenance"
        )
    upstream = json.loads(args.upstream_record.read_text())
    entities, relations = mapping(ORIGINAL / "ent_id"), mapping(ORIGINAL / "rel_id")
    arrays, trace = {}, {}
    for split in ("train", "valid", "test"):
        source = args.source / f"{split}.txt"
        local = ROOT / "src_data/MMKB-DB15K" / split
        lines = [tuple(line.split("\t")) for line in source.read_text().splitlines()]
        encoded = np.asarray(
            [(entities[h], relations[r], entities[t]) for h, r, t in lines],
            dtype=np.uint64,
        )
        arrays[split] = load_pickle(ORIGINAL / f"{split}.pickle")
        assert source.read_bytes() == local.read_bytes(), split
        assert np.array_equal(encoded, arrays[split]), split
        assert sha256(source) == upstream["files"][split]["sha256"], split
        # Validate the fresh copies themselves, not only the recorded digests.
        assert sha256(args.upstream_record.parent / f"{split}.txt") == sha256(source)
        trace[split] = {
            "downloaded_source": str(source.resolve()),
            "sha256": sha256(source),
            "raw_copy_byte_equal": True,
            "processed_row_order_and_ids_equal": True,
            "fresh_public_download_byte_equal": True,
        }
    original_stats = split_stats(arrays)
    train, valid, excluded = make_holdout(
        arrays["train"], arrays["test"], args.valid_size, args.split_seed
    )
    clean = {"train": train, "valid": valid, "test": arrays["test"]}
    stats = split_stats(clean)
    assert not any(stats["same_relation_undirected_group_overlaps"].values())
    # Dataset currently infers dimensions from train maxima: guard that contract.
    assert max(train[:, 0].max(), train[:, 2].max()) + 1 == len(entities)
    assert train[:, 1].max() + 1 == len(relations)

    preserved = {}
    paths = list(ORIGINAL.glob("*")) + list((ROOT / "src_data/MMKB-DB15K").glob("*"))
    paths += list(args.source.glob("*.txt"))
    for stage in ("teacher", "student"):
        paths += list((ROOT / "runs" / stage / "MMKB-DB15K").rglob("result.json"))
        # Preserve hashes of the original Exp01 baseline checkpoints as well as metrics.
        paths += list((ROOT / "runs" / stage / "MMKB-DB15K/full").rglob("*.pth"))
    for path in sorted(set(p for p in paths if p.is_file())):
        key = (
            str(path.relative_to(ROOT))
            if path.is_relative_to(ROOT)
            else str(path.resolve())
        )
        preserved[key] = sha256(path)

    OUTPUT.mkdir(parents=True)
    RAW_OUTPUT.mkdir(parents=True)
    for name in ("ent_id", "rel_id", "img_vec.pickle", "dscp_vec.pickle"):
        shutil.copyfile(ORIGINAL / name, OUTPUT / name)
    for split, array in clean.items():
        if split == "test":
            shutil.copyfile(ORIGINAL / "test.pickle", OUTPUT / "test.pickle")
            shutil.copyfile(ROOT / "src_data/MMKB-DB15K/test", RAW_OUTPUT / "test")
        else:
            dump_pickle(OUTPUT / f"{split}.pickle", array)
            inv_e, inv_r = (
                {v: k for k, v in entities.items()},
                {v: k for k, v in relations.items()},
            )
            (RAW_OUTPUT / split).write_text(
                "".join(
                    f"{inv_e[h]}\t{inv_r[r]}\t{inv_e[t]}\n"
                    for h, r, t in triples(array)
                )
            )
    dump_pickle(OUTPUT / "excluded_test_reversals.pickle", excluded)
    filters = {"lhs": defaultdict(set), "rhs": defaultdict(set)}
    for array in clean.values():
        for h, r, t in triples(array):
            filters["rhs"][(h, r)].add(t)
            filters["lhs"][(t, r + len(relations))].add(h)
    dump_pickle(
        OUTPUT / "to_skip.pickle",
        {k: {q: sorted(v) for q, v in d.items()} for k, d in filters.items()},
    )
    counts = {
        "lhs": np.bincount(train[:, 0].astype(np.int64), minlength=len(entities)),
        "rhs": np.bincount(train[:, 2].astype(np.int64), minlength=len(entities)),
    }
    counts["both"] = counts["lhs"] + counts["rhs"]
    dump_pickle(OUTPUT / "probas.pickle", {k: v / v.sum() for k, v in counts.items()})
    created = {
        str(p.relative_to(ROOT)): sha256(p)
        for folder in (OUTPUT, RAW_OUTPUT)
        for p in sorted(folder.iterdir())
    }
    manifest = {
        "dataset": NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "official-test-preserved; original-validation-discarded; test-reversal-groups-excluded; relation-stratified-train-holdout",
        "split_seed": args.split_seed,
        "validation_size": args.valid_size,
        "n_entities": len(entities),
        "n_original_relations": len(relations),
        "excluded_train_test_reversals": len(excluded),
        "coverage_rule": "Holdout retains every entity and relation present in eligible training pool",
        "limitations": [
            "Does not remove semantic inverse relations with different relation IDs",
            "Retains official test cold entities",
            "Single seed; reduced training size; not a direct paper benchmark result",
        ],
        "original_statistics": original_stats,
        "clean_statistics": stats,
        "raw_source_trace": trace,
        "upstream": upstream,
        "preprocessing_script_sha256": sha256(ROOT / "src/process_datasets.py"),
        "builder_sha256": sha256(Path(__file__)),
        "relation_counts": {
            s: dict(sorted(Counter(int(r) for r in a[:, 1]).items()))
            for s, a in clean.items()
        },
        "preserved_original_sha256": preserved,
        "created_sha256": created,
    }
    (OUTPUT / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(
            {
                "conclusion": "Overlap originates in the currently published MMRNS raw splits, not local preprocessing",
                "raw_source_trace": trace,
                "upstream": upstream,
                "original_statistics": original_stats,
                "clean_statistics": stats,
                "clean_dataset": NAME,
                "split_seed": args.split_seed,
                "excluded_train_test_reversals": len(excluded),
                "manifest": str((OUTPUT / "split_manifest.json").relative_to(ROOT)),
            },
            indent=2,
        )
        + "\n"
    )
    verify()


if __name__ == "__main__":
    main()
