#!/usr/bin/env python3
"""Audit WN18 structural preprocessing for DSoM."""

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "src_data" / "WN18"
DATA = ROOT / "data" / "WN18"

EXPECTED = {
    "train": 141442,
    "valid": 5000,
    "test": 5000,
}
N_ENTITIES = 40943
N_RELATIONS = 18


def load_mapping(filename):
    mapping = {}

    for line in (DATA / filename).read_text(encoding="utf-8").splitlines():
        name, index = line.split("\t")
        assert name not in mapping
        mapping[name] = int(index)

    assert sorted(mapping.values()) == list(range(len(mapping)))
    assert list(mapping) == sorted(mapping)

    return mapping


def load_pickle(filename):
    with open(DATA / filename, "rb") as f:
        return pickle.load(f)


def main():
    assert DATA.is_dir(), f"Missing {DATA}"

    ent2id = load_mapping("ent_id")
    rel2id = load_mapping("rel_id")

    assert len(ent2id) == N_ENTITIES
    assert len(rel2id) == N_RELATIONS

    print("===== VOCABULARY =====")
    print("Entities :", len(ent2id))
    print("Relations:", len(rel2id))

    all_triples = []

    for split, expected_count in EXPECTED.items():
        raw_triples = []

        for line in (RAW / split).read_text(encoding="utf-8").splitlines():
            h, r, t = line.split("\t")
            raw_triples.append([ent2id[h], rel2id[r], ent2id[t]])

        processed = np.asarray(load_pickle(f"{split}.pickle"))

        assert processed.shape == (expected_count, 3)
        assert len(raw_triples) == expected_count
        assert np.issubdtype(processed.dtype, np.integer)

        expected_array = np.asarray(raw_triples, dtype=np.int64)

        assert np.array_equal(processed, expected_array), (
            f"{split}: raw/processed mismatch"
        )

        assert np.all(processed[:, 0] < N_ENTITIES)
        assert np.all(processed[:, 2] < N_ENTITIES)
        assert np.all(processed[:, 1] < N_RELATIONS)

        all_triples.extend(map(tuple, processed.tolist()))

        print(f"{split:5s}: {len(processed):6d} | dtype={processed.dtype} | PASS")

    print("\n===== FILTERED EVALUATION =====")

    expected_filters = {
        "lhs": defaultdict(set),
        "rhs": defaultdict(set),
    }

    for h, r, t in all_triples:
        expected_filters["rhs"][(h, r)].add(t)
        expected_filters["lhs"][(t, r + N_RELATIONS)].add(h)

    filters = load_pickle("to_skip.pickle")
    assert set(filters) == {"lhs", "rhs"}

    for direction in ("lhs", "rhs"):
        actual = {
            tuple(map(int, key)): set(map(int, values))
            for key, values in filters[direction].items()
        }

        expected = expected_filters[direction]

        assert actual == expected, f"{direction}: filtering mismatch"

        print(f"{direction}: {len(actual)} filter keys | PASS")

    print("\n===== TRAIN PROBABILITIES =====")

    train = np.asarray(load_pickle("train.pickle"))
    probs = load_pickle("probas.pickle")

    assert set(probs) == {"lhs", "rhs", "both"}

    for name in ("lhs", "rhs", "both"):
        values = np.asarray(probs[name])

        assert values.shape == (N_ENTITIES,)
        assert np.isfinite(values).all()
        assert np.all(values >= 0)
        assert np.isclose(values.sum(), 1.0)

        expected_counts = np.zeros(N_ENTITIES, dtype=np.float64)

        if name in ("lhs", "both"):
            np.add.at(expected_counts, train[:, 0], 1)

        if name in ("rhs", "both"):
            np.add.at(expected_counts, train[:, 2], 1)

        expected_probs = expected_counts / expected_counts.sum()

        assert np.allclose(values, expected_probs)

        print(f"{name}: shape={values.shape} | PASS")

    print("\n================================")
    print("WN18 STRUCTURAL AUDIT: PASS")


if __name__ == "__main__":
    main()
