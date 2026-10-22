#!/usr/bin/env python3
"""Post-hoc A5 selection statistics; no GPU/model loading."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_DIR = "runs/audit/db15k_a5_selection/20261021_200624_3249374"


def load_pairs(path):
    with np.load(path, allow_pickle=False) as z:
        q = z["queries"]
        ce = z["ce"].astype(np.float64)
        actions = z["A5_NDKD"]

    n = len(q)
    assert n % 2 == 0
    assert ce.shape == (n, 7)
    assert actions.shape == (n,)
    assert np.isfinite(ce).all()
    assert np.all((actions >= 0) & (actions < 7))

    half = n // 2

    # Verify forward/inverse query alignment.
    assert np.array_equal(q[:half, 0], q[half:, 2])
    assert np.array_equal(q[:half, 2], q[half:, 0])

    offsets = q[half:, 1] - q[:half, 1]
    assert np.all(offsets == offsets[0])
    assert offsets[0] > 0

    # Deduplicate original triples, retaining both directions.
    unique, inverse, counts = np.unique(
        q[:half],
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    m = len(unique)
    delta = np.zeros((m, 2), dtype=np.float64)
    selected = np.zeros((m, 2), dtype=np.float64)
    full_delta = np.zeros((m, 2), dtype=np.float64)

    for direction in range(2):
        sl = slice(direction * half, (direction + 1) * half)

        d = ce[sl, 4] - ce[sl, 0]
        a = actions[sl]
        s = (a == 4).astype(np.float64)

        f = ce[sl][np.arange(half), a] - ce[sl, 0]

        delta[:, direction] = np.bincount(inverse, weights=d, minlength=m) / counts
        selected[:, direction] = np.bincount(inverse, weights=s, minlength=m) / counts
        full_delta[:, direction] = np.bincount(inverse, weights=f, minlength=m) / counts

    return unique, delta, selected, full_delta, half


def compute_metrics(delta, selected, full_delta):
    wins = delta < 0
    n_selected = selected.sum()

    actual = (selected * delta).mean()
    random_expected = selected.mean() * delta.mean()

    return {
        "a5_minus_structural_ce": float(full_delta.mean()),
        "sd_contribution_ce": float(actual),
        "random_global_sd_contribution_ce": float(random_expected),
        "targeting_gain_global": float(random_expected - actual),
        "sd_selection_fraction": float(selected.mean()),
        "sd_precision": (
            float((selected * wins).sum() / n_selected) if n_selected > 0 else None
        ),
        "sd_win_base_rate": float(wins.mean()),
        "sd_win_recall": (
            float((selected * wins).sum() / wins.sum()) if wins.any() else None
        ),
    }


def bootstrap(delta, selected, full_delta, rng, rounds):
    m = len(delta)
    samples = np.empty((rounds, 3), dtype=np.float64)

    keys = (
        "a5_minus_structural_ce",
        "sd_contribution_ce",
        "targeting_gain_global",
    )

    for i in range(rounds):
        indices = rng.integers(0, m, size=m)
        stats = compute_metrics(
            delta[indices],
            selected[indices],
            full_delta[indices],
        )
        samples[i] = [stats[k] for k in keys]

    return {
        key: np.quantile(samples[:, j], [0.025, 0.975]).tolist()
        for j, key in enumerate(keys)
    }


def permutation_test(delta, selected, relations, rng, rounds):
    # Preserve both directions and per-relation selection patterns.
    groups_by_relation = defaultdict(list)

    for i, relation in enumerate(relations):
        groups_by_relation[int(relation)].append(i)

    groups = [np.asarray(ids, dtype=np.int64) for ids in groups_by_relation.values()]

    observed = float((delta * selected).mean())

    expected = sum(
        (delta[g].mean(axis=0) * selected[g].mean(axis=0)).sum() * len(g)
        for g in groups
    ) / (2 * len(delta))

    shuffled = np.empty_like(selected)
    null_values = np.empty(rounds, dtype=np.float64)

    for i in range(rounds):
        for g in groups:
            shuffled[g] = selected[rng.permutation(g)]

        null_values[i] = (delta * shuffled).mean()

    # Lower CE contribution is better.
    p_value = (1 + np.count_nonzero(null_values <= observed + 1e-12)) / (rounds + 1)

    return {
        "observed_sd_contribution": observed,
        "relation_stratified_random_expected": float(expected),
        "relation_stratified_targeting_gain": float(expected - observed),
        "one_sided_p_better_than_relation_random": float(p_value),
        "null_contribution_95pct_interval": (
            np.quantile(null_values, [0.025, 0.975]).tolist()
        ),
        "n_relation_groups": len(groups),
        "n_permutable_relation_groups": sum(len(g) > 1 for g in groups),
        "permutations": rounds,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path(DEFAULT_DIR),
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        default="test",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2022)

    args = parser.parse_args()

    assert args.bootstrap >= 100
    assert args.permutations >= 100

    source = args.audit_dir / f"{args.split}_per_query.npz"

    unique, delta, selected, full_delta, half = load_pairs(source)

    rng = np.random.default_rng(args.seed)

    metrics = compute_metrics(delta, selected, full_delta)

    ci = bootstrap(
        delta,
        selected,
        full_delta,
        rng,
        args.bootstrap,
    )

    permutation = permutation_test(
        delta,
        selected,
        unique[:, 1],
        rng,
        args.permutations,
    )

    report = {
        "split": args.split,
        "source": str(source),
        "original_triples_including_duplicates": half,
        "unique_original_triples": len(unique),
        "duplicate_original_rows": half - len(unique),
        "paired_queries_after_deduplication": 2 * len(unique),
        "point_estimate_deduplicated": metrics,
        "paired_triple_bootstrap_95pct_CI": ci,
        "relation_stratified_permutation": permutation,
        "limitations": [
            "Post-hoc exploratory statistical analysis.",
            "Original triples deduplicated, directions paired.",
            "Permutation preserves relation-specific selection rates.",
            "Unfiltered teacher CE, not filtered student MRR.",
            "Statistical significance does not imply causality.",
            "Test results were inspected before hypothesis testing.",
        ],
    }

    output = args.audit_dir / (
        f"{args.split}_a5_statistical_audit_seed{args.seed}.json"
    )

    output.write_text(json.dumps(report, indent=2) + "\n")

    print("\n=== A5 STATISTICAL AUDIT ===")
    print("Split:", args.split)
    print("Raw original triples:", half)
    print("Unique original triples:", len(unique))
    print("Duplicate rows:", half - len(unique))

    print("\n--- Point estimates ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\n--- Cluster bootstrap 95% CI ---")
    for key, value in ci.items():
        print(f"{key}: {value}")

    print("\n--- Relation-stratified permutation ---")
    for key, value in permutation.items():
        print(f"{key}: {value}")

    print("\nJSON:", output)
    print("A5 STATISTICAL AUDIT: PASS")


if __name__ == "__main__":
    main()
