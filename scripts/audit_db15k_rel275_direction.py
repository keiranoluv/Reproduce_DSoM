#!/usr/bin/env python3
import json
from pathlib import Path

import numpy as np

root = Path("runs/audit/db15k_a5_selection/20261021_200624_3249374")

with np.load(root / "test_per_query.npz", allow_pickle=False) as z:
    q = z["queries"]
    ce = z["ce"].astype(np.float64)
    actions = z["A5_NDKD"]

n = len(q)
assert n % 2 == 0 and len(ce) == n and len(actions) == n
h = n // 2

assert np.array_equal(q[:h, 0], q[h:, 2])
assert np.array_equal(q[:h, 2], q[h:, 0])

offset = q[h:, 1] - q[:h, 1]
assert np.all(offset == offset[0]) and offset[0] > 0

orig_rel = q[:h, 1]
delta = ce[:, 4] - ce[:, 0]
selected = actions == 4


def calc(mask):
    count = int(mask.sum())
    k = int(selected[mask].sum())
    values = delta[mask]
    chosen = delta[mask & selected]

    actual = float(chosen.sum() / n)
    expected = float((k / count) * values.sum() / n) if count else 0.0

    return {
        "n": count,
        "k": k,
        "rate": k / count if count else 0.0,
        "mean_d": float(values.mean()) if count else float("nan"),
        "chosen_d": float(chosen.mean()) if k else float("nan"),
        "actual": actual,
        "expected": expected,
        "gain": expected - actual,
    }


def show(label, x):
    print(
        f"{label:<23} {x['n']:>6} {x['k']:>5} "
        f"{x['rate']:>7.2%} "
        f"{x['mean_d']:>9.4f} "
        f"{x['chosen_d']:>10.4f} "
        f"{x['actual']:>10.6f} "
        f"{x['expected']:>10.6f} "
        f"{x['gain']:>+10.6f}"
    )


groups = []

for direction, start in (("rhs", 0), ("lhs", h)):
    for rel in np.unique(orig_rel):
        mask = np.zeros(n, dtype=bool)
        mask[start : start + h] = orig_rel == rel
        groups.append((int(rel), direction, calc(mask)))

print("\n=== RELATION 275: DIRECTION DECOMPOSITION ===")
print(
    f"{'Group':<23} {'N':>6} {'S+D':>5} "
    f"{'Rate':>7} {'Mean D':>9} "
    f"{'Chosen D':>10} {'Actual':>10} "
    f"{'Random':>10} {'Gain':>10}"
)

for rel, direction, stats in groups:
    if rel == 275:
        show(f"rel275/{direction}", stats)

r275 = np.tile(orig_rel == 275, 2)

show("rel275/pooled", calc(r275))
show("other/pooled", calc(~r275))
show("all/pooled", calc(np.ones(n, dtype=bool)))

expected = sum(x["expected"] for _, _, x in groups)
actual = sum(x["actual"] for _, _, x in groups)

print("\n=== RELATION x DIRECTION CONTROL ===")
print("Observed CE contribution:", round(actual, 9))
print("Random expected CE contribution:", round(expected, 9))
print("Targeting gain:", round(expected - actual, 9))

previous = json.loads((root / "test_a5_statistical_audit_seed2022.json").read_text())[
    "relation_stratified_permutation"
]

assert abs(expected - previous["relation_stratified_random_expected"]) < 1e-6

assert abs(actual - previous["observed_sd_contribution"]) < 1e-6

print("PREVIOUS PERMUTATION CROSSCHECK: PASS")

print("\n=== LOWEST TARGETING GAINS ===")
ranked = sorted((g for g in groups if g[2]["k"] > 0), key=lambda g: g[2]["gain"])

for rel, direction, stats in ranked[:10]:
    show(f"rel{rel}/{direction}", stats)

print("\n=== HIGHEST TARGETING GAINS ===")
for rel, direction, stats in ranked[-10:][::-1]:
    show(f"rel{rel}/{direction}", stats)
