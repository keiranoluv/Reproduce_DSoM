#!/usr/bin/env python3
"""Generate DB15K RC research report from existing audit artifacts."""

import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent.parent
p = root / "runs/audit/db15k_rc_policy/20261021_194427_3194329"
a = root / "runs/audit/db15k_a5_selection/20261021_200624_3249374"
out = root / "docs/experiments/db15k_rc_policy_analysis.md"


def j(path):
    assert path.is_file(), f"Missing {path}"
    return json.loads(path.read_text())


mapping = {}
for line in (root / "data/MMKB-DB15K/rel_id").read_text().splitlines():
    name, id_ = line.rsplit("\t", 1)
    mapping[int(id_)] = name

assert len(mapping) == 279
assert mapping[275] == ("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")

pol = {k: j(p / k / "report.json") for k in ("A5_NDKD", "A9_KL", "A10_DKD")}

ps = j(p / "summary.json")
at = j(a / "summary.json")

stats = {s: j(a / f"{s}_a5_statistical_audit_seed2022.json") for s in ("valid", "test")}

assert ps["mode"] == at["mode"] == "full"

with np.load(a / "test_per_query.npz", allow_pickle=False) as z:
    q = z["queries"]
    ce = z["ce"].astype("float64")
    act = z["A5_NDKD"]

n = len(q)
h = n // 2

assert n == 19804
assert ce.shape == (n, 7)
assert np.array_equal(q[:h, 0], q[h:, 2])
assert np.array_equal(q[:h, 2], q[h:, 0])
assert np.all(q[h:, 1] - q[:h, 1] == 279)

sd = act == 4
delta = ce[:, 4] - ce[:, 0]

rhs = np.zeros(n, bool)
lhs = np.zeros(n, bool)
rhs[:h] = q[:h, 1] == 275
lhs[h:] = q[:h, 1] == 275

assert rhs.sum() == lhs.sum() == 1345
assert sd.sum() == 945
assert (sd & rhs).sum() == 1
assert (sd & lhs).sum() == 859


def group(mask):
    d = delta[mask]
    chosen = delta[mask & sd]
    return (
        len(d),
        len(chosen),
        len(chosen) / len(d),
        d.mean(),
        chosen.mean(),
        chosen.sum() / n,
    )


def gline(label, mask):
    count, k, rate, mean, selected, contribution = group(mask)
    return (
        f"| {label} | {count} | {k} | {rate:.2%} | "
        f"{mean:+.4f} | {selected:+.4f} | "
        f"{contribution:+.6f} |"
    )


lines = [
    "# DB15K RC Policy Audit — A5, A9, A10",
    "",
    "**Status:** Completed; exploratory, single seed (2022).",
    "",
    "## Main findings",
    "",
    "- Learned RC policies are near deterministic and Structural-dominant.",
    "- A5 makes sparse S+D selections concentrated on `rdf:type` LHS queries.",
    "- A5 improves aggregate test Teacher CE against Structural-only.",
    "- This advantage reverses against relation-direction-stratified random.",
    "- Query-level adaptation and causal Student MRR benefits remain unverified.",
    "",
    "## Verified relation mapping",
    "",
    f"- Relation 275: `{mapping[275]}`.",
    "- RDF `rdf:type` relates an entity to a class/type.",
    "- RHS predicts type from entity; LHS predicts entity from type.",
    "- Directional asymmetry is a hypothesis, not a causal explanation.",
    "",
    "## Student best-checkpoint metrics",
    "",
    "| Experiment | Loss | Epoch | MRR | H@1 | H@3 | H@10 |",
    "|---|---|---:|---:|---:|---:|---:|",
]

base = root / "runs/student/MMKB-DB15K"

result_files = {
    "A5_NDKD": (
        base / "learned_rc_adam_full/20261021_173746_2988787/checkpoints/0/result.json"
    ),
    "A9_KL": (
        base / "table4_learned_rc_a9_a10/full/"
        "20261021_191355_3114489/A9_KL/checkpoints/0/result.json"
    ),
    "A10_DKD": (
        base / "table4_learned_rc_a9_a10/full/"
        "20261021_191355_3114489/A10_DKD/checkpoints/0/result.json"
    ),
}

for name, loss in [
    ("A5_NDKD", "NDKD"),
    ("A9_KL", "Vanilla KD"),
    ("A10_DKD", "DKD"),
]:
    r = j(result_files[name])
    t = r["final_result"]
    hits = t["hits@[1,3,10]"]

    lines.append(
        f"| {name} | {loss} | {r['best_epoch']} | "
        f"{t['MRR']:.6f} | {hits[0]:.6f} | "
        f"{hits[1]:.6f} | {hits[2]:.6f} |"
    )

lines += [
    "",
    "## Policy behavior by split",
    "",
    "Greedy argmax, dropout disabled.",
    "",
    "| Model | Split | N | Structural | S+D | Entropy | Oracle match |",
    "|---|---|---:|---:|---:|---:|---:|",
]

for name, r in pol.items():
    for split in ("train", "valid", "test"):
        d = r["splits"][split]
        f = d["greedy_action_fractions"]

        lines.append(
            f"| {name} | {split} | {d['n_queries']} | "
            f"{f[0]:.2%} | {f[4]:.2%} | "
            f"{d['mean_entropy']:.6f} | "
            f"{d['greedy_oracle_match_fraction']:.2%} |"
        )

lines += [
    "",
    "## CE-oracle distribution",
    "",
    "The oracle uses ground-truth labels and is diagnostic only.",
    "",
    "| Split | Structural | Description | S+D |",
    "|---|---:|---:|---:|",
]

for split in ("train", "valid", "test"):
    f = pol["A5_NDKD"]["splits"][split]["oracle_action_fractions"]
    lines.append(f"| {split} | {f[0]:.2%} | {f[2]:.2%} | {f[4]:.2%} |")

lines += [
    "",
    "## A5 Teacher CE attribution",
    "",
    "Negative CE differences indicate improvement.",
    "",
    "| Split | S+D rate | S+D precision | A5-S CE | S+D contribution | Random same-rate CE |",
    "|---|---:|---:|---:|---:|---:|",
]

for split in ("train", "valid", "test"):
    d = at["splits"][split]

    lines.append(
        f"| {split} | {d['a5_sd_fraction']:.2%} | "
        f"{d['sd_win_fraction_when_a5_selects_sd']:.2%} | "
        f"{d['mean_a5_minus_structural_ce']:+.6f} | "
        f"{d['sd_contribution_to_full_ce']:+.6f} | "
        f"{d['random_sd_at_same_rate_expected_ce']:+.6f} |"
    )

lines += [
    "",
    "## Relation 275 direction decomposition",
    "",
    "860 of 945 S+D choices are on `rdf:type`.",
    "**859 of 945 are specifically on reciprocal/LHS queries.**",
    "",
    "| Group | Queries | S+D | Rate | Mean delta | Chosen delta | CE contribution |",
    "|---|---:|---:|---:|---:|---:|---:|",
    gline("rdf:type RHS", rhs),
    gline("rdf:type LHS", lhs),
    gline("Other relations", ~(rhs | lhs)),
    gline("All test", np.ones(n, bool)),
    "",
    "Within rdf:type/LHS, random S+D has greater mean CE benefit "
    "than the subset selected by A5.",
]

lines += [
    "",
    "## Statistical robustness",
    "",
    "Bootstrap pairs reciprocal directions of unique original triples. "
    "Conditional permutations preserve relation and direction selection rates.",
    "",
    "| Split | Unique triples | A5-S CE | Bootstrap 95% CI | Global gain | Conditional gain | One-sided p |",
    "|---|---:|---:|---|---:|---:|---:|",
]

for split, v in stats.items():
    x = v["point_estimate_deduplicated"]
    ci = v["paired_triple_bootstrap_95pct_CI"]["a5_minus_structural_ce"]
    perm = v["relation_stratified_permutation"]

    lines.append(
        f"| {split} | {v['unique_original_triples']} | "
        f"{x['a5_minus_structural_ce']:+.6f} | "
        f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | "
        f"{x['targeting_gain_global']:+.6f} | "
        f"{perm['relation_stratified_targeting_gain']:+.6f} | "
        f"{perm['one_sided_p_better_than_relation_random']:.6f} |"
    )

lines += [
    "",
    "## Policy agreement on test",
    "",
    "| Pair | Greedy agreement | Mean soft L1 |",
    "|---|---:|---:|",
]

for name, d in ps["pairwise"]["test"].items():
    lines.append(f"| {name} | {d['greedy_agreement']:.2%} | {d['mean_soft_l1']:.6f} |")

lines += [
    "",
    "## Limitations",
    "",
    "- CE-oracle uses ground-truth labels.",
    "- Teacher CE is unfiltered; Student MRR uses filtered ranking.",
    "- Teacher selection occurs during Student training, not Student inference.",
    "- DB15K validation contains duplicate triples and split leakage.",
    "- Tests are exploratory and post-hoc.",
    "- Experiments use one seed and a reconstructed feature pipeline.",
    "- Learned RC differs from released-code source-frozen RC.",
    "",
    "## Next steps",
    "",
    "- Verify NDKD neighbor/non-neighbor decomposition before Table 4 ablations.",
    "- Reproduce key experiments with additional seeds.",
    "- Evaluate sensitivity to DB15K validation leakage.",
    "",
    "## Provenance",
    "",
    f"- Policy audit: `{p.relative_to(root)}`",
    f"- Selection attribution: `{a.relative_to(root)}`",
    "- Relation mapping: `data/MMKB-DB15K/rel_id`",
    "- Generated by `scripts/write_db15k_rc_report.py`.",
]

expected = at["splits"]["test"]["sd_contribution_to_full_ce"]
observed = group(np.ones(n, bool))[-1]
assert abs(observed - expected) < 1e-6

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("REPORT GENERATED:", out)
print("RELATION:", mapping[275])
print("A5 S+D:", int(sd.sum()))
print("rdf:type/LHS:", int((sd & lhs).sum()))
