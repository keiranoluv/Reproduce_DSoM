#!/usr/bin/env python3
"""Analyze the contribution of A5's rare non-Structural choices."""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

NAMES = ("Structural", "Image", "Description", "S+I", "S+D", "I+D", "Average")

WEIGHTS = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0.5, 0.5, 0],
    [0.5, 0, 0.5],
    [0, 0.5, 0.5],
    [1 / 3, 1 / 3, 1 / 3],
]

RUNS = ("A5_NDKD", "A9_KL", "A10_DKD")


def get_queries(ds, split):
    if split == "train":
        q = ds.get_train()
    elif split == "valid":
        q = ds.get_valid()
    else:
        raw = ds.get_examples("test")
        inv = raw.copy()
        inv[:, 0] = raw[:, 2]
        inv[:, 2] = raw[:, 0]
        inv[:, 1] += ds.n_predicates // 2
        q = np.vstack((raw, inv))

    return np.ascontiguousarray(q, dtype=np.int64)


def analyze(ce, actions):
    a5, a9, a10 = (actions[k] for k in RUNS)
    n = len(a5)

    sd = a5 == 4
    other = (a5 != 0) & ~sd

    d = ce[:, 4].astype("float64") - ce[:, 0].astype("float64")

    full_d = ce[np.arange(n), a5].astype("float64") - ce[:, 0].astype("float64")

    wins = d < 0

    contrib_sd = np.where(sd, d, 0).mean()
    contrib_other = np.where(other, full_d, 0).mean()
    random_ce = sd.mean() * d.mean()

    ans = dict(
        n=n,
        a5_actions=np.bincount(a5, minlength=7).tolist(),
        a5_structural_fraction=float((a5 == 0).mean()),
        a5_sd_fraction=float(sd.mean()),
        a9_structural_on_a5_sd=(float((a9[sd] == 0).mean()) if sd.any() else None),
        a10_structural_on_a5_sd=(float((a10[sd] == 0).mean()) if sd.any() else None),
        sd_win_over_structural_fraction_all=float(wins.mean()),
        sd_win_fraction_when_a5_selects_sd=(
            float(wins[sd].mean()) if sd.any() else None
        ),
        sd_win_recall=(float((sd & wins).sum() / wins.sum()) if wins.any() else None),
        mean_sd_minus_structural_on_chosen=(float(d[sd].mean()) if sd.any() else None),
        mean_sd_minus_structural_on_not_chosen=(
            float(d[~sd].mean()) if (~sd).any() else None
        ),
        mean_structural_ce=float(ce[:, 0].mean(dtype=np.float64)),
        mean_a5_ce=float(ce[np.arange(n), a5].mean(dtype=np.float64)),
        mean_a5_minus_structural_ce=float(full_d.mean()),
        sd_contribution_to_full_ce=float(contrib_sd),
        other_contribution_to_full_ce=float(contrib_other),
        random_sd_at_same_rate_expected_ce=float(random_ce),
        targeting_gain_over_random=float(random_ce - contrib_sd),
        structural_or_sd_oracle_ce=float(
            np.minimum(ce[:, 0], ce[:, 4]).mean(dtype=np.float64)
        ),
        strategy_contributions={
            NAMES[j]: float(np.where(a5 == j, full_d, 0).mean()) for j in range(7)
        },
    )

    assert abs(ans["mean_a5_minus_structural_ce"] - contrib_sd - contrib_other) < 1e-7

    return ans


def fmt_pct(value):
    return "N/A" if value is None else f"{value:.2%}"


def fmt_num(value):
    return "N/A" if value is None else f"{value:+.6f}"


def main():
    ap = argparse.ArgumentParser()

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--full", action="store_true")

    ap.add_argument("--audit-dir", type=Path)
    ap.add_argument("--batch-size", type=int, default=128)

    args = ap.parse_args()
    assert args.batch_size > 0

    root = Path(__file__).resolve().parent.parent

    prior_dir = (
        args.audit_dir or root / ("runs/audit/db15k_rc_policy/20261021_194427_3194329")
    ).resolve()

    prior = json.loads((prior_dir / "summary.json").read_text())

    assert prior["dataset"] == "MMKB-DB15K"
    assert prior["mode"] == "full"
    assert prior["strategies"] == [
        "Structural",
        "Image",
        "Description",
        "Structural+Image",
        "Structural+Description",
        "Image+Description",
        "Average",
    ]

    sys.path.insert(0, str(root / "src"))

    from datasets import Dataset
    from models import ComplExMDR

    assert torch.cuda.is_available(), "CUDA required"

    os.chdir(root / "src")
    ds = Dataset("MMKB-DB15K")

    a5_result = root / (
        "runs/student/MMKB-DB15K/learned_rc_adam_full/"
        "20261021_173746_2988787/"
        "checkpoints/0/result.json"
    )

    meta = json.loads(a5_result.read_text())
    checkpoint = Path(meta["teacher_checkpoint"])

    assert checkpoint.is_file()

    print("Loading trusted Teacher:", checkpoint, flush=True)

    teacher = torch.load(
        checkpoint,
        map_location="cuda",
        weights_only=False,
    )

    assert isinstance(teacher, ComplExMDR)
    assert tuple(teacher.sizes) == tuple(ds.get_shape())

    teacher.eval()

    weights = torch.tensor(
        WEIGHTS,
        device="cuda",
        dtype=torch.float32,
    )

    output = (
        root
        / "runs/audit/db15k_a5_selection"
        / (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}")
    )

    output.mkdir(parents=True, exist_ok=False)
    print("Output:", output, flush=True)

    reports = {}

    for split in ("train", "valid", "test"):
        q = get_queries(ds, split)

        digest = hashlib.sha256(q.tobytes()).hexdigest()

        assert digest == prior["query_sha256"][split], f"{split}: query order changed!"

        actions = {k: np.load(prior_dir / k / f"{split}_greedy.npy") for k in RUNS}

        for a in actions.values():
            assert len(a) == len(q)
            assert np.all(a < 7)

        if args.smoke:
            k = min(64, len(q) // 2)

            idx = np.r_[
                np.arange(k),
                np.arange(
                    len(q) // 2,
                    len(q) // 2 + k,
                ),
            ]

            q = q[idx]

            actions = {name: a[idx] for name, a in actions.items()}

        parts = []

        with torch.inference_mode():
            for i in range(0, len(q), args.batch_size):
                batch = torch.from_numpy(q[i : i + args.batch_size]).cuda()

                result = teacher.forward(batch)

                modality = torch.stack(
                    [result[0], result[2], result[4]],
                    dim=-1,
                )

                scores = torch.einsum(
                    "bem,sm->bse",
                    modality,
                    weights,
                )

                labels = batch[:, 2]

                ce = torch.logsumexp(scores, dim=-1) - scores.gather(
                    2,
                    labels[:, None, None].expand(-1, 7, 1),
                ).squeeze(-1)

                assert torch.isfinite(ce).all(), (split, i)

                parts.append(ce.cpu().numpy())

                if (i // args.batch_size + 1) % 250 == 0:
                    print(
                        split,
                        min(i + len(batch), len(q)),
                        "/",
                        len(q),
                        flush=True,
                    )

        ce = np.concatenate(parts)
        r = analyze(ce, actions)

        if args.full:
            expected = json.loads((prior_dir / "A5_NDKD/report.json").read_text())[
                "splits"
            ][split]

            assert abs(r["mean_structural_ce"] - expected["mean_structural_ce"]) < 0.005

            assert abs(r["mean_a5_ce"] - expected["mean_greedy_ce"]) < 0.005

        reports[split] = r

        np.savez_compressed(
            output / f"{split}_per_query.npz",
            queries=q,
            ce=ce,
            **actions,
        )

        print(
            f"{split}: "
            f"S+D={r['a5_sd_fraction']:.2%} "
            f"win-rate={fmt_pct(r['sd_win_fraction_when_a5_selects_sd'])} "
            f"A5-Structural={r['mean_a5_minus_structural_ce']:+.6f} "
            f"S+D contribution={r['sd_contribution_to_full_ce']:+.6f} "
            f"random same-rate={r['random_sd_at_same_rate_expected_ce']:+.6f}",
            flush=True,
        )

    notes = [
        "Greedy actions reused from full policy audit.",
        "CE is unfiltered Teacher CE, not filtered Student MRR.",
        "Oracle uses labels and is diagnostic only.",
        "Random same-rate is an expectation, not a fitted policy.",
        "Test comparisons are exploratory and post-hoc.",
        "DB15K validation has leakage and duplicate triples.",
    ]

    summary = dict(
        dataset="MMKB-DB15K",
        mode="full" if args.full else "smoke",
        prior_audit=str(prior_dir),
        teacher_checkpoint=str(checkpoint),
        splits=reports,
        caveats=notes,
    )

    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# A5 RC Structural+Description Selection Attribution",
        "",
        "| Split | N | A5 S+D | S+D better when chosen | A5-Structural CE | S+D contribution | Other contribution | Random same-rate S+D |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for split, x in reports.items():
        lines.append(
            f"| {split} | {x['n']} | "
            f"{x['a5_sd_fraction']:.2%} | "
            f"{fmt_pct(x['sd_win_fraction_when_a5_selects_sd'])} | "
            f"{x['mean_a5_minus_structural_ce']:+.6f} | "
            f"{x['sd_contribution_to_full_ce']:+.6f} | "
            f"{x['other_contribution_to_full_ce']:+.6f} | "
            f"{x['random_sd_at_same_rate_expected_ce']:+.6f} |"
        )

    lines += [
        "",
        "## Selection diagnostics",
        "",
        "| Split | S+D better in all queries | A5 S+D precision | Recall of S+D wins | Mean S+D-S when chosen | Gain vs random | A9 Structural on A5 S+D | A10 Structural on A5 S+D |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for split, x in reports.items():
        lines.append(
            f"| {split} | "
            f"{x['sd_win_over_structural_fraction_all']:.2%} | "
            f"{fmt_pct(x['sd_win_fraction_when_a5_selects_sd'])} | "
            f"{fmt_pct(x['sd_win_recall'])} | "
            f"{fmt_num(x['mean_sd_minus_structural_on_chosen'])} | "
            f"{x['targeting_gain_over_random']:+.6f} | "
            f"{fmt_pct(x['a9_structural_on_a5_sd'])} | "
            f"{fmt_pct(x['a10_structural_on_a5_sd'])} |"
        )

    lines += ["", "## Caveats", ""]
    lines += [f"- {s}" for s in notes]

    (output / "report.md").write_text("\n".join(lines) + "\n")

    print("Summary:", output / "summary.json")
    print("Markdown:", output / "report.md")
    print("DB15K A5 SELECTION ATTRIBUTION: PASS")


if __name__ == "__main__":
    main()
