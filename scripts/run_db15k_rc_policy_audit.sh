#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---help}"

case "$MODE" in
  --smoke) LIMIT=128 ;;
  --full) LIMIT=0 ;;
  *)
    echo "Usage: $0 --smoke|--full"
    exit 2
    ;;
esac

cd "$ROOT"
PY="$ROOT/dsom/bin/python"
[[ -x "$PY" ]] || { echo "Missing Python: $PY"; exit 1; }

OUT="$ROOT/runs/audit/db15k_rc_policy/$(date -u +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUT"

"$PY" - "$ROOT" "$OUT" "$LIMIT" <<'PY'
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(sys.argv[1])
out = Path(sys.argv[2])
limit = int(sys.argv[3])

sys.path.insert(0, str(root / "src"))
from datasets import Dataset
from kd import KD_RL

assert torch.cuda.is_available(), "CUDA required"

base = root / "runs/student/MMKB-DB15K"

paths = {
    "A5_NDKD": (
        base / "learned_rc_adam_full/"
        "20261021_173746_2988787/checkpoints/0/result.json"
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

strategy_names = [
    "Structural",
    "Image",
    "Description",
    "Structural+Image",
    "Structural+Description",
    "Image+Description",
    "Average",
]

expected_losses = {
    "A5_NDKD": "ndkd",
    "A9_KL": "kd",
    "A10_DKD": "dkd",
}

# Dataset.DATA_PATH is relative to src/.
os.chdir(root / "src")
dataset = Dataset("MMKB-DB15K")

relation_offset = dataset.n_predicates // 2


def get_queries(split):
    if split == "train":
        queries = dataset.get_train()
    elif split == "valid":
        queries = dataset.get_valid()
    else:
        raw = dataset.get_examples("test")
        inverse = raw.copy()

        inverse[:, 0] = raw[:, 2]
        inverse[:, 2] = raw[:, 0]
        inverse[:, 1] += relation_offset

        queries = np.vstack([raw, inverse])

    queries = np.asarray(queries, dtype=np.int64)

    if limit:
        half = len(queries) // 2
        k = max(1, limit // 2)

        queries = np.vstack([
            queries[:k],
            queries[half:half + k],
        ])

    return np.ascontiguousarray(queries)


splits = {
    name: get_queries(name)
    for name in ("train", "valid", "test")
}

query_hashes = {
    name: hashlib.sha256(q.tobytes()).hexdigest()
    for name, q in splits.items()
}

print("=== DB15K RC POLICY AUDIT ===", flush=True)
print("Mode:", "SMOKE" if limit else "FULL", flush=True)
print("Output:", out, flush=True)
print("Queries:", {k: len(v) for k, v in splits.items()}, flush=True)

for name, result_path in paths.items():
    assert result_path.is_file(), f"Missing: {result_path}"

    metadata = json.loads(result_path.read_text())

    assert metadata["dataset"] == "MMKB-DB15K"
    assert metadata["kd_loss_type"] == "kdrl"
    assert metadata["teacher_strategy"] == "rc"
    assert metadata["policy_optimizer"] == "Adam"
    assert abs(metadata["policy_lr"] - 1e-5) < 1e-12
    assert metadata["rc_reward_mode"] == "binary"
    assert metadata["max_train_examples"] == 0
    assert metadata["eval_queries"] == -1
    assert metadata.get("rc_distill_loss", "ndkd") == expected_losses[name]

    checkpoints = list(
        result_path.parent.glob("distiller-*.pth")
    )
    assert len(checkpoints) == 1, (name, checkpoints)

    print(f"\n=== LOADING {name} ===", flush=True)
    print(checkpoints[0], flush=True)

    # Only load trusted, locally generated checkpoints.
    model = torch.load(
        checkpoints[0],
        map_location="cuda",
        weights_only=False,
    )

    assert isinstance(model, KD_RL)
    assert model.strategy.shape == (7, 3)
    assert tuple(model.teacher.sizes) == tuple(dataset.get_shape())

    # Disable dropout for deterministic policy evaluation.
    model.eval()
    assert not model.rewarder.training

    strategy_matrix = model.strategy.to("cuda")

    run_output = out / name
    run_output.mkdir()

    report = {
        "run": name,
        "best_epoch": metadata["best_epoch"],
        "git_commit": metadata.get("git_commit"),
        "checkpoint": str(checkpoints[0]),
        "loss": expected_losses[name],
        "splits": {},
    }

    with torch.inference_mode():

        for split_name, queries in splits.items():
            n = len(queries)

            actions = np.empty(n, dtype=np.uint8)
            probabilities = np.empty((n, 7), dtype=np.float32)

            action_counts = np.zeros(7, dtype=np.int64)
            oracle_counts = np.zeros(7, dtype=np.int64)
            confusion = np.zeros((7, 7), dtype=np.int64)

            # entropy, maxp, deterministic, policy CE,
            # structural CE, oracle CE, oracle match
            totals = np.zeros(7, dtype=np.float64)

            for start in range(0, n, 128):
                batch = torch.from_numpy(
                    queries[start:start + 128]
                ).to("cuda")

                labels = batch[:, 2]

                teacher_out = model.teacher.forward(batch)

                logits = torch.stack([
                    teacher_out[0],
                    teacher_out[2],
                    teacher_out[4],
                ], dim=-1)

                probs = model.policy_probs(logits)

                assert probs.shape == (len(batch), 7)
                assert torch.isfinite(probs).all()
                assert torch.allclose(
                    probs.sum(-1),
                    torch.ones(len(batch), device="cuda"),
                    atol=1e-4,
                )

                greedy = probs.argmax(-1)

                entropy = -(
                    probs * probs.clamp_min(1e-12).log()
                ).sum(-1)

                max_prob = probs.max(-1).values

                # All seven teacher combinations.
                candidate_logits = torch.einsum(
                    "bem,sm->bse",
                    logits,
                    strategy_matrix,
                )

                # Unfiltered cross-entropy for each strategy.
                ce = (
                    torch.logsumexp(candidate_logits, dim=-1)
                    - candidate_logits.gather(
                        2,
                        labels[:, None, None].expand(-1, 7, 1),
                    ).squeeze(-1)
                )

                # Diagnostic oracle knows the true answer.
                oracle = ce.argmin(-1)

                greedy_ce = ce.gather(
                    1, greedy[:, None]
                ).squeeze(-1)

                oracle_ce = ce.gather(
                    1, oracle[:, None]
                ).squeeze(-1)

                structural_ce = ce[:, 0]

                size = len(batch)

                actions[start:start + size] = (
                    greedy.cpu().numpy()
                )

                probabilities[start:start + size] = (
                    probs.cpu().numpy()
                )

                action_counts += torch.bincount(
                    greedy, minlength=7
                ).cpu().numpy()

                oracle_counts += torch.bincount(
                    oracle, minlength=7
                ).cpu().numpy()

                confusion += torch.bincount(
                    greedy * 7 + oracle,
                    minlength=49,
                ).cpu().numpy().reshape(7, 7)

                totals += np.array([
                    entropy.sum().item(),
                    max_prob.sum().item(),
                    (max_prob > 0.999).sum().item(),
                    greedy_ce.sum().item(),
                    structural_ce.sum().item(),
                    oracle_ce.sum().item(),
                    (greedy == oracle).sum().item(),
                ])

                if (start // 128 + 1) % 250 == 0:
                    print(
                        name, split_name,
                        min(start + size, n), "/", n,
                        flush=True,
                    )

            np.save(
                run_output / f"{split_name}_greedy.npy",
                actions,
            )
            np.save(
                run_output / f"{split_name}_probs.npy",
                probabilities,
            )

            result = {
                "n_queries": n,
                "queries_sha256": query_hashes[split_name],
                "greedy_action_counts": action_counts.tolist(),
                "greedy_action_fractions": (
                    action_counts / n
                ).tolist(),
                "mean_probability_mass": (
                    probabilities.mean(0)
                ).tolist(),
                "oracle_action_counts": oracle_counts.tolist(),
                "oracle_action_fractions": (
                    oracle_counts / n
                ).tolist(),
                "greedy_vs_oracle_confusion": confusion.tolist(),
                "mean_entropy": totals[0] / n,
                "mean_max_probability": totals[1] / n,
                "near_deterministic_fraction": totals[2] / n,
                "mean_greedy_ce": totals[3] / n,
                "mean_structural_ce": totals[4] / n,
                "mean_oracle_ce": totals[5] / n,
                "greedy_oracle_match_fraction": totals[6] / n,
                "greedy_ce_minus_structural_ce": (
                    totals[3] - totals[4]
                ) / n,
                "greedy_ce_regret_to_oracle": (
                    totals[3] - totals[5]
                ) / n,
            }

            report["splits"][split_name] = result

            print(
                f"{name}/{split_name}: "
                f"structural={result['greedy_action_fractions'][0]:.5f}, "
                f"entropy={result['mean_entropy']:.6f}, "
                f"oracle_match={result['greedy_oracle_match_fraction']:.5f}",
                flush=True,
            )

            print(
                "  Greedy strategy fractions:",
                np.round(action_counts / n, 5).tolist(),
                flush=True,
            )

            print(
                "  CE-oracle strategy fractions:",
                np.round(oracle_counts / n, 5).tolist(),
                flush=True,
            )

    (run_output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    del model, strategy_matrix
    gc.collect()
    torch.cuda.empty_cache()


# Compare all policies on the same ordered queries.
pairs = [
    ("A5_NDKD", "A9_KL"),
    ("A5_NDKD", "A10_DKD"),
    ("A9_KL", "A10_DKD"),
]

pairwise = {}

for split_name in splits:
    pairwise[split_name] = {}

    for left, right in pairs:
        a = np.load(out / left / f"{split_name}_greedy.npy")
        b = np.load(out / right / f"{split_name}_greedy.npy")

        pa = np.load(out / left / f"{split_name}_probs.npy")
        pb = np.load(out / right / f"{split_name}_probs.npy")

        pairwise[split_name][f"{left}_vs_{right}"] = {
            "greedy_agreement": float((a == b).mean()),
            "mean_soft_l1": float(
                np.abs(pa - pb).sum(-1).mean()
            ),
        }

summary = {
    "dataset": "MMKB-DB15K",
    "mode": "smoke" if limit else "full",
    "strategies": strategy_names,
    "query_sha256": query_hashes,
    "pairwise": pairwise,
    "warnings": [
        "Validation leakage: validation is not independent.",
        "CE-oracle uses true labels and is diagnostic only.",
        "Unfiltered CE is not filtered MRR.",
        "Policy evaluation uses greedy argmax with dropout disabled.",
    ],
}

(out / "summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)

lines = [
    "# DB15K RC Policy Audit",
    "",
    "Policy: greedy argmax, evaluation mode (dropout disabled).",
    "",
    "| Run | Split | Queries | Structural | Entropy | Oracle match |",
    "|---|---|---:|---:|---:|---:|",
]

for run in paths:
    r = json.loads((out / run / "report.json").read_text())

    for split_name, v in r["splits"].items():
        lines.append(
            f"| {run} | {split_name} | {v['n_queries']} | "
            f"{v['greedy_action_fractions'][0]:.4%} | "
            f"{v['mean_entropy']:.6f} | "
            f"{v['greedy_oracle_match_fraction']:.4%} |"
        )

lines.extend([
    "",
    "## Pairwise Policy Agreement",
    "",
    "| Split | Pair | Argmax agreement | Soft L1 |",
    "|---|---|---:|---:|",
])

for split_name, pairs_data in pairwise.items():
    for pair, v in pairs_data.items():
        lines.append(
            f"| {split_name} | {pair} | "
            f"{v['greedy_agreement']:.4%} | "
            f"{v['mean_soft_l1']:.6f} |"
        )

lines.extend([
    "",
    "## Limitations",
    "",
    *["- " + warning for warning in summary["warnings"]],
])

(out / "report.md").write_text("\n".join(lines) + "\n")

print("\n=== PAIRWISE POLICY AGREEMENT ===")

for split_name, pairs_data in pairwise.items():
    for pair, v in pairs_data.items():
        print(
            split_name, pair,
            f"agreement={v['greedy_agreement']:.4%}",
            f"L1={v['mean_soft_l1']:.6f}",
        )

print("\nSummary:", out / "summary.json")
print("Markdown:", out / "report.md")
print("DB15K RC POLICY AUDIT: PASS")
PY
