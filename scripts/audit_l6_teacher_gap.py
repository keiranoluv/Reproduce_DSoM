#!/usr/bin/env python3
"""
L6: Teacher Generalization Gap Audit.

Evaluate the SAME frozen Teacher on train, valid and test.
No student, no optimizer, no gradient updates.

CE: unfiltered cross-entropy over all entities.
Queries: original and reciprocal triples.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import Dataset
from models import ComplExMDR

NAMES = [
    "Structural",
    "Image",
    "Description",
    "Structural+Image",
    "Structural+Description",
    "Image+Description",
    "All three",
]

STRATEGIES = torch.tensor(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [1 / 3, 1 / 3, 1 / 3],
    ],
    dtype=torch.float32,
)


def select_queries(data, limit, seed):
    queries = torch.from_numpy(data.astype("int64"))

    if limit > 0 and limit < len(queries):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(queries), generator=generator)[:limit]
        queries = queries[indices]

    return queries


@torch.inference_mode()
def evaluate(teacher, queries, batch_size, device):
    strategy = STRATEGIES.to(device)
    n_strategy = len(NAMES)

    ce_sum = torch.zeros(n_strategy, dtype=torch.float64)
    oracle_counts = torch.zeros(n_strategy, dtype=torch.long)
    win_matrix = torch.zeros((n_strategy, n_strategy), dtype=torch.long)

    n_total = 0
    sum_oracle_ce = 0.0

    for begin in range(0, len(queries), batch_size):
        batch = queries[begin : begin + batch_size].to(device)
        targets = batch[:, 2]
        n = len(batch)

        out = teacher.forward(batch)

        if len(out) < 6:
            raise RuntimeError("Unexpected teacher.forward output")

        logits = torch.stack(
            [out[0], out[2], out[4]],
            dim=-1,
        )

        # [batch, 7, entities]
        candidates = torch.einsum(
            "bem,sm->bse",
            logits,
            strategy,
        )

        correct = candidates.gather(
            2,
            targets[:, None, None].expand(-1, n_strategy, 1),
        ).squeeze(-1)

        ce = torch.logsumexp(candidates, dim=-1) - correct

        if not torch.isfinite(ce).all():
            raise RuntimeError(f"Non-finite CE at batch {begin}")

        oracle_ce, oracle_idx = ce.min(dim=1)

        ce_sum += ce.double().sum(dim=0).cpu()
        oracle_counts += torch.bincount(
            oracle_idx.cpu(),
            minlength=n_strategy,
        )
        sum_oracle_ce += oracle_ce.double().sum().item()

        # How often strategy i has lower CE than strategy j.
        wins = ce[:, :, None] < ce[:, None, :]
        win_matrix += wins.sum(dim=0).cpu()

        n_total += n

        if (
            begin == 0
            or (begin // batch_size + 1) % 100 == 0
            or begin + n == len(queries)
        ):
            print(
                f"  processed {n_total}/{len(queries)}",
                flush=True,
            )

    return {
        "n_queries": n_total,
        "mean_ce": (ce_sum / n_total).tolist(),
        "oracle_rate": (oracle_counts.double() / n_total).tolist(),
        "oracle_count": oracle_counts.tolist(),
        "mean_oracle_ce": sum_oracle_ce / n_total,
        "structural_beats_description": (win_matrix[0, 2].item() / n_total),
        "description_beats_structural": (win_matrix[2, 0].item() / n_total),
        "structural_description_ties": (
            1.0 - (win_matrix[0, 2].item() + win_matrix[2, 0].item()) / n_total
        ),
    }


def print_report(results):
    print("\n" + "=" * 75)
    print("L6 TEACHER GENERALIZATION GAP AUDIT")
    print("=" * 75)

    for split, result in results.items():
        print(f"\n[{split.upper()}] Queries: {result['n_queries']}")
        print(f"{'Strategy':<26}{'Mean CE':>12}{'Oracle %':>13}")

        for i, name in enumerate(NAMES):
            print(
                f"{name:<26}"
                f"{result['mean_ce'][i]:>12.4f}"
                f"{result['oracle_rate'][i] * 100:>12.2f}%"
            )

        print(
            "Mean oracle CE:",
            f"{result['mean_oracle_ce']:.4f}",
        )
        print(
            "Structural beats Description:",
            f"{100 * result['structural_beats_description']:.2f}%",
        )
        print(
            "Description beats Structural:",
            f"{100 * result['description_beats_structural']:.2f}%",
        )

    train = results["train"]

    print("\n" + "=" * 75)
    print("GAP RELATIVE TO TRAIN (Mean CE)")
    print("=" * 75)

    print(f"{'Strategy':<26}{'Valid - Train':>16}{'Test - Train':>16}")

    for i, name in enumerate(NAMES):
        valid_gap = results["valid"]["mean_ce"][i] - train["mean_ce"][i]
        test_gap = results["test"]["mean_ce"][i] - train["mean_ce"][i]

        print(f"{name:<26}{valid_gap:>+16.4f}{test_gap:>+16.4f}")

    print("\nNOTE: CE is unfiltered, not filtered ranking.")


def save_results(results, output_dir, args):
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "experiment": "L6",
        "teacher_checkpoint": str(args.teacher),
        "dataset": args.dataset,
        "seed": args.seed,
        "limit_per_split": args.queries,
        "batch_size": args.batch_size,
        "query_type": "original + reciprocal",
        "metric": "unfiltered cross-entropy",
        "strategy_names": NAMES,
        "results": results,
    }

    json_path = output_dir / "results.json"
    json_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    csv_path = output_dir / "strategy_metrics.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "strategy",
                "n_queries",
                "mean_ce",
                "oracle_rate",
                "oracle_count",
            ]
        )

        for split, result in results.items():
            for i, name in enumerate(NAMES):
                writer.writerow(
                    [
                        split,
                        name,
                        result["n_queries"],
                        result["mean_ce"][i],
                        result["oracle_rate"][i],
                        result["oracle_count"][i],
                    ]
                )

    print("\nSaved:", json_path)
    print("Saved:", csv_path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--dataset", default="FB15K-237")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--queries",
        type=int,
        default=2048,
        help="Per split; 0 means the entire split",
    )
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.batch_size <= 0 or args.queries < 0:
        parser.error("batch-size must be positive; queries must be >= 0")

    args.teacher = args.teacher.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    if not args.teacher.is_file():
        raise FileNotFoundError(args.teacher)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    # Dataset uses ../data relative to src.
    os.chdir(ROOT / "src")
    dataset = Dataset(args.dataset)

    device = torch.device("cuda")

    # Only load a checkpoint created by this trusted project.
    teacher = torch.load(
        args.teacher,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(teacher, ComplExMDR):
        raise TypeError(f"Expected ComplExMDR, got {type(teacher)}")

    if tuple(teacher.sizes) != tuple(dataset.get_shape()):
        raise ValueError("Teacher checkpoint and dataset shape mismatch")

    if not (teacher.modality_split and teacher.fusion_img and teacher.fusion_dscp):
        raise ValueError("Teacher must support all three modalities")

    teacher = teacher.to(device)
    teacher.requires_grad_(False)
    teacher.eval()

    print("Teacher:", args.teacher)
    print("Dataset:", args.dataset)
    print("Teacher frozen: YES")
    print("Mode: evaluation only")
    print("Queries per split:", args.queries)

    splits = {
        "train": dataset.get_train(),
        "valid": dataset.get_valid(),
        "test": dataset.get_examples("test"),
    }

    # Dataset.get_examples('test') contains only original
    # triples. Convert to original + reciprocal queries.
    test_original = splits["test"]
    import numpy as np

    reciprocal = np.copy(test_original)
    reciprocal[:, 0] = test_original[:, 2]
    reciprocal[:, 2] = test_original[:, 0]
    reciprocal[:, 1] += dataset.n_predicates // 2

    splits["test"] = np.vstack([test_original, reciprocal])

    results = {}

    for split_idx, (name, data) in enumerate(splits.items()):
        queries = select_queries(
            data,
            args.queries,
            args.seed + split_idx,
        )

        print(
            f"\nEvaluating {name}: {len(queries)} queries",
            flush=True,
        )

        results[name] = evaluate(
            teacher,
            queries,
            args.batch_size,
            device,
        )

    print_report(results)
    save_results(results, args.output, args)


if __name__ == "__main__":
    main()
