#!/usr/bin/env python3
"""
L8 - Filtered Teacher Ranking Audit.

Evaluate seven teacher strategies with the exact
filtering convention used by DSoM's KBCModel.get_ranking:

    rank = 1 + number of unfiltered candidates
               with score >= target score

The target entity and all other known correct
answers are excluded from the candidate count.

Runs RHS and LHS independently.
No training or parameter updates.
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

    if 0 < limit < len(queries):
        generator = torch.Generator().manual_seed(seed)

        indices = torch.randperm(
            len(queries),
            generator=generator,
        )[:limit]

        queries = queries[indices]

    return queries


def make_queries(original, direction, n_relations):
    queries = original.clone()

    if direction == "lhs":
        head = queries[:, 0].clone()
        queries[:, 0] = queries[:, 2]
        queries[:, 2] = head
        queries[:, 1] += n_relations // 2

    return queries


@torch.inference_mode()
def evaluate_direction(
    teacher,
    queries,
    filters,
    batch_size,
    device,
):
    strategies = STRATEGIES.to(device)
    n_strategies = len(NAMES)

    rank_sum = torch.zeros(n_strategies, dtype=torch.float64)
    reciprocal_rank_sum = torch.zeros_like(rank_sum)
    hits1 = torch.zeros_like(rank_sum)
    hits3 = torch.zeros_like(rank_sum)
    hits10 = torch.zeros_like(rank_sum)

    total = 0
    n_entities = teacher.sizes[2]

    for begin in range(0, len(queries), batch_size):
        batch_cpu = queries[begin : begin + batch_size]
        batch = batch_cpu.to(device)

        n = batch.shape[0]
        target_ids = batch[:, 2]

        output = teacher.forward(batch)

        logits_stack = torch.stack(
            [output[0], output[2], output[4]],
            dim=-1,
        )

        # [batch, strategies, entities]
        scores = torch.einsum(
            "bem,sm->bse",
            logits_stack,
            strategies,
        )

        if scores.shape != (n, n_strategies, n_entities):
            raise RuntimeError(f"Unexpected score shape: {scores.shape}")

        if not torch.isfinite(scores).all():
            raise RuntimeError("Non-finite teacher scores")

        # Target scores MUST be collected before filtering.
        target_scores = scores.gather(
            2,
            target_ids[:, None, None].expand(-1, n_strategies, 1),
        )

        # Build one filter mask per query.
        # Mask includes the current target entity.
        # This matches DSoM's get_ranking convention.
        filtered = torch.zeros(
            (n, n_entities),
            dtype=torch.bool,
            device=device,
        )

        for i, triple in enumerate(batch_cpu.tolist()):
            head, relation, tail = map(int, triple)

            key = (head, relation)

            if key not in filters:
                raise KeyError(f"Missing filter key {key}")

            answer_ids = list(filters[key])
            answer_ids.append(tail)

            filtered[
                i,
                torch.as_tensor(
                    answer_ids,
                    dtype=torch.long,
                    device=device,
                ),
            ] = True

        # Count candidates with score >= target score
        # that are not in the filtered answer set.
        # No arbitrary -1e6 sentinel is needed.
        ranks = (
            1 + ((scores >= target_scores) & (~filtered[:, None, :])).sum(dim=-1)
        ).double()

        rank_sum += ranks.sum(dim=0).cpu()
        reciprocal_rank_sum += (1.0 / ranks).sum(dim=0).cpu()

        hits1 += (ranks <= 1).double().sum(dim=0).cpu()

        hits3 += (ranks <= 3).double().sum(dim=0).cpu()

        hits10 += (ranks <= 10).double().sum(dim=0).cpu()

        total += n

        if begin == 0 or (begin // batch_size + 1) % 100 == 0 or total == len(queries):
            print(
                f"  processed {total}/{len(queries)}",
                flush=True,
            )

    rows = []

    for i, name in enumerate(NAMES):
        rows.append(
            {
                "strategy": name,
                "queries": total,
                "mr": rank_sum[i].item() / total,
                "mrr": (reciprocal_rank_sum[i].item() / total),
                "hits1": hits1[i].item() / total,
                "hits3": hits3[i].item() / total,
                "hits10": hits10[i].item() / total,
            }
        )

    return rows


def combine_directions(rhs, lhs):
    combined = []

    for r, l in zip(rhs, lhs):
        if r["strategy"] != l["strategy"]:
            raise RuntimeError("Strategy order mismatch")

        count = r["queries"] + l["queries"]

        row = {
            "strategy": r["strategy"],
            "queries": count,
        }

        for key in [
            "mr",
            "mrr",
            "hits1",
            "hits3",
            "hits10",
        ]:
            row[key] = (r[key] * r["queries"] + l[key] * l["queries"]) / count

        combined.append(row)

    return combined


def show_table(title, rows):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print(f"{'Strategy':<25}{'MR':>10}{'MRR':>10}{'H@1':>10}{'H@3':>10}{'H@10':>10}")

    for r in rows:
        print(
            f"{r['strategy']:<25}"
            f"{r['mr']:>10.2f}"
            f"{r['mrr']:>10.4f}"
            f"{r['hits1']:>10.4f}"
            f"{r['hits3']:>10.4f}"
            f"{r['hits10']:>10.4f}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--teacher",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        default="FB15K-237",
    )
    parser.add_argument(
        "--split",
        choices=["valid", "test"],
        default="valid",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=1024,
        help="Number of ORIGINAL triples; 0 means full split",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2022,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    if args.batch_size <= 0 or args.queries < 0:
        parser.error("Invalid batch size or query count")

    args.teacher = args.teacher.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    if not args.teacher.is_file():
        raise FileNotFoundError(args.teacher)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    # Dataset path is relative to src/.
    os.chdir(ROOT / "src")
    dataset = Dataset(args.dataset)

    # Load trusted project checkpoint only.
    teacher = torch.load(
        args.teacher,
        map_location="cuda",
        weights_only=False,
    )

    if not isinstance(teacher, ComplExMDR):
        raise TypeError(f"Expected ComplExMDR, got {type(teacher)}")

    if tuple(teacher.sizes) != tuple(dataset.get_shape()):
        raise ValueError("Teacher checkpoint does not match dataset")

    if not (teacher.modality_split and teacher.fusion_img and teacher.fusion_dscp):
        raise RuntimeError("Teacher requires three modalities")

    teacher.requires_grad_(False)
    teacher.eval()

    original = select_queries(
        dataset.get_examples(args.split),
        args.queries,
        args.seed,
    )

    print("Experiment: L8")
    print("Teacher:", args.teacher)
    print("Dataset:", args.dataset)
    print("Split:", args.split)
    print("Original triples:", len(original))
    print("Evaluation: FILTERED")
    print("Score tie convention: >=")
    print("No optimizer, no training")

    results = {}

    for direction in ["rhs", "lhs"]:
        queries = make_queries(
            original,
            direction,
            dataset.n_predicates,
        )

        print(
            f"\nEvaluating {direction.upper()}: {len(queries)} queries",
            flush=True,
        )

        results[direction] = evaluate_direction(
            teacher,
            queries,
            dataset.to_skip[direction],
            args.batch_size,
            "cuda",
        )

        show_table(
            f"L8 {args.split.upper()} - {direction.upper()}",
            results[direction],
        )

    results["both"] = combine_directions(
        results["rhs"],
        results["lhs"],
    )

    show_table(
        f"L8 {args.split.upper()} - BOTH DIRECTIONS (FILTERED)",
        results["both"],
    )

    args.output.mkdir(parents=True, exist_ok=True)

    csv_path = args.output / "metrics.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "direction",
                "strategy",
                "queries",
                "mr",
                "mrr",
                "hits1",
                "hits3",
                "hits10",
            ],
        )

        writer.writeheader()

        for direction, rows in results.items():
            for row in rows:
                writer.writerow(
                    {
                        "direction": direction,
                        **row,
                    }
                )

    metadata = {
        "experiment": "L8",
        "dataset": args.dataset,
        "split": args.split,
        "teacher": str(args.teacher),
        "original_queries": len(original),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "ranking": "filtered",
        "tie_convention": ">=",
        "results": results,
    }

    json_path = args.output / "results.json"

    json_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nSaved:", csv_path)
    print("Saved:", json_path)


if __name__ == "__main__":
    main()
