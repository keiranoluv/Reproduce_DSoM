#!/usr/bin/env python3
"""
L7: Frozen Teacher Calibration & Ranking Audit.

- Same pretrained teacher for every split.
- No training or optimizer.
- Original + reciprocal queries.
- Unfiltered CE and unfiltered ranking.
- Temperature sensitivity: T=1, 2, 4.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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


def load_queries(dataset, split, limit, seed):
    if split == "train":
        data = dataset.get_train()
    elif split == "valid":
        data = dataset.get_valid()
    elif split == "test":
        # Dataset stores original triples for test.
        # Build reciprocal triples explicitly.
        import numpy as np

        original = dataset.get_examples("test")
        reverse = original.copy()
        reverse[:, 0] = original[:, 2]
        reverse[:, 2] = original[:, 0]
        reverse[:, 1] += dataset.n_predicates // 2
        data = np.vstack((original, reverse))
    else:
        raise ValueError(split)

    queries = torch.from_numpy(data.astype("int64"))

    if 0 < limit < len(queries):
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(
            len(queries),
            generator=generator,
        )[:limit]
        queries = queries[indices]

    return queries


@torch.inference_mode()
def evaluate(teacher, queries, batch_size, temperatures):
    device = "cuda"
    strategy = STRATEGIES.to(device)
    n_strategies = len(NAMES)

    metrics = {
        "ce": torch.zeros(n_strategies, dtype=torch.float64),
        "mrr": torch.zeros(n_strategies, dtype=torch.float64),
        "hits1": torch.zeros(n_strategies, dtype=torch.float64),
        "hits10": torch.zeros(n_strategies, dtype=torch.float64),
        "confidence": torch.zeros(n_strategies, dtype=torch.float64),
        "wrong_confidence_sum": torch.zeros(n_strategies, dtype=torch.float64),
        "wrong_count": torch.zeros(n_strategies, dtype=torch.float64),
        "true_probability": torch.zeros(n_strategies, dtype=torch.float64),
    }

    for temp in temperatures:
        metrics[f"ce_t{temp:g}"] = torch.zeros(n_strategies, dtype=torch.float64)

    # ECE: Expected Calibration Error
    # (sai số hiệu chỉnh kỳ vọng), 10 bins.
    n_bins = 10
    ece_count = torch.zeros((n_strategies, n_bins), dtype=torch.float64)
    ece_conf = torch.zeros_like(ece_count)
    ece_correct = torch.zeros_like(ece_count)

    total = 0

    for begin in range(0, len(queries), batch_size):
        batch = queries[begin : begin + batch_size].to(device)
        truth = batch[:, 2]

        out = teacher.forward(batch)

        stack = torch.stack(
            [out[0], out[2], out[4]],
            dim=-1,
        )

        # [batch, 7, entities]
        logits = torch.einsum(
            "bem,sm->bse",
            stack,
            strategy,
        )

        bsz, n_strategies, n_entities = logits.shape

        flat = logits.reshape(-1, n_entities)

        target = truth[:, None].expand(-1, n_strategies).reshape(-1)

        # Per-query CE, using original logits.
        log_norm = torch.logsumexp(flat, dim=-1)
        correct_score = flat.gather(1, target[:, None]).squeeze(1)

        ce = (log_norm - correct_score).reshape(bsz, n_strategies)

        if not torch.isfinite(ce).all():
            raise RuntimeError("Non-finite CE")

        # Exact unfiltered rank with deterministic tie handling.
        # Rank = 1 + candidates strictly better than target
        #        + smaller entity IDs tied with target.
        scores = logits
        true_scores = scores.gather(
            2,
            truth[:, None, None].expand(-1, n_strategies, 1),
        )

        entity_ids = torch.arange(n_entities, device=device).view(1, 1, -1)

        better = scores > true_scores
        tied_before = (scores == true_scores) & (entity_ids < truth[:, None, None])

        ranks = (1 + better.sum(dim=-1) + tied_before.sum(dim=-1)).float()

        # Top-1 confidence, before temperature scaling.
        max_logit, prediction = flat.max(dim=-1)
        confidence = (max_logit - log_norm).exp()

        correct = prediction == target
        wrong = ~correct

        true_prob = (-ce.reshape(-1)).exp()

        def accumulate(key, values):
            values = values.reshape(bsz, n_strategies)
            metrics[key] += values.double().sum(0).cpu()

        accumulate("ce", ce)
        accumulate("mrr", 1.0 / ranks)
        accumulate("hits1", (ranks == 1).float())
        accumulate("hits10", (ranks <= 10).float())
        accumulate("confidence", confidence)
        accumulate("true_probability", true_prob)
        accumulate(
            "wrong_confidence_sum",
            confidence * wrong,
        )
        accumulate("wrong_count", wrong.float())

        for temp in temperatures:
            scaled = flat / temp

            scaled_ce = torch.logsumexp(scaled, dim=-1) - scaled.gather(
                1, target[:, None]
            ).squeeze(-1)

            accumulate(f"ce_t{temp:g}", scaled_ce)

        # ECE bins for top-1 confidence.
        bin_id = torch.clamp(
            (confidence * n_bins).long(),
            max=n_bins - 1,
        ).reshape(bsz, n_strategies)

        c = confidence.reshape(bsz, n_strategies)
        hit = correct.float().reshape(bsz, n_strategies)

        for j in range(n_strategies):
            ids = bin_id[:, j]

            ece_count[j] += torch.bincount(
                ids.cpu(),
                minlength=n_bins,
            ).double()

            ece_conf[j] += torch.bincount(
                ids.cpu(),
                weights=c[:, j].double().cpu(),
                minlength=n_bins,
            )

            ece_correct[j] += torch.bincount(
                ids.cpu(),
                weights=hit[:, j].double().cpu(),
                minlength=n_bins,
            )

        total += bsz

        if begin == 0 or (begin // batch_size + 1) % 100 == 0 or total == len(queries):
            print(
                f"  processed {total}/{len(queries)}",
                flush=True,
            )

    results = []

    for j, name in enumerate(NAMES):
        error = 0.0

        for k in range(n_bins):
            count = ece_count[j, k].item()

            if count == 0:
                continue

            avg_conf = ece_conf[j, k].item() / count
            accuracy = ece_correct[j, k].item() / count

            error += (count / total) * abs(avg_conf - accuracy)

        n_wrong = metrics["wrong_count"][j].item()

        row = {
            "strategy": name,
            "queries": total,
            "ce": metrics["ce"][j].item() / total,
            "mrr": metrics["mrr"][j].item() / total,
            "hits1": metrics["hits1"][j].item() / total,
            "hits10": metrics["hits10"][j].item() / total,
            "mean_confidence": (metrics["confidence"][j].item() / total),
            "mean_wrong_confidence": (
                metrics["wrong_confidence_sum"][j].item() / n_wrong if n_wrong else 0.0
            ),
            "mean_true_probability": (metrics["true_probability"][j].item() / total),
            "ece10": error,
        }

        for temp in temperatures:
            row[f"ce_t{temp:g}"] = metrics[f"ce_t{temp:g}"][j].item() / total

        results.append(row)

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--dataset", default="FB15K-237")
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test", "all"],
        default="valid",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=2048,
        help="Queries per split; 0 evaluates all",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    if args.batch_size <= 0 or args.queries < 0:
        parser.error("Invalid batch-size or queries")

    args.teacher = args.teacher.expanduser().resolve()
    args.output = args.output.expanduser().resolve()

    if not args.teacher.is_file():
        raise FileNotFoundError(args.teacher)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    os.chdir(ROOT / "src")
    dataset = Dataset(args.dataset)

    # Only load a trusted checkpoint.
    teacher = torch.load(
        args.teacher,
        map_location="cuda",
        weights_only=False,
    )

    if not isinstance(teacher, ComplExMDR):
        raise TypeError("Expected ComplExMDR")

    if tuple(teacher.sizes) != tuple(dataset.get_shape()):
        raise ValueError("Teacher/dataset shape mismatch")

    teacher.requires_grad_(False)
    teacher.eval()

    temperatures = [1.0, 2.0, 4.0]

    splits = ["train", "valid", "test"] if args.split == "all" else [args.split]

    args.output.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for split_index, split in enumerate(splits):
        queries = load_queries(
            dataset,
            split,
            args.queries,
            args.seed + split_index,
        )

        print(
            f"\n[{split.upper()}] Queries={len(queries)}",
            flush=True,
        )

        rows = evaluate(
            teacher,
            queries,
            args.batch_size,
            temperatures,
        )

        print(
            f"{'Strategy':<25}"
            f"{'CE':>8}"
            f"{'MRR':>8}"
            f"{'H@1':>8}"
            f"{'Conf':>8}"
            f"{'ECE':>8}"
            f"{'CE T=2':>9}"
            f"{'CE T=4':>9}"
        )

        for row in rows:
            row["split"] = split
            all_rows.append(row)

            print(
                f"{row['strategy']:<25}"
                f"{row['ce']:>8.4f}"
                f"{row['mrr']:>8.4f}"
                f"{row['hits1']:>8.4f}"
                f"{row['mean_confidence']:>8.4f}"
                f"{row['ece10']:>8.4f}"
                f"{row['ce_t2']:>9.4f}"
                f"{row['ce_t4']:>9.4f}"
            )

    csv_path = args.output / "metrics.csv"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nSaved:", csv_path)
    print("Ranking metrics are UNFILTERED.")
    print("No model parameters were updated.")


if __name__ == "__main__":
    main()
