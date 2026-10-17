#!/usr/bin/env python3
"""Full filtered evaluation for Experiment K Student checkpoints."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Existing Dataset uses DATA_PATH = "../data".
os.chdir(ROOT / "src")
sys.path.insert(0, str(ROOT / "src"))

import torch

from datasets import Dataset
from models import ComplExMDR


def metrics_for_split(dataset, model, split):
    # -1 means ALL queries, no random subsampling.
    mrs, mrrs, hits = dataset.eval(
        model,
        split=split,
        n_queries=-1,
        missing_eval="both",
        at=(1, 3, 10),
    )

    return {
        "MR": (mrs["lhs"] + mrs["rhs"]) / 2.0,
        "MRR": (mrrs["lhs"] + mrrs["rhs"]) / 2.0,
        "Hits@1": (hits["lhs"][0].item() + hits["rhs"][0].item()) / 2.0,
        "Hits@3": (hits["lhs"][1].item() + hits["rhs"][1].item()) / 2.0,
        "Hits@10": (hits["lhs"][2].item() + hits["rhs"][2].item()) / 2.0,
        "queries_per_direction": len(dataset.get_examples(split)),
    }


def find_checkpoint(name):
    folder = ROOT / "runs/student/FB15K-237/experiment_k" / name / "checkpoints"

    matches = sorted(folder.glob("*/student-*.pth"))

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one Student checkpoint for {name}, "
            f"found {len(matches)}: {matches}"
        )

    return matches[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "runs/student/FB15K-237/experiment_k/full_evaluation.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this evaluation")

    dataset = Dataset("FB15K-237")

    print("Dataset shape:", dataset.get_shape(), flush=True)
    print(
        "Full validation queries per direction:",
        len(dataset.get_examples("valid")),
        flush=True,
    )
    print(
        "Full test queries per direction:",
        len(dataset.get_examples("test")),
        flush=True,
    )

    results = {
        "dataset": "FB15K-237",
        "protocol": "full_filtered_ranking",
        "eval_queries": -1,
        "checkpoint_selection": "best_validation_MRR",
        "runs": {},
    }

    for name in ["k1_description", "k2_learned_rc"]:
        checkpoint = find_checkpoint(name)

        print(f"\n===== {name} =====", flush=True)
        print("Checkpoint:", checkpoint, flush=True)

        # Load only checkpoints created in our trusted project.
        model = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(model, ComplExMDR):
            raise TypeError(f"Unexpected Student model type: {type(model)}")

        if tuple(model.sizes) != tuple(dataset.get_shape()):
            raise ValueError(f"Checkpoint shape mismatch for {name}")

        model = model.cuda()
        model.eval()

        model_results = {
            "checkpoint": str(checkpoint),
            "splits": {},
        }

        with torch.inference_mode():
            for split in ["valid", "test"]:
                start = time.perf_counter()

                metrics = metrics_for_split(dataset, model, split)
                metrics["evaluation_seconds"] = time.perf_counter() - start

                model_results["splits"][split] = metrics

                print(
                    f"{split}: "
                    f"MR={metrics['MR']:.3f}, "
                    f"MRR={metrics['MRR']:.6f}, "
                    f"H@1={metrics['Hits@1'] * 100:.2f}%, "
                    f"H@3={metrics['Hits@3'] * 100:.2f}%, "
                    f"H@10={metrics['Hits@10'] * 100:.2f}%, "
                    f"time={metrics['evaluation_seconds']:.1f}s",
                    flush=True,
                )

        results["runs"][name] = model_results

        del model
        torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n===== COMPARISON =====", flush=True)

    for split in ["valid", "test"]:
        print(f"\n{split.upper()}")
        print(f"{'Metric':<12} {'K1':>12} {'K2':>12} {'K1 - K2':>12}")

        k1 = results["runs"]["k1_description"]["splits"][split]
        k2 = results["runs"]["k2_learned_rc"]["splits"][split]

        for metric in ["MR", "MRR", "Hits@1", "Hits@3", "Hits@10"]:
            a, b = k1[metric], k2[metric]
            print(f"{metric:<12} {a:12.6f} {b:12.6f} {a - b:+12.6f}")

    print("\nSaved:", output, flush=True)


if __name__ == "__main__":
    main()
