#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

os.chdir(ROOT / "src")
sys.path.insert(0, str(ROOT / "src"))

from datasets import Dataset
from models import ComplExMDR


def evaluate(dataset, model, split):
    mrs, mrrs, hits = dataset.eval(
        model,
        split=split,
        n_queries=-1,
        missing_eval="both",
        at=(1, 3, 10),
    )

    return {
        "MR": (mrs["lhs"] + mrs["rhs"]) / 2,
        "MRR": (mrrs["lhs"] + mrrs["rhs"]) / 2,
        "Hits@1": (hits["lhs"][0].item() + hits["rhs"][0].item()) / 2,
        "Hits@3": (hits["lhs"][1].item() + hits["rhs"][1].item()) / 2,
        "Hits@10": (hits["lhs"][2].item() + hits["rhs"][2].item()) / 2,
    }


def main():
    directory = ROOT / "runs/student/FB15K-237" / "l9_fixed_structural_description"

    checkpoints = sorted((directory / "checkpoints").glob("*/student-*.pth"))

    if len(checkpoints) != 1:
        raise RuntimeError(
            f"Expected exactly one Student checkpoint; "
            f"found {len(checkpoints)}: {checkpoints}"
        )

    dataset = Dataset("FB15K-237")

    # Trusted project checkpoint only.
    model = torch.load(
        checkpoints[0],
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(model, ComplExMDR):
        raise TypeError(f"Unexpected model: {type(model)}")

    if tuple(model.sizes) != tuple(dataset.get_shape()):
        raise ValueError("Student/dataset shape mismatch")

    model = model.cuda()
    model.eval()

    results = {
        "experiment": "L9",
        "teacher_strategy": "structural_description",
        "checkpoint": str(checkpoints[0]),
        "evaluation": "full_filtered_ranking",
        "splits": {},
    }

    with torch.inference_mode():
        for split in ["valid", "test"]:
            metrics = evaluate(dataset, model, split)

            results["splits"][split] = metrics

            print(f"\n[{split.upper()}]")
            print(
                f"MR={metrics['MR']:.4f} "
                f"MRR={metrics['MRR']:.6f} "
                f"H@1={metrics['Hits@1']:.4%} "
                f"H@3={metrics['Hits@3']:.4%} "
                f"H@10={metrics['Hits@10']:.4%}",
                flush=True,
            )

    output = directory / "full_evaluation.json"
    output.write_text(json.dumps(results, indent=2))

    print("\nSaved:", output)


if __name__ == "__main__":
    main()
