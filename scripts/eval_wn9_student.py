#!/usr/bin/env python3
"""Evaluate a locally trained WN9 student checkpoint in raw and filtered settings."""

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--queries", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2000)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.queries < 0 or args.batch_size < 1:
        raise ValueError("queries must be nonnegative and batch-size positive")
    import torch

    sys.path.insert(0, str(ROOT / "src"))
    from models import ComplExMDR
    from wn9_evaluation import evaluate_student

    # Only full-object checkpoints produced by this trusted repository are accepted.
    model = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if not isinstance(model, ComplExMDR) or tuple(model.sizes) != (6555, 18, 6555):
        raise ValueError("Checkpoint is not a WN9 ComplExMDR student")
    model.to(args.device).requires_grad_(False).eval()
    data = ROOT / "data/WN9"
    with (data / "to_skip.pickle").open("rb") as stream:
        filters = pickle.load(stream)
    results = {}
    for split in ("valid", "test"):
        with (data / f"{split}.pickle").open("rb") as stream:
            examples = pickle.load(stream)
        results[split] = evaluate_student(
            model, examples, filters, args.device, args.batch_size, args.queries
        )
    report = dict(
        dataset="WN9",
        checkpoint=str(args.checkpoint.resolve()),
        device=args.device,
        seed=2022,
        query_limit=args.queries,
        tie_rule="pessimistic >=, target excluded",
        results=results,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
