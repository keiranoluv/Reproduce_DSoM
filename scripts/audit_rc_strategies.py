#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from datasets import Dataset
from kd import KD_RL


def audit(distiller, queries, batch_size):
    device = "cuda"
    strategy = distiller.strategy.to(device)

    n_strategies = strategy.shape[0]

    chosen_counts = torch.zeros(n_strategies, dtype=torch.long)
    oracle_counts = torch.zeros(n_strategies, dtype=torch.long)

    total = 0
    matches = 0

    policy_ce_total = 0.0
    oracle_ce_total = 0.0
    structural_ce_total = 0.0
    student_ce_total = 0.0

    policy_reward_positive = 0
    oracle_reward_positive = 0
    structural_reward_positive = 0

    entropy_total = 0.0
    regret_total = 0.0
    structural_regret_total = 0.0

    for begin in range(0, len(queries), batch_size):
        batch = queries[begin : begin + batch_size].to(device)
        truth = batch[:, 2]
        n = len(batch)

        with torch.no_grad():
            student_logits, _ = distiller.student(batch)

            teacher_out = distiller.teacher.forward(batch)

            preds_stack = torch.stack(
                [
                    teacher_out[0],
                    teacher_out[2],
                    teacher_out[4],
                ],
                dim=-1,
            )

            probs = distiller.policy_probs(preds_stack)
            chosen = probs.argmax(dim=-1)

            # Every strategy is evaluated on the same queries.
            # Shape: [batch, strategies, entities]
            candidate_logits = torch.einsum(
                "bem,sm->bse",
                preds_stack,
                strategy,
            )

            target_logits = candidate_logits.gather(
                dim=2,
                index=truth[:, None, None].expand(-1, n_strategies, 1),
            ).squeeze(-1)

            candidate_ce = torch.logsumexp(candidate_logits, dim=-1) - target_logits

            student_ce = F.cross_entropy(
                student_logits,
                truth,
                reduction="none",
            )

            oracle_ce, oracle = candidate_ce.min(dim=-1)

            policy_ce = candidate_ce.gather(1, chosen[:, None]).squeeze(1)

            structural_ce = candidate_ce[:, 0]

            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)

            for idx in range(n_strategies):
                chosen_counts[idx] += (chosen == idx).sum().cpu()
                oracle_counts[idx] += (oracle == idx).sum().cpu()

            total += n
            matches += (chosen == oracle).sum().item()

            policy_ce_total += policy_ce.sum().item()
            oracle_ce_total += oracle_ce.sum().item()
            structural_ce_total += structural_ce.sum().item()
            student_ce_total += student_ce.sum().item()

            policy_reward_positive += (policy_ce < student_ce).sum().item()

            oracle_reward_positive += (oracle_ce < student_ce).sum().item()

            structural_reward_positive += (structural_ce < student_ce).sum().item()

            entropy_total += entropy.sum().item()

            regret_total += (policy_ce - oracle_ce).sum().item()

            structural_regret_total += (structural_ce - oracle_ce).sum().item()

    print(f"\nQueries: {total}")
    print(f"Policy mean entropy: {entropy_total / total:.6f}")
    print(f"Policy-oracle agreement: {matches / total:.4%}")

    print("\nMean Cross-Entropy (lower is better)")
    print(f"  Student:         {student_ce_total / total:.6f}")
    print(f"  Policy argmax:   {policy_ce_total / total:.6f}")
    print(f"  Structural-only: {structural_ce_total / total:.6f}")
    print(f"  Oracle strategy: {oracle_ce_total / total:.6f}")

    print("\nMean regret relative to per-query oracle")
    print(f"  Policy:     {regret_total / total:.6f}")
    print(f"  Structural: {structural_regret_total / total:.6f}")

    print("\nPositive reward rate (+1 instead of -10)")
    print(f"  Policy:     {policy_reward_positive / total:.4%}")
    print(f"  Structural: {structural_reward_positive / total:.4%}")
    print(f"  Oracle:     {oracle_reward_positive / total:.4%}")

    print("\nStrategy distribution")
    names = [
        "Structural",
        "Image",
        "Description",
        "Structural+Image",
        "Structural+Description",
        "Image+Description",
        "All three",
    ]

    for idx in range(n_strategies):
        print(
            f"  {idx} {names[idx]:24s} "
            f"policy={chosen_counts[idx].item() / total:7.2%} "
            f"oracle={oracle_counts[idx].item() / total:7.2%}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        default="FB15K-237",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=1024,
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

    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    torch.manual_seed(args.seed)

    # dataset.py uses ../data relative to src/
    import os

    os.chdir(ROOT / "src")

    dataset = Dataset(args.dataset)

    # Load only a checkpoint generated by this trusted project.
    distiller = torch.load(
        args.checkpoint.resolve(),
        map_location="cuda",
        weights_only=False,
    )

    if not isinstance(distiller, KD_RL):
        raise TypeError(f"Expected KD_RL distiller checkpoint, got {type(distiller)}")

    distiller.eval()

    if distiller.strategy.shape != (7, 3):
        raise ValueError("Expected seven 3-modality strategies")

    valid = torch.from_numpy(dataset.get_valid().astype("int64"))

    # Include both original and reciprocal queries.
    if args.queries > 0 and args.queries < len(valid):
        indices = torch.randperm(len(valid))[: args.queries]
        valid = valid[indices]

    print("Checkpoint:", args.checkpoint)
    print("Dataset:", args.dataset)
    print("Split: VALID (including reciprocal queries)")
    print("Selection: policy argmax, not stochastic sampling")

    audit(distiller, valid, args.batch_size)


if __name__ == "__main__":
    main()
