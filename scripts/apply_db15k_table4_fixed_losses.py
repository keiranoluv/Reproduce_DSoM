#!/usr/bin/env python3
"""Add fixed-teacher standard KD and DKD modes to the DSoM fork.

Run from repo root (or any cwd):
  python scripts/apply_db15k_table4_fixed_losses.py --repo /workspace/workspace/DSoM

Guards ensure exactly the reviewed anchors are modified; idempotent.
"""

import argparse
import ast
import sys
from pathlib import Path

LEARN_OLD = """if args.kd_loss_type == "fixed_ndkd":
    if args.teacher_strategy == "rc":
        raise ValueError("fixed_ndkd requires a fixed teacher_strategy")"""
LEARN_NEW = """if args.kd_loss_type in ("fixed_ndkd", "fixed_kd", "fixed_dkd"):
    if args.teacher_strategy == "rc":
        raise ValueError(f"{args.kd_loss_type} requires a fixed teacher_strategy")"""

OPT_OLD = """                    elif self.kd_loss == "fixed_ndkd":
                        golden_truths = ["""
OPT_NEW = """                    elif self.kd_loss == "fixed_kd":
                        # Same fixed teacher logits as fixed_ndkd; standard KL only.
                        l_kd = self.kd_weight * kd_loss(
                            preds_student, preds_assistant, self.temperature
                        )
                    elif self.kd_loss == "fixed_dkd":
                        # Target/non-target DKD; unlike NDKD, no neighbor mask.
                        l_kd = self.kd_weight * dkd_loss(
                            preds_student,
                            preds_assistant,
                            truth,
                            self.args.dkd_alpha,
                            self.args.dkd_beta,
                            self.temperature,
                        )
                    elif self.kd_loss == "fixed_ndkd":
                        golden_truths = ["""


def patch(path: Path, original: str, replacement: str, check: bool) -> None:
    source = path.read_text(encoding="utf-8")
    if replacement in source:
        print("ALREADY PATCHED:", path)
        ast.parse(source, filename=str(path))
        return
    if check:
        raise RuntimeError(f"Patch not applied yet: {path}. Run without --check first.")
    count = source.count(original)
    if count != 1:
        raise RuntimeError(
            f"Unexpected source layout in {path}: expected one anchor; found {count}. "
            "Do not apply blindly; inspect the code."
        )
    updated = source.replace(original, replacement, 1)
    ast.parse(updated, filename=str(path))
    path.write_text(updated, encoding="utf-8")
    print("PATCHED:", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument(
        "--check",
        action="store_true",
        help="Verify both source files are already patched (no writes)",
    )
    args = ap.parse_args()
    root = args.repo.resolve()
    learn, optimizer = root / "src/kd_learn.py", root / "src/kd_optimizers.py"
    for file in (learn, optimizer):
        if not file.is_file():
            raise FileNotFoundError(file)
    # Preflight both before writing either (avoid half-applied patches).
    for file, old, new in (
        (learn, LEARN_OLD, LEARN_NEW),
        (optimizer, OPT_OLD, OPT_NEW),
    ):
        data = file.read_text(encoding="utf-8")
        if new not in data and data.count(old) != 1:
            raise RuntimeError(f"Source mismatch: {file}; refusing to patch")
        ast.parse(data.replace(old, new, 1), filename=str(file))
    patch(learn, LEARN_OLD, LEARN_NEW, args.check)
    patch(optimizer, OPT_OLD, OPT_NEW, args.check)
    print("TABLE4_FIXED_LOSSES_PATCH:", "CHECK PASS" if args.check else "PASS")


if __name__ == "__main__":
    main()
