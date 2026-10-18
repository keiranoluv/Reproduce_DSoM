#!/usr/bin/env python3
"""Preprocess WN18 raw triples using the DSoM pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from process_datasets import prepare_dataset


def main():
    source = ROOT / "src_data" / "WN18"
    target = ROOT / "data" / "WN18"

    # Step 1: Validate raw inputs.
    for split in ("train", "valid", "test"):
        path = source / split

        if not path.is_file():
            raise FileNotFoundError(f"Missing input file: {path}")

    # Step 2: Prevent overwriting processed data.
    if target.exists():
        raise FileExistsError(f"Processed dataset already exists: {target}")

    print("===== WN18 Preprocessing =====")
    print(f"Input : {source}")
    print(f"Output: {target}")

    # Step 3: Use the existing DSoM preprocessing.
    prepare_dataset(str(source), "WN18")

    print("WN18 preprocessing: PASS")


if __name__ == "__main__":
    main()
