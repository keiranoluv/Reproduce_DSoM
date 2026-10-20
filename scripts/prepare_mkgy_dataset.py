#!/usr/bin/env python3
"""Prepare MKG-Y structural triples for DSoM Table 1 reproduction."""

from pathlib import Path

from src.process_datasets import prepare_dataset

ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = ROOT / "src_data" / "MKG-Y"
OUTPUT_DIR = ROOT / "data" / "MKG-Y"

EXPECTED_COUNTS = {
    "train": 21310,
    "valid": 2665,
    "test": 2663,
}


def main():
    # Validate raw dataset availability and split sizes.
    for split, expected in EXPECTED_COUNTS.items():
        path = SOURCE_DIR / split

        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open(encoding="utf-8") as f:
            actual = sum(1 for _ in f)

        if actual != expected:
            raise ValueError(f"{split}: expected {expected}, got {actual}")

    # Never overwrite existing preprocessing artifacts.
    if OUTPUT_DIR.exists():
        raise FileExistsError(
            f"{OUTPUT_DIR} already exists. Existing artifacts will not be overwritten."
        )

    prepare_dataset(
        path=str(SOURCE_DIR),
        name="MKG-Y",
    )

    print("MKG-Y STRUCTURAL PREPROCESSING COMPLETE")


if __name__ == "__main__":
    main()
