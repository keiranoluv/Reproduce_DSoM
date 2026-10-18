#!/usr/bin/env python3
"""Prepare raw WN18 triples from KG-BERT for DSoM."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "kg-bert/data/WN18"
TARGET = ROOT / "src_data/WN18"

SPLITS = {
    "train.tsv": ("train", 141442),
    "dev.tsv": ("valid", 5000),
    "test.tsv": ("test", 5000),
}


def main():
    if TARGET.exists():
        raise FileExistsError(f"Output already exists: {TARGET}")

    prepared = {}

    for filename, (output, expected) in SPLITS.items():
        path = SOURCE / filename
        if not path.is_file():
            raise FileNotFoundError(path)

        triples = []

        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) != 3:
                raise ValueError(f"{filename}:{line_number}: Invalid triple")

            triples.append("\t".join(parts))

        if len(triples) != expected:
            raise ValueError(f"{filename}: expected {expected}, got {len(triples)}")

        prepared[output] = triples

    TARGET.mkdir(parents=True)

    for name, triples in prepared.items():
        (TARGET / name).write_text(
            "\n".join(triples) + "\n",
            encoding="utf-8",
        )
        print(f"{name}: {len(triples)} triples")

    print("WN18 raw preparation: PASS")


if __name__ == "__main__":
    main()
