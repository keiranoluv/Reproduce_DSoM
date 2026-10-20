#!/usr/bin/env python3
"""Stage, audit and preprocess the original MMRNS MKG-W structural splits."""

import argparse
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import numpy as np

from src.process_datasets import prepare_dataset

ROOT = Path(__file__).resolve().parents[1]
COUNTS = {"train": 34196, "valid": 4276, "test": 4274}


def digest(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_splits(directory, extension):
    splits = {}
    entities = set()
    relations = set()
    for split, expected in COUNTS.items():
        file = directory / (split + extension)
        triples = []
        with file.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                triple = tuple(line.rstrip("\n").rstrip("\r").split("\t"))
                if len(triple) != 3 or not all(triple):
                    raise ValueError(f"Invalid triple: {file}:{lineno}: {line!r}")
                triples.append(triple)
                entities.update((triple[0], triple[2]))
                relations.add(triple[1])
        if len(triples) != expected or len(set(triples)) != len(triples):
            raise ValueError(f"Unexpected size/duplicates in {file}: {len(triples)}")
        splits[split] = triples
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if set(splits[a]) & set(splits[b]):
            raise ValueError(f"Leakage: {a} overlaps {b}")
    if (len(entities), len(relations)) != (15000, 169):
        raise ValueError(
            f"Unexpected vocabulary: {len(entities)} entities, {len(relations)} relations"
        )
    ent_ids = {ent: idx for idx, ent in enumerate(sorted(entities))}
    rel_ids = {rel: idx for idx, rel in enumerate(sorted(relations))}
    max_ent = max(max(ent_ids[h], ent_ids[t]) for h, _, t in splits["train"])
    max_rel = max(rel_ids[r] for _, r, _ in splits["train"])
    # src/datasets.py uses maxima from train, not the complete vocabulary.
    if max_ent != 14999 or max_rel != 168:
        raise RuntimeError(
            f"Training-only shape inference mismatch: max entity ID={max_ent}, relation ID={max_rel}. "
            "Do not train; inspect src/datasets.py before proceeding."
        )
    return splits


def verify_outputs(stage, output):
    assert output.is_dir(), output
    entities = {}
    relations = {}
    for filename, dest in (("ent_id", entities), ("rel_id", relations)):
        for line in (output / filename).read_text(encoding="utf-8").splitlines():
            entity, idx = line.split("\t")
            dest[entity] = int(idx)
    assert sorted(entities.values()) == list(range(15000))
    assert sorted(relations.values()) == list(range(169))
    for split, expected in COUNTS.items():
        raw = [
            tuple(line.split("\t")) for line in (stage / split).read_text().splitlines()
        ]
        with (output / f"{split}.pickle").open("rb") as f:
            array = pickle.load(f)
        assert array.shape == (expected, 3), split
        expected_array = np.array(
            [[entities[h], relations[r], entities[t]] for h, r, t in raw],
            dtype=np.uint64,
        )
        assert np.array_equal(array, expected_array), split
        print(f"{split}: PASS ({expected} triples)")
    for file in ("to_skip.pickle", "probas.pickle"):
        assert (output / file).exists(), file
    print("MKG-W STRUCTURAL PREPROCESSING: PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mmrns-root", type=Path, default=ROOT.parent / "dataset_audit/MMRNS"
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    source = args.mmrns_root / "MKG-W"
    stage = ROOT / "src_data/MKG-W"
    output = ROOT / "data/MKG-W"

    if args.verify_only:
        parse_splits(stage, "")
        verify_outputs(stage, output)
        return

    parse_splits(source, ".txt")
    ent_links = source / "ent_links"
    if not ent_links.is_file():
        ent_links = args.mmrns_root / "MKG-W-Y-raw_data/MKG-W/ent_links"
    if not ent_links.is_file():
        raise FileNotFoundError("MKG-W ent_links missing from both MMRNS locations")

    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing {output}; use --verify-only"
        )
    if stage.exists():
        for split in COUNTS:
            if digest(stage / split) != digest(source / (split + ".txt")):
                raise RuntimeError(f"Staged {split} differs from upstream raw file")
        if digest(stage / "ent_links") != digest(ent_links):
            raise RuntimeError("Staged ent_links differs from upstream")
    else:
        stage.mkdir(parents=True)
        for split in COUNTS:
            shutil.copy2(source / (split + ".txt"), stage / split)
        shutil.copy2(ent_links, stage / "ent_links")

    print("MKG-W raw source hashes:")
    for split in COUNTS:
        print(split, digest(stage / split))
    print("ent_links", digest(stage / "ent_links"))

    prepare_dataset(str(stage), "MKG-W")
    verify_outputs(stage, output)
    manifest = {
        "dataset": "MKG-W",
        "source": "MMRNS (quqxui/MMRNS)",
        "raw_hashes": {x: digest(stage / x) for x in (*COUNTS.keys(), "ent_links")},
        "splits": COUNTS,
        "entities": 15000,
        "relations": 169,
    }
    (output / "structural_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
