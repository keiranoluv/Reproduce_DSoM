#!/usr/bin/env python3
"""Build MKG-Y BERT text features for DSoM Table 1."""

import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/MKG-Y"
ENT_FILE = DATA / "ent_id"

TEXT_FILE = (
    ROOT.parent / "dataset_audit/MMRNS/MKG-W-Y-raw_data" / "MKG-Y_ent2descreption.txt"
)

OUTPUT = DATA / "dscp_vec.pickle"
CHECKPOINT = DATA / "dscp_features_checkpoint.npz"
MANIFEST = DATA / "dscp_features_manifest.json"

MODEL_NAME = "bert-base-uncased"
PREFIX = "http://dbpedia.org/resource/"
PATTERN = re.compile(r'^"(.*)"(?:@[a-zA-Z-]+)?\s*\.\s*$')

N_ENTITIES = 15000
DIM = 768
SEED = 2022


def load_entities():
    entries = []

    with ENT_FILE.open(encoding="utf-8") as f:
        for line in f:
            entity, idx = line.rstrip("\n").split("\t")
            entries.append((entity, int(idx)))

    entries.sort(key=lambda x: x[1])

    assert len(entries) == N_ENTITIES
    assert len({e for e, _ in entries}) == N_ENTITIES
    assert [idx for _, idx in entries] == list(range(N_ENTITIES))

    return entries


def load_descriptions():
    descriptions = {}

    with TEXT_FILE.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            parts = line.strip().split(maxsplit=1)

            if len(parts) != 2:
                raise ValueError(f"Invalid row {line_no}")

            uri, literal = parts

            if not uri.startswith(PREFIX):
                raise ValueError(f"Invalid URI at row {line_no}")

            match = PATTERN.match(literal)

            if match is None:
                raise ValueError(f"Invalid literal at row {line_no}")

            entity = uri[len(PREFIX) :]
            text = match.group(1).strip()

            if entity in descriptions:
                raise ValueError(f"Duplicate entity: {entity}")

            descriptions[entity] = text

    return descriptions


def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).type_as(hidden)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def save_checkpoint(features, next_index, fingerprint):
    tmp = CHECKPOINT.with_name(CHECKPOINT.name + ".tmp")

    with tmp.open("wb") as f:
        np.savez(
            f,
            features=features,
            next_index=np.int64(next_index),
            fingerprint=np.array(fingerprint),
        )

    os.replace(tmp, CHECKPOINT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    args = parser.parse_args()

    if args.batch_size < 1 or args.save_every < 1:
        parser.error("Invalid batch-size or save-every")

    if OUTPUT.exists():
        raise FileExistsError(f"Output exists: {OUTPUT}")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if args.device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    entries = load_entities()
    descriptions = load_descriptions()

    available = [
        (entity, idx, descriptions[entity])
        for entity, idx in entries
        if descriptions.get(entity)
    ]

    assert len(descriptions) == 12316
    assert len(available) == 12316

    fingerprint = hashlib.sha256(
        ENT_FILE.read_bytes() + TEXT_FILE.read_bytes()
    ).hexdigest()

    features = np.random.normal(0.0, 1.0, (N_ENTITIES, DIM)).astype(np.float32)

    start_index = 0

    if CHECKPOINT.exists():
        print("Resuming checkpoint...", flush=True)

        with np.load(CHECKPOINT, allow_pickle=False) as state:
            if str(state["fingerprint"].item()) != fingerprint:
                raise RuntimeError("Checkpoint input mismatch")

            features = state["features"].copy()
            start_index = int(state["next_index"])

        assert features.shape == (N_ENTITIES, DIM)
        assert 0 <= start_index <= len(available)

    print("Entities:", N_ENTITIES)
    print("Available texts:", len(available))
    print("Missing texts:", N_ENTITIES - len(available))

    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = (
        AutoModel.from_pretrained(
            MODEL_NAME,
            output_hidden_states=True,
        )
        .to(device)
        .eval()
    )

    assert model.config.hidden_size == DIM

    batch_count = 0

    with torch.inference_mode():
        for start in tqdm(
            range(start_index, len(available), args.batch_size),
            desc="MKG-Y BERT",
        ):
            batch = available[start : start + args.batch_size]

            inputs = tokenizer(
                [row[2] for row in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model(**inputs)

            vectors = (
                mean_pool(
                    outputs.hidden_states[-2],
                    inputs["attention_mask"],
                )
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            for (_, idx, _), vector in zip(batch, vectors):
                features[idx] = vector

            batch_count += 1
            next_index = min(
                start + args.batch_size,
                len(available),
            )

            if batch_count % args.save_every == 0:
                save_checkpoint(features, next_index, fingerprint)

    assert features.shape == (N_ENTITIES, DIM)
    assert np.isfinite(features).all()

    save_checkpoint(features, len(available), fingerprint)

    tmp = OUTPUT.with_name(OUTPUT.name + ".tmp")

    with tmp.open("wb") as f:
        pickle.dump(
            features,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(tmp, OUTPUT)

    manifest = {
        "dataset": "MKG-Y",
        "encoder": MODEL_NAME,
        "representation": "hidden_states[-2]",
        "pooling": "attention_mask_mean",
        "max_length": 512,
        "seed": SEED,
        "feature_shape": [N_ENTITIES, DIM],
        "available_texts": len(available),
        "missing_texts": N_ENTITIES - len(available),
        "input_fingerprint": fingerprint,
    }

    with MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    CHECKPOINT.unlink(missing_ok=True)

    print("\n=== MKG-Y TEXT FEATURES ===")
    print("Output:", OUTPUT)
    print("Shape:", features.shape)
    print("dtype:", features.dtype)
    print("MKG-Y TEXT FEATURES: PASS")


if __name__ == "__main__":
    main()
