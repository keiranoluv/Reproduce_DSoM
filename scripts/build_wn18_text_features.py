#!/usr/bin/env python3
"""Extract WN18 BERT features aligned with DSoM entity IDs."""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ENT_FILE = ROOT / "data/WN18/ent_id"
TEXT_FILE = ROOT.parent / "kg-bert/data/WN18/entity2text.txt"
OUT_FILE = ROOT / "data/WN18/dscp_vec.pickle"
CHECKPOINT = ROOT / "data/WN18/dscp_features_checkpoint.npz"

MODEL_NAME = "bert-base-uncased"
SEED = 2022
MAX_LENGTH = 512
HIDDEN_SIZE = 768
EXPECTED_ENTITIES = 40943


def load_entities():
    mapping = {}

    for line in ENT_FILE.read_text(encoding="utf-8").splitlines():
        entity, idx = line.split("\t")
        mapping[entity] = int(idx)

    assert len(mapping) == EXPECTED_ENTITIES
    assert sorted(mapping.values()) == list(range(len(mapping)))

    return mapping


def load_texts():
    descriptions = {}

    for line in TEXT_FILE.read_text(encoding="utf-8").splitlines():
        if "\t" not in line:
            raise ValueError(f"Invalid description: {line[:100]}")

        entity, text = line.split("\t", 1)
        text = text.replace("\\n", " ").strip()

        if entity in descriptions:
            raise ValueError(f"Duplicate entity: {entity}")

        descriptions[entity] = text

    return descriptions


def mean_pool(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(hidden)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def save_checkpoint(features, next_index):
    tmp = CHECKPOINT.with_name(CHECKPOINT.name + ".tmp")

    with open(tmp, "wb") as f:
        np.savez(
            f,
            features=features,
            next_index=np.int64(next_index),
        )

    os.replace(tmp, CHECKPOINT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    args = parser.parse_args()

    if args.batch_size <= 0 or args.save_every <= 0:
        parser.error("batch-size and save-every must be positive")

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        torch.cuda.manual_seed_all(SEED)

    entities = load_entities()
    descriptions = load_texts()

    ordered = sorted(entities.items(), key=lambda x: x[1])

    missing = [(entity, idx) for entity, idx in ordered if not descriptions.get(entity)]

    available = [
        (entity, idx, descriptions[entity])
        for entity, idx in ordered
        if descriptions.get(entity)
    ]

    print("Entities:", len(entities))
    print("Available texts:", len(available))
    print("Missing texts:", len(missing))

    if OUT_FILE.exists():
        raise FileExistsError(f"Output already exists: {OUT_FILE}")

    if CHECKPOINT.exists():
        with np.load(CHECKPOINT) as state:
            features = state["features"].copy()
            start_index = int(state["next_index"])

        if features.shape != (EXPECTED_ENTITIES, HIDDEN_SIZE):
            raise ValueError("Checkpoint feature shape mismatch")

        if not 0 <= start_index <= len(available):
            raise ValueError("Invalid checkpoint next_index")

        print("Resuming at entity:", start_index)
    else:
        features = np.random.normal(
            loc=0.0,
            scale=1.0,
            size=(EXPECTED_ENTITIES, HIDDEN_SIZE),
        ).astype(np.float32)
        start_index = 0

    device = torch.device(args.device)

    print("Loading tokenizer and BERT...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
    ).to(device)

    model.eval()

    assert model.config.hidden_size == HIDDEN_SIZE

    processed_batches = 0

    with torch.inference_mode():
        progress = tqdm(
            range(start_index, len(available), args.batch_size),
            desc="WN18 BERT",
        )

        for start in progress:
            batch = available[start : start + args.batch_size]
            texts = [item[2] for item in batch]

            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            inputs = {key: value.to(device) for key, value in inputs.items()}

            outputs = model(**inputs)

            pooled = mean_pool(
                outputs.hidden_states[-2],
                inputs["attention_mask"],
            )

            vectors = pooled.cpu().numpy().astype(np.float32)

            for (_, idx, _), vector in zip(batch, vectors):
                features[idx] = vector

            processed_batches += 1
            next_index = min(
                start + args.batch_size,
                len(available),
            )

            if processed_batches % args.save_every == 0:
                save_checkpoint(features, next_index)

    assert features.shape == (EXPECTED_ENTITIES, HIDDEN_SIZE)
    assert np.isfinite(features).all()

    tmp_out = OUT_FILE.with_name(OUT_FILE.name + ".tmp")
    with open(tmp_out, "wb") as f:
        pickle.dump(features, f)

    os.replace(tmp_out, OUT_FILE)

    print("Saved:", OUT_FILE)
    print("Shape:", features.shape)
    print("dtype:", features.dtype)
    print("WN18 BERT FEATURES: PASS")

    if CHECKPOINT.exists():
        CHECKPOINT.unlink()


if __name__ == "__main__":
    main()
