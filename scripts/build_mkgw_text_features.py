#!/usr/bin/env python3
"""BERT text feature extraction for MKG-W, aligned by MMRNS ent_links."""

import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np

from scripts.mkgw_alignment import (
    DATA,
    ENT_FILE,
    LINKS_FILE,
    ROOT,
    dbpedia_key,
    load_alignment,
)

TEXT_FILE = (
    ROOT.parent / "dataset_audit/MMRNS/MKG-W-Y-raw_data/MKG-W_ent2descreption.txt"
)
OUTPUT = DATA / "dscp_vec.pickle"
CHECKPOINT = DATA / "dscp_features_checkpoint.npz"
MANIFEST = DATA / "dscp_features_manifest.json"
MODEL = "bert-base-uncased"
N, DIM, SEED = 15000, 768, 2022
PATTERN = re.compile(r'^"(.*)"(?:@[a-zA-Z-]+)?\s*\.\s*$')


def parse_descriptions():
    texts = {}
    for number, line in enumerate(
        TEXT_FILE.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Invalid text row {number}: {line[:160]}")
        uri, literal = parts
        key = dbpedia_key(uri)
        match = PATTERN.match(literal)
        if key is None or match is None:
            raise ValueError(
                f"Unexpected description syntax at {TEXT_FILE}:{number}: {line[:160]}"
            )
        if key in texts:
            raise ValueError(f"Duplicate description for {key}")
        texts[key] = match.group(1).strip()
    return texts


def mean_pool(hidden, mask):
    mask = mask.unsqueeze(-1).type_as(hidden)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def save_checkpoint(features, next_index, fingerprint):
    temp = CHECKPOINT.with_name(CHECKPOINT.name + ".tmp")
    with temp.open("wb") as f:
        np.savez(
            f,
            features=features,
            next_index=np.int64(next_index),
            fingerprint=np.array(fingerprint),
        )
    os.replace(temp, CHECKPOINT)


def digest_inputs():
    h = hashlib.sha256()
    for path in (ENT_FILE, LINKS_FILE, TEXT_FILE):
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    h.update(b"BERT|hidden_states[-2]|attention_mask_mean|max_length=512|seed=2022")
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1 or args.save_every < 1:
        parser.error("batch-size and save-every must be positive")

    _, keys = load_alignment()
    texts = parse_descriptions()
    available = [
        (idx, texts[key])
        for idx, key in enumerate(keys)
        if key is not None and texts.get(key)
    ]
    print("=== MKG-W TEXT ALIGNMENT ===")
    print("Raw descriptions:", len(texts))
    print("Matched entities:", len(available))
    print("Missing descriptions:", N - len(available))
    print("Output dimension:", DIM)
    if len(available) < 13500:
        raise RuntimeError(
            "Description coverage is unexpectedly low; inspect ent_links before encoding"
        )
    if args.audit_only:
        print("MKG-W TEXT ALIGNMENT: PASS")
        return

    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    fingerprint = digest_inputs()
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(SEED)
    features = np.random.normal(0, 1, (N, DIM)).astype(np.float32)
    start = 0
    if CHECKPOINT.exists():
        with np.load(CHECKPOINT, allow_pickle=False) as cp:
            if str(cp["fingerprint"].item()) != fingerprint:
                raise RuntimeError("Checkpoint inputs changed. Refusing unsafe resume")
            features, start = cp["features"].copy(), int(cp["next_index"])
        if features.shape != (N, DIM) or not (0 <= start <= len(available)):
            raise RuntimeError("Invalid checkpoint dimensions")
        print("Resuming text encoding at", start)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = (
        AutoModel.from_pretrained(MODEL, output_hidden_states=True).to(device).eval()
    )
    if model.config.hidden_size != DIM:
        raise RuntimeError("Unexpected BERT feature dimension")

    with torch.inference_mode():
        for batch_no, pos in enumerate(
            tqdm(range(start, len(available), args.batch_size), desc="MKG-W BERT"), 1
        ):
            batch = available[pos : pos + args.batch_size]
            inputs = tokenizer(
                [text for _, text in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            output = model(**inputs)
            vectors = (
                mean_pool(output.hidden_states[-2], inputs["attention_mask"])
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            if vectors.shape != (len(batch), DIM):
                raise RuntimeError("Invalid feature shape")
            for (idx, _), vec in zip(batch, vectors):
                features[idx] = vec
            if batch_no % args.save_every == 0:
                save_checkpoint(
                    features, min(pos + args.batch_size, len(available)), fingerprint
                )

    if not np.isfinite(features).all():
        raise RuntimeError("Non-finite text feature")
    save_checkpoint(features, len(available), fingerprint)
    tmp = OUTPUT.with_name(OUTPUT.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, OUTPUT)
    manifest = {
        "dataset": "MKG-W",
        "encoder": MODEL,
        "representation": "hidden_states[-2]",
        "pooling": "attention_mask_mean",
        "max_length": 512,
        "seed": SEED,
        "feature_shape": [N, DIM],
        "available_texts": len(available),
        "missing_texts": N - len(available),
        "input_fingerprint": fingerprint,
        "alignment": "structural Wikidata URI -> ent_links DBpedia resource key -> raw description",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    CHECKPOINT.unlink(missing_ok=True)
    print("MKG-W TEXT FEATURES: PASS", features.shape, "available=", len(available))


if __name__ == "__main__":
    main()
