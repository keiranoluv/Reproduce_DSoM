#!/usr/bin/env python3
"""Offline BERT feature extraction for DSoM MMKB-DB15K.

Protocol matched to the existing MKG-Y/MKG-W reproduction scripts:
bert-base-uncased, penultimate hidden layer, attention-mask mean pooling,
max_length=512, missing-entity N(0,1) with NumPy seed 2022.
"""

import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MMKB-DB15K"
ENT_FILE = DATA / "ent_id"
TEXT_FILE = ROOT.parent / "dataset_audit/MMRNS/MMKB-DB15K/DB15K_description.txt"
OUTPUT = DATA / "dscp_vec.pickle"
CHECKPOINT = DATA / "dscp_features_checkpoint.npz"
MANIFEST = DATA / "dscp_features_manifest.json"
N_ENTITIES, DIM, SEED = 12842, 768, 2022
MODEL = "bert-base-uncased"
PATTERN = re.compile(r'^"(.*)"(?:@[a-zA-Z-]+)?\s*\.\s*$')


def load_entities():
    mapping = {}
    for row_number, line in enumerate(
        ENT_FILE.read_text(encoding="utf-8").splitlines(), 1
    ):
        uri, idx_str = line.split("\t")
        if uri in mapping:
            raise ValueError(f"Duplicate KG entity on line {row_number}: {uri}")
        mapping[uri] = int(idx_str)
    if len(mapping) != N_ENTITIES or set(mapping.values()) != set(range(N_ENTITIES)):
        raise ValueError("Unexpected MMKB-DB15K ent_id mapping")
    return mapping


def load_descriptions():
    texts = {}
    for line_number, line in enumerate(
        TEXT_FILE.read_text(encoding="utf-8").splitlines(), 1
    ):
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed DB15K description line {line_number}")
        uri, literal = parts
        if not uri.startswith("http://dbpedia.org/resource/"):
            raise ValueError(f"Unexpected entity URI on line {line_number}: {uri}")
        match = PATTERN.fullmatch(literal)
        if match is None:
            raise ValueError(
                f"Unexpected description literal on line {line_number}: {literal[:160]}"
            )
        if uri in texts:
            raise ValueError(f"Duplicate DB15K description for {uri}")
        texts[uri] = match.group(1).strip()
    return texts


def digest_inputs():
    hasher = hashlib.sha256()
    for path in (ENT_FILE, TEXT_FILE):
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
    hasher.update(b"bert-base-uncased|hidden_states[-2]|masked_mean|512|seed2022")
    return hasher.hexdigest()


def pool(hidden, mask):
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)


def save_checkpoint(features, next_index, fingerprint):
    temporary = CHECKPOINT.with_name(CHECKPOINT.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            features=features,
            next_index=np.int64(next_index),
            fingerprint=np.array(fingerprint),
        )
    os.replace(temporary, CHECKPOINT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--save-every", type=int, default=100, help="Checkpoint every N batches"
    )
    ap.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = ap.parse_args()
    if args.batch_size <= 0 or args.save_every <= 0:
        ap.error("batch-size and save-every must be positive")

    entities = load_entities()
    texts = load_descriptions()
    available = sorted(
        ((idx, texts[uri]) for uri, idx in entities.items() if texts.get(uri)),
        key=lambda row: row[0],
    )
    print("=== DB15K BERT INPUT AUDIT ===")
    print("Structural entities:", len(entities))
    print("Raw unique descriptions:", len(texts))
    print("Non-empty matched descriptions:", len(available))
    print("Missing/empty descriptions:", N_ENTITIES - len(available))
    print("Expected output shape:", (N_ENTITIES, DIM))
    if len(texts) != 9085 or len(available) != 9085:
        raise RuntimeError("DB15K description coverage mismatch; stop and inspect")
    if args.audit_only:
        print("DB15K BERT INPUT AUDIT: PASS")
        return
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if MANIFEST.exists():
        raise FileExistsError(f"Refusing to overwrite {MANIFEST}")

    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    fingerprint = digest_inputs()
    features = np.random.normal(0.0, 1.0, (N_ENTITIES, DIM)).astype(np.float32)
    start_index = 0
    if CHECKPOINT.exists():
        with np.load(CHECKPOINT, allow_pickle=False) as checkpoint:
            if checkpoint["fingerprint"].item() != fingerprint:
                raise RuntimeError("Input fingerprint changed; refusing resume")
            features = checkpoint["features"].copy()
            start_index = int(checkpoint["next_index"])
        if features.shape != (N_ENTITIES, DIM) or not 0 <= start_index <= len(
            available
        ):
            raise RuntimeError("Invalid BERT resume state")
        print("Resuming at text entity", start_index)

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = (
        AutoModel.from_pretrained(MODEL, output_hidden_states=True).to(device).eval()
    )
    if model.config.hidden_size != DIM:
        raise RuntimeError("Unexpected BERT hidden size")

    with torch.inference_mode():
        for batch_number, start in enumerate(
            tqdm(
                range(start_index, len(available), args.batch_size), desc="DB15K BERT"
            ),
            1,
        ):
            batch = available[start : start + args.batch_size]
            inputs = tokenizer(
                [text for _, text in batch],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            vectors = (
                pool(outputs.hidden_states[-2], inputs["attention_mask"])
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            if vectors.shape != (len(batch), DIM):
                raise RuntimeError("Unexpected BERT batch shape")
            for (idx, _), vector in zip(batch, vectors):
                features[idx] = vector
            if batch_number % args.save_every == 0:
                save_checkpoint(
                    features, min(start + args.batch_size, len(available)), fingerprint
                )

    if features.shape != (N_ENTITIES, DIM) or not np.isfinite(features).all():
        raise RuntimeError("Invalid BERT feature matrix")
    save_checkpoint(features, len(available), fingerprint)
    temporary = OUTPUT.with_name(OUTPUT.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(features, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, OUTPUT)
    manifest = {
        "dataset": "MMKB-DB15K",
        "feature_type": "BERT",
        "encoder": MODEL,
        "representation": "hidden_states[-2]",
        "pooling": "attention_mask_mean",
        "max_length": 512,
        "seed": SEED,
        "text_shape": [N_ENTITIES, DIM],
        "matched_text_entities": len(available),
        "missing_text_entities": N_ENTITIES - len(available),
        "missing_imputation": "np.random.normal(0, 1), seed 2022",
        "input_fingerprint": fingerprint,
        "protocol_note": "Matches the project's MKG-Y/W BERT feature extraction protocol",
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    CHECKPOINT.unlink(missing_ok=True)
    print("DB15K BERT FEATURES: PASS", features.shape)
    print("Output:", OUTPUT)
    print("Manifest:", MANIFEST)


if __name__ == "__main__":
    main()
