#!/usr/bin/env python3
"""Extract aligned MKG-Y image features for DSoM Table 1."""

import argparse
import hashlib
import io
import json
import os
import pickle
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from pytorch_pretrained_vit import ViT
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "MKG-Y"

ARCHIVE = ROOT.parent / "dataset_audit/MMRNS/MKG-W-Y-raw_data/MKG-Y_img.zip"

ENT_FILE = DATA / "ent_id"
OUTPUT = DATA / "img_vec.pickle"
CHECKPOINT = DATA / "img_features_checkpoint.npz"
MANIFEST = DATA / "img_features_manifest.json"

N_ENTITIES = 15000
DIM = 1000
SEED = 2022

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

transform = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


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


def index_images(archive):
    image_map = defaultdict(list)

    for name in archive.namelist():
        if name.endswith("/"):
            continue

        parts = name.split("/", 2)

        if len(parts) != 3 or parts[0] != "DY_open":
            continue

        if Path(name).suffix.lower() not in EXTENSIONS:
            continue

        entity = parts[1]
        image_map[entity].append(name)

    for entity in image_map:
        image_map[entity].sort(key=lambda path: (Path(path).name, path))

    return image_map


def load_image(archive, paths):
    for path in paths:
        try:
            with archive.open(path) as stream:
                content = stream.read()

            with Image.open(io.BytesIO(content)) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                img.load()

                if min(img.size) < 32:
                    continue

                return transform(img), path

        except (OSError, ValueError, RuntimeError, EOFError):
            continue

    return None, ""


def save_checkpoint(features, completed, sources, fingerprint):
    tmp = CHECKPOINT.with_name(CHECKPOINT.name + ".tmp")

    with tmp.open("wb") as f:
        np.savez(
            f,
            features=features,
            completed=completed,
            sources=np.asarray(sources, dtype=str),
            fingerprint=np.array(fingerprint),
        )

    os.replace(tmp, CHECKPOINT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=300)
    args = parser.parse_args()

    if args.batch_size < 1 or args.save_every < 1:
        parser.error("batch-size and save-every must be positive")

    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)

    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    entries = load_entities()

    # Prevent resuming with a different entity mapping or archive.
    fingerprint = hashlib.sha256(
        ENT_FILE.read_bytes()
        + str(ARCHIVE.stat().st_size).encode()
        + str(ARCHIVE.stat().st_mtime_ns).encode()
    ).hexdigest()

    rng = np.random.default_rng(SEED)

    features = rng.standard_normal((N_ENTITIES, DIM)).astype(np.float32)

    completed = np.zeros(N_ENTITIES, dtype=bool)
    sources = [""] * N_ENTITIES

    if CHECKPOINT.exists():
        print("Resuming checkpoint...", flush=True)

        with np.load(CHECKPOINT, allow_pickle=False) as state:
            if str(state["fingerprint"].item()) != fingerprint:
                raise RuntimeError("Checkpoint input fingerprint mismatch")

            features = state["features"].copy()
            completed = state["completed"].copy()
            sources = state["sources"].tolist()

        assert features.shape == (N_ENTITIES, DIM)
        assert completed.shape == (N_ENTITIES,)
        assert len(sources) == N_ENTITIES

    print("Loading pretrained ViT...", flush=True)

    device = torch.device("cuda")
    model = ViT("B_16_imagenet1k", pretrained=True)
    model = model.to(device).eval()

    pending_tensors = []
    pending_ids = []

    newly_encoded = 0
    since_save = 0

    def flush_batch():
        nonlocal newly_encoded

        if not pending_tensors:
            return

        batch = torch.stack(pending_tensors).to(device)

        with torch.inference_mode():
            vectors = model(batch).cpu().numpy().astype(np.float32)

        assert vectors.shape == (len(pending_ids), DIM)

        for idx, vector in zip(pending_ids, vectors):
            features[idx] = vector
            completed[idx] = True

        newly_encoded += len(pending_ids)

        pending_tensors.clear()
        pending_ids.clear()

    with zipfile.ZipFile(ARCHIVE) as archive:
        print("Indexing MKG-Y images...", flush=True)
        image_map = index_images(archive)

        available = sum(entity in image_map for entity, _ in entries)

        print("Entities:", len(entries))
        print("Raw image folders:", len(image_map))
        print("Entities with candidate images:", available)
        print("Entities without candidate images:", N_ENTITIES - available)

        for entity, idx in tqdm(entries, desc="MKG-Y ViT"):
            if completed[idx]:
                continue

            tensor, path = load_image(
                archive,
                image_map.get(entity, []),
            )

            if tensor is None:
                sources[idx] = ""
                completed[idx] = True
            else:
                pending_tensors.append(tensor)
                pending_ids.append(idx)
                sources[idx] = path

            if len(pending_tensors) >= args.batch_size:
                flush_batch()

            since_save += 1

            if since_save >= args.save_every:
                flush_batch()
                save_checkpoint(features, completed, sources, fingerprint)
                since_save = 0

        flush_batch()

    save_checkpoint(features, completed, sources, fingerprint)

    assert completed.all()
    assert features.shape == (N_ENTITIES, DIM)
    assert np.isfinite(features).all()

    real_count = sum(bool(path) for path in sources)
    missing_count = N_ENTITIES - real_count

    manifest = {
        "dataset": "MKG-Y",
        "entities": N_ENTITIES,
        "archive": str(ARCHIVE),
        "encoder": "pytorch_pretrained_vit/B_16_imagenet1k",
        "preprocessing": ("RGB, Resize(384,384), ToTensor, Normalize(0.5,0.5)"),
        "image_selection": (
            "first decodable image by sorted filename; minimum image side 32 pixels"
        ),
        "image_selection_provenance": (
            "WN18 reproduction baseline; original DSoM MKG-Y policy unverified"
        ),
        "seed": SEED,
        "feature_shape": [N_ENTITIES, DIM],
        "real_image_count": real_count,
        "missing_image_count": missing_count,
        "newly_encoded": newly_encoded,
        "input_fingerprint": fingerprint,
        "sources_by_internal_id": sources,
    }

    tmp_output = OUTPUT.with_name(OUTPUT.name + ".tmp")

    with tmp_output.open("wb") as f:
        pickle.dump(
            features,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(tmp_output, OUTPUT)

    with MANIFEST.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    CHECKPOINT.unlink(missing_ok=True)

    print("\n=== MKG-Y IMAGE FEATURES ===")
    print("Output:", OUTPUT)
    print("Shape:", features.shape)
    print("dtype:", features.dtype)
    print("Real images:", real_count)
    print("Missing/invalid images:", missing_count)
    print("Newly encoded:", newly_encoded)
    print("Manifest:", MANIFEST)
    print("MKG-Y IMAGE FEATURES: PASS")


if __name__ == "__main__":
    main()
