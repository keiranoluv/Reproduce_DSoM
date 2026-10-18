#!/usr/bin/env python3
"""Build WN18 ViT features directly from the MKGformer ZIP archive."""

import argparse
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
DATA = ROOT / "data/WN18"
ARCHIVE = ROOT.parent / "fb15k_wn18_images.zip"

ENT_FILE = DATA / "ent_id"
OUTPUT = DATA / "img_vec.pickle"
CHECKPOINT = DATA / "img_features_checkpoint.npz"
MANIFEST = DATA / "img_features_manifest.json"

N_ENTITIES = 40943
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
    for line in ENT_FILE.read_text(encoding="utf-8").splitlines():
        entity, idx = line.split("\t")
        entries.append((entity, int(idx)))

    entries.sort(key=lambda item: item[1])

    assert len(entries) == N_ENTITIES
    assert [idx for _, idx in entries] == list(range(N_ENTITIES))
    return entries


def index_images(archive):
    image_map = defaultdict(list)

    for name in archive.namelist():
        if not name.startswith("wn18-images/"):
            continue

        parts = name.split("/")
        if len(parts) != 3:
            continue

        if Path(name).suffix.lower() not in EXTENSIONS:
            continue

        image_map[parts[1]].append(name)

    for synset in image_map:
        image_map[synset].sort(key=lambda x: Path(x).name)

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


def save_checkpoint(features, completed, sources):
    tmp = CHECKPOINT.with_name(CHECKPOINT.name + ".tmp")

    with open(tmp, "wb") as f:
        np.savez(
            f,
            features=features,
            completed=completed,
            sources=np.asarray(sources, dtype=str),
        )

    os.replace(tmp, CHECKPOINT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1 or args.save_every < 1 or args.limit < 0:
        parser.error("Invalid batch-size, save-every or limit")

    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for ViT extraction")

    if OUTPUT.exists():
        raise FileExistsError(f"Output exists: {OUTPUT}")

    if args.reset and CHECKPOINT.exists():
        CHECKPOINT.unlink()

    entries = load_entities()
    selected = entries[: args.limit] if args.limit else entries

    rng = np.random.default_rng(SEED)
    features = rng.standard_normal((N_ENTITIES, DIM)).astype(np.float32)

    completed = np.zeros(N_ENTITIES, dtype=bool)
    sources = [""] * N_ENTITIES

    if CHECKPOINT.exists():
        print("Resuming from checkpoint", flush=True)
        with np.load(CHECKPOINT) as cp:
            features = cp["features"].copy()
            completed = cp["completed"].copy()
            sources = cp["sources"].tolist()

        assert features.shape == (N_ENTITIES, DIM)
        assert completed.shape == (N_ENTITIES,)
        assert len(sources) == N_ENTITIES

    print("Loading pretrained ViT...", flush=True)
    device = torch.device("cuda")
    model = ViT("B_16_imagenet1k", pretrained=True)
    model = model.to(device).eval()

    pending_tensors = []
    pending_ids = []
    processed = 0
    since_save = 0

    def flush_batch():
        nonlocal processed

        if not pending_tensors:
            return

        batch = torch.stack(pending_tensors).to(device)

        with torch.inference_mode():
            vectors = model(batch).cpu().numpy().astype(np.float32)

        assert vectors.shape == (len(pending_ids), DIM)

        for idx, vec in zip(pending_ids, vectors):
            features[idx] = vec
            completed[idx] = True

        processed += len(pending_ids)
        pending_tensors.clear()
        pending_ids.clear()

    with zipfile.ZipFile(ARCHIVE) as archive:
        print("Indexing WN18 images...", flush=True)
        image_map = index_images(archive)

        print("Entities:", len(entries))
        print("Image synsets:", len(image_map))

        for entity, idx in tqdm(selected, desc="WN18 ViT"):
            if completed[idx]:
                continue

            synset = "n" + entity
            tensor, path = load_image(
                archive,
                image_map.get(synset, []),
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
                save_checkpoint(features, completed, sources)
                since_save = 0

    flush_batch()
    save_checkpoint(features, completed, sources)

    done = sum(bool(completed[idx]) for _, idx in selected)
    real = sum(bool(sources[idx]) for _, idx in selected)

    print("Selected:", len(selected))
    print("Completed:", done)
    print("Real images:", real)
    print("Missing/invalid images:", done - real)
    print("Newly encoded:", processed)

    if not completed.all():
        print("Partial run: checkpoint saved, no final pickle")
        return

    assert features.shape == (N_ENTITIES, DIM)
    assert np.isfinite(features).all()

    tmp = OUTPUT.with_name(OUTPUT.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    os.replace(tmp, OUTPUT)

    manifest = {
        "dataset": "WN18",
        "archive": str(ARCHIVE),
        "encoder": "pytorch_pretrained_vit/B_16_imagenet1k",
        "preprocessing": ("RGB, resize 384x384, ToTensor, Normalize(0.5,0.5)"),
        "image_selection": "first valid image by sorted filename",
        "seed": SEED,
        "real_image_count": sum(bool(x) for x in sources),
        "missing_image_count": sum(not bool(x) for x in sources),
        "sources_by_internal_id": sources,
    }

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("SAVED:", OUTPUT)
    print("SHAPE:", features.shape)
    print("DTYPE:", features.dtype)
    print("WN18 IMAGE FEATURES: PASS")

    CHECKPOINT.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
