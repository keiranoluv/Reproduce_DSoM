import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm
from torchvision import transforms
from pytorch_pretrained_vit import ViT


ROOT = Path("/workspace/workspace/DSoM")
DATA = ROOT / "data/FB15K-237"
IMG_ROOT = Path("/workspace/workspace/image_dataset/FB15k-images")

ENT_FILE = DATA / "ent_id"
OUTPUT = DATA / "img_vec.pickle"
CHECKPOINT = DATA / "img_features_checkpoint.npz"
MANIFEST = DATA / "img_features_manifest.json"

DIM = 1000
SEED = 2022

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_entities():
    entries = []
    with open(ENT_FILE, encoding="utf-8") as f:
        for line in f:
            mid, idx = line.split()
            entries.append((mid, int(idx)))
    return sorted(entries, key=lambda x: x[1])


def index_images():
    image_map = defaultdict(list)

    for path in IMG_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in EXTENSIONS:
            continue

        mid = path.parent.name
        if mid.startswith("m."):
            image_map[mid].append(path)

    for mid in image_map:
        image_map[mid].sort(key=lambda p: p.name)

    return image_map


def load_image(paths):
    for path in paths:
        try:
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                img.load()

                if min(img.size) < 32:
                    continue

                return transform(img), str(path)

        except (OSError, ValueError):
            continue

    return None, None


def save_checkpoint(features, completed, sources):
    import os

    tmp = CHECKPOINT.with_name(CHECKPOINT.name + ".tmp")

    with open(tmp, "wb") as f:
        np.savez(
            f,
            features=features,
            completed=completed,
            sources=np.asarray(sources, dtype=str)
        )

    os.replace(tmp, CHECKPOINT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1 or args.limit < 0:
        parser.error("batch-size must be positive; limit must be nonnegative")

    if not IMG_ROOT.exists():
        raise FileNotFoundError(IMG_ROOT)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this run")

    entries = load_entities()
    n = len(entries)

    assert n == 14541
    assert sorted(idx for _, idx in entries) == list(range(n))

    print("Indexing extracted images...", flush=True)
    image_map = index_images()

    print("Entities:", n)
    print("Image directories:", len(image_map))

    # Deterministic random vectors for missing images
    rng = np.random.default_rng(SEED)
    features = rng.standard_normal((n, DIM)).astype(np.float32)

    completed = np.zeros(n, dtype=bool)
    sources = [""] * n

    if CHECKPOINT.exists() and not args.reset:
        print("Loading checkpoint...")
        with np.load(CHECKPOINT) as cp:
            features = cp["features"].copy()
            completed = cp["completed"].copy()
            sources = cp["sources"].tolist()

        assert features.shape == (n, DIM)
        assert completed.shape == (n,)

    # Limit is based on entity-ID order, including missing-image entities.
    selected = entries[:args.limit] if args.limit else entries

    print("Loading pretrained ViT...", flush=True)
    device = torch.device("cuda")

    model = ViT("B_16_imagenet1k", pretrained=True)
    model = model.to(device).eval()

    pending_tensors = []
    pending_ids = []
    processed = 0

    def flush_batch():
        nonlocal processed

        if not pending_tensors:
            return

        x = torch.stack(pending_tensors).to(device)

        with torch.inference_mode():
            y = model(x).cpu().numpy().astype(np.float32)

        assert y.shape == (len(pending_ids), DIM)

        for idx, vec in zip(pending_ids, y):
            features[idx] = vec

        processed += len(pending_ids)
        pending_tensors.clear()
        pending_ids.clear()

    for mid, idx in tqdm(selected, desc="Encoding entities"):
        if completed[idx]:
            continue

        image_mid = mid.replace("/m/", "m.", 1)
        tensor, path = load_image(image_map.get(image_mid, []))

        if tensor is not None:
            pending_tensors.append(tensor)
            pending_ids.append(idx)
            sources[idx] = path
        else:
            sources[idx] = ""

        completed[idx] = True

        if len(pending_tensors) >= args.batch_size:
            flush_batch()

        if completed.sum() % args.save_every == 0:
            flush_batch()
            save_checkpoint(features, completed, sources)

    flush_batch()
    save_checkpoint(features, completed, sources)

    selected_ids = [idx for _, idx in selected]
    done = sum(bool(completed[i]) for i in selected_ids)
    with_image = sum(bool(sources[i]) for i in selected_ids)

    print("\nSelected entities:", len(selected))
    print("Completed:", done)
    print("Real image features:", with_image)
    print("Missing/invalid images:", done - with_image)
    print("Newly encoded:", processed)

    # Do not overwrite final output during a partial test.
    if not completed.all():
        print("Partial run complete. Final pickle not written.")
        return

    assert features.shape == (14541, 1000)
    assert np.isfinite(features).all()

    with open(OUTPUT, "wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "dataset": "FB15K-237",
        "encoder": "pytorch_pretrained_vit/B_16_imagenet1k",
        "preprocessing": "RGB, resize 384x384, ToTensor, Normalize(0.5,0.5)",
        "image_selection": "first valid image in sorted filename order",
        "seed": SEED,
        "real_image_count": sum(bool(p) for p in sources),
        "missing_image_count": sum(not bool(p) for p in sources),
        "sources_by_internal_id": sources,
    }

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nSAVED:", OUTPUT)
    print("SHAPE:", features.shape)
    print("DTYPE:", features.dtype)
    print("Manifest:", MANIFEST)


if __name__ == "__main__":
    main()
