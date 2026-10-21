#!/usr/bin/env python3
"""Encode downloaded MMKB Freebase-keyed raw images with ViT for DSoM.

Needs actual images in image-graph_images/m.xyz/provider_index.jpg.
This is NOT the MMRNS BEiT H5 and will never read H5 files.
"""

import argparse
import hashlib
import io
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/MMKB-DB15K"
ENT_FILE = DATA / "ent_id"
LINKS_FILE = ROOT.parent / "dataset_audit/MMRNS/MMKB-DB15K/DB15K_SameAsLink.txt"
DEFAULT_IMAGE_ROOT = ROOT.parent / "mmkb/image-graph_images"
OUTPUT = DATA / "img_vec.pickle"
CHECKPOINT = DATA / "img_features_checkpoint.npz"
MANIFEST = DATA / "img_features_manifest.json"
N_ENTITIES, DIM, SEED = 12842, 1000, 2022
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_alignment():
    entities = {}
    for line in ENT_FILE.read_text(encoding="utf-8").splitlines():
        uri, idx = line.split("\t")
        entities[uri] = int(idx)
    if len(entities) != N_ENTITIES or set(entities.values()) != set(range(N_ENTITIES)):
        raise RuntimeError("Invalid ent_id")
    fb_to_uri = {}
    for line in LINKS_FILE.read_text(encoding="utf-8").splitlines():
        fb, relation, db, dot = line.split()
        if relation != "<SameAs>" or dot != ".":
            raise ValueError("Malformed SameAs")
        uri = db[1:-1]
        key = "m." + fb.removeprefix("/m/")
        if key in fb_to_uri and fb_to_uri[key] != uri:
            raise ValueError(f"Conflicting SameAs for {key}")
        fb_to_uri[key] = uri
    return entities, fb_to_uri


def scan_images(root, entities, fb_to_uri):
    if not root.is_dir():
        raise FileNotFoundError(f"Downloaded raw-image directory not found: {root}")
    images = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in EXTS:
            continue
        uri = fb_to_uri.get(path.parent.name)
        idx = entities.get(uri)
        if idx is not None:
            images[idx].append(path)
    for paths in images.values():
        paths.sort(key=lambda p: (p.name, str(p)))
    return images


def digest_inputs(image_root, images):
    h = hashlib.sha256()
    h.update(ENT_FILE.read_bytes())
    h.update(LINKS_FILE.read_bytes())
    h.update(b"vit/B_16_imagenet1k|384x384|firstvalid|seed2022")
    for idx, paths in sorted(images.items()):
        for path in paths:
            stat = path.stat()
            h.update(
                f"{idx}|{path.relative_to(image_root)}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
            )
    return h.hexdigest()


def save_checkpoint(features, done, sources, fingerprint):
    temp = CHECKPOINT.with_name(CHECKPOINT.name + ".tmp")
    with temp.open("wb") as f:
        np.savez(
            f,
            features=features,
            done=done,
            sources=np.asarray(sources, dtype=str),
            fingerprint=np.array(fingerprint),
        )
    os.replace(temp, CHECKPOINT)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument(
        "--save-every", type=int, default=300, help="Checkpoint every N entities"
    )
    ap.add_argument(
        "--min-coverage",
        type=int,
        default=1000,
        help="Minimum entities with raw image candidates before extraction",
    )
    args = ap.parse_args()
    if args.batch_size < 1 or args.save_every < 1 or args.min_coverage < 1:
        ap.error("batch-size/save-every/min-coverage must be positive")

    entities, fb_to_uri = load_alignment()
    image_root = args.image_root.resolve()
    images = scan_images(image_root, entities, fb_to_uri)
    print("=== DB15K RAW IMAGE ALIGNMENT ===")
    print("Image directory:", image_root)
    print("KG entities:", N_ENTITIES)
    print("Raw-image candidate entities:", len(images))
    print("Missing raw-image candidates:", N_ENTITIES - len(images))
    print(
        "Example paths:",
        [str(paths[0]) for _, paths in list(sorted(images.items()))[:5]],
    )
    if args.audit_only:
        print("DB15K ViT IMAGE INPUT AUDIT: PASS")
        return
    if len(images) < args.min_coverage:
        raise RuntimeError(
            f"Only {len(images)} entities have raw images (< {args.min_coverage}); do not encode a nearly empty dataset"
        )
    if OUTPUT.exists() or MANIFEST.exists():
        raise FileExistsError(
            "Refusing to overwrite existing image feature output/manifest"
        )

    import torch
    from PIL import Image, ImageOps
    from pytorch_pretrained_vit import ViT
    from torchvision import transforms
    from tqdm import tqdm

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ViT extraction")

    transform = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    def open_first_valid(paths):
        for path in paths:
            try:
                with path.open("rb") as stream:
                    raw = stream.read()
                with Image.open(io.BytesIO(raw)) as obj:
                    obj = ImageOps.exif_transpose(obj).convert("RGB")
                    obj.load()
                    if min(obj.size) < 32:
                        continue
                    return transform(obj), str(path.relative_to(image_root))
            except (OSError, EOFError, RuntimeError, ValueError):
                continue
        return None, ""

    fingerprint = digest_inputs(image_root, images)
    rng = np.random.default_rng(SEED)
    features = rng.standard_normal((N_ENTITIES, DIM)).astype(np.float32)
    done = np.zeros(N_ENTITIES, dtype=bool)
    sources = [""] * N_ENTITIES
    if CHECKPOINT.exists():
        with np.load(CHECKPOINT, allow_pickle=False) as cp:
            if cp["fingerprint"].item() != fingerprint:
                raise RuntimeError(
                    "Raw images changed since checkpoint; refusing resume"
                )
            features, done, sources = (
                cp["features"].copy(),
                cp["done"].copy(),
                cp["sources"].tolist(),
            )
        if (
            features.shape != (N_ENTITIES, DIM)
            or done.shape != (N_ENTITIES,)
            or len(sources) != N_ENTITIES
        ):
            raise RuntimeError("Invalid ViT checkpoint")
        print("Resuming:", int(done.sum()), "entities completed")

    model = ViT("B_16_imagenet1k", pretrained=True).to("cuda").eval()
    pending, pending_ids = [], []

    def flush():
        if not pending:
            return
        with torch.inference_mode():
            vecs = (
                model(torch.stack(pending).to("cuda")).cpu().numpy().astype(np.float32)
            )
        if vecs.shape != (len(pending_ids), DIM):
            raise RuntimeError("Unexpected ViT vector dimensions")
        for idx, vec in zip(pending_ids, vecs):
            features[idx] = vec
            done[idx] = True
        pending.clear()
        pending_ids.clear()

    processed_since_save = 0
    for idx in tqdm(range(N_ENTITIES), desc="DB15K ViT"):
        if done[idx]:
            continue
        tensor, source = open_first_valid(images.get(idx, ()))
        if tensor is not None:
            pending.append(tensor)
            pending_ids.append(idx)
            sources[idx] = source
            if len(pending) >= args.batch_size:
                flush()
        else:
            done[idx] = True
        processed_since_save += 1
        if processed_since_save >= args.save_every:
            flush()
            save_checkpoint(features, done, sources, fingerprint)
            processed_since_save = 0
    flush()

    if not done.all() or not np.isfinite(features).all():
        raise RuntimeError("Incomplete/invalid ViT extraction")
    real_count = sum(bool(source) for source in sources)
    save_checkpoint(features, done, sources, fingerprint)
    temp = OUTPUT.with_name(OUTPUT.name + ".tmp")
    with temp.open("wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temp, OUTPUT)
    manifest = {
        "dataset": "MMKB-DB15K",
        "encoder": "pytorch_pretrained_vit/B_16_imagenet1k",
        "preprocessing": "RGB, Resize(384,384), ToTensor, Normalize(0.5,0.5)",
        "image_selection": "first decodable JPG/PNG etc by sorted filename; min side >=32 pixels",
        "selection_provenance": "Project MKG-Y/W reproduction protocol; exact author image-selection policy unverified",
        "image_root": str(image_root),
        "feature_shape": [N_ENTITIES, DIM],
        "seed": SEED,
        "candidate_entities": len(images),
        "real_image_entities": real_count,
        "missing_image_entities": N_ENTITIES - real_count,
        "input_fingerprint": fingerprint,
        "sources_by_internal_id": sources,
        "alignment": "Freebase folder m.xyz -> SameAsLink -> DBpedia URI -> KG ent_id",
        "missing_imputation": "numpy standard_normal seed 2022",
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    CHECKPOINT.unlink(missing_ok=True)
    print("DB15K ViT FEATURES: PASS", features.shape, "real images:", real_count)
    print("Output:", OUTPUT)


if __name__ == "__main__":
    main()
