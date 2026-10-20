#!/usr/bin/env python3
"""ViT image feature extraction for MKG-W, aligned through MMRNS ent_links."""

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

from scripts.mkgw_alignment import DATA, ENT_FILE, LINKS_FILE, ROOT, load_alignment

ARCHIVE = ROOT.parent / "dataset_audit/MMRNS/MKG-W-Y-raw_data/MKG-W_img.zip"
OUTPUT = DATA / "img_vec.pickle"
CHECKPOINT = DATA / "img_features_checkpoint.npz"
MANIFEST = DATA / "img_features_manifest.json"
N, DIM, SEED = 15000, 1000, 2022
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TRANSFORM = None


def index_images(archive):
    out = defaultdict(list)
    samples = []
    for info in archive.infolist():
        name = info.filename
        if info.is_dir() or Path(name).suffix.lower() not in EXTENSIONS:
            continue
        parts = name.split("/")
        if len(parts) < 2:
            continue
        # MKG-W media is DBpedia-keyed; use the image's immediate parent folder.
        # The audit-only step checks that this actually matches entity link keys.
        key = parts[-2]
        if not key:
            continue
        out[key].append(name)
        if len(samples) < 5:
            samples.append(name)
    for paths in out.values():
        paths.sort(key=lambda p: (Path(p).name, p))
    return out, samples


def load_first_valid(archive, paths):
    from PIL import Image, ImageOps

    for path in paths:
        try:
            with archive.open(path) as source:
                content = source.read()
            with Image.open(io.BytesIO(content)) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                image.load()
                if min(image.size) < 32:
                    continue
                return TRANSFORM(image), path
        except (OSError, EOFError, ValueError, RuntimeError):
            continue
    return None, ""


def save_checkpoint(features, done, sources, fingerprint):
    temp = CHECKPOINT.with_name(CHECKPOINT.name + ".tmp")
    with temp.open("wb") as f:
        np.savez(
            f,
            features=features,
            completed=done,
            sources=np.asarray(sources, dtype=str),
            fingerprint=np.array(fingerprint),
        )
    os.replace(temp, CHECKPOINT)


def input_fingerprint():
    h = hashlib.sha256()
    for path in (ENT_FILE, LINKS_FILE):
        h.update(path.read_bytes())
    st = ARCHIVE.stat()
    h.update(f"{st.st_size}|{st.st_mtime_ns}|vit384|seed2022|firstvalid".encode())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=300)
    args = parser.parse_args()
    if args.batch_size < 1 or args.save_every < 1:
        parser.error("batch-size and save-every must be positive")
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    _, keys = load_alignment()
    with zipfile.ZipFile(ARCHIVE) as archive:
        image_map, sample_paths = index_images(archive)
    candidates = sum(key is not None and key in image_map for key in keys)
    print("=== MKG-W RAW IMAGE ALIGNMENT ===")
    print("ZIP sample paths:", sample_paths)
    print("Raw image folders:", len(image_map))
    print("Entities with candidate raw images:", candidates)
    print("Entities without raw images:", N - candidates)
    print("Important: raw image coverage may be lower than released MMRNS H5 coverage")
    if candidates < 7000:
        raise RuntimeError(
            "Image alignment unexpectedly low; inspect archive layout and ent_links"
        )
    if args.audit_only:
        print("MKG-W IMAGE ALIGNMENT: PASS")
        return

    import torch
    from PIL import Image, ImageOps
    from pytorch_pretrained_vit import ViT
    from torchvision import transforms
    from tqdm import tqdm

    global TRANSFORM
    TRANSFORM = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    fingerprint = input_fingerprint()
    rng = np.random.default_rng(SEED)
    features = rng.standard_normal((N, DIM)).astype(np.float32)
    done = np.zeros(N, dtype=bool)
    sources = [""] * N
    if CHECKPOINT.exists():
        with np.load(CHECKPOINT, allow_pickle=False) as cp:
            if str(cp["fingerprint"].item()) != fingerprint:
                raise RuntimeError("Checkpoint inputs changed; refusing unsafe resume")
            features = cp["features"].copy()
            done = cp["completed"].copy()
            sources = cp["sources"].tolist()
        if features.shape != (N, DIM) or done.shape != (N,) or len(sources) != N:
            raise RuntimeError("Invalid image checkpoint")
        print("Resuming with", int(done.sum()), "entities completed")

    model = ViT("B_16_imagenet1k", pretrained=True).to("cuda").eval()
    pending, pending_ids = [], []
    since_save = 0

    def flush():
        if not pending:
            return
        with torch.inference_mode():
            vectors = (
                model(torch.stack(pending).to("cuda")).cpu().numpy().astype(np.float32)
            )
        if vectors.shape != (len(pending_ids), DIM):
            raise RuntimeError("Unexpected ViT output shape")
        for idx, vector in zip(pending_ids, vectors):
            features[idx] = vector
            done[idx] = True
        pending.clear()
        pending_ids.clear()

    with zipfile.ZipFile(ARCHIVE) as archive:
        for idx, key in enumerate(tqdm(keys, desc="MKG-W ViT")):
            if done[idx]:
                continue
            tensor, path = (
                load_first_valid(archive, image_map.get(key, [])) if key else (None, "")
            )
            if tensor is None:
                done[idx] = True
                sources[idx] = ""
            else:
                pending.append(tensor)
                pending_ids.append(idx)
                sources[idx] = path
            if len(pending) >= args.batch_size:
                flush()
            since_save += 1
            if since_save >= args.save_every:
                flush()
                save_checkpoint(features, done, sources, fingerprint)
                since_save = 0
        flush()

    if not done.all() or not np.isfinite(features).all():
        raise RuntimeError("Image feature extraction incomplete/invalid")
    save_checkpoint(features, done, sources, fingerprint)
    real_count = sum(bool(p) for p in sources)
    tmp = OUTPUT.with_name(OUTPUT.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, OUTPUT)
    manifest = {
        "dataset": "MKG-W",
        "entities": N,
        "archive": str(ARCHIVE),
        "encoder": "pytorch_pretrained_vit/B_16_imagenet1k",
        "preprocessing": "RGB, Resize(384,384), ToTensor, Normalize(0.5,0.5)",
        "image_selection": "first decodable image by sorted filename, min side 32px",
        "selection_provenance": "MKG-Y/WN18 reproduction assumption, not verified author preprocessing",
        "feature_shape": [N, DIM],
        "seed": SEED,
        "candidate_raw_image_entities": candidates,
        "real_image_count": real_count,
        "missing_image_count": N - real_count,
        "input_fingerprint": fingerprint,
        "sources_by_internal_id": sources,
        "alignment": "MKG-W structural Wikidata URI -> ent_links DBpedia key -> raw ZIP parent folder",
    }
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    CHECKPOINT.unlink(missing_ok=True)
    print(
        "MKG-W IMAGE FEATURES: PASS",
        features.shape,
        "real_images=",
        real_count,
        "missing=",
        N - real_count,
    )


if __name__ == "__main__":
    main()
