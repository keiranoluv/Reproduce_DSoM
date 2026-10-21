#!/usr/bin/env python3
"""Build deterministic, entity-aligned DB15K H5 pilot features."""

import json
import os
import pickle
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "dataset_audit/MMRNS/MMKB-DB15K"
EMB = ROOT.parent / "dataset_audit/MMRNS/emb_data"
DATA = ROOT / "data/MMKB-DB15K"
OUT = DATA / "h5_pilot"

SEED = 2022
N_ENTITIES = 12842


def save_pickle(path, array):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(array, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def main():
    output_text = OUT / "dscp_vec.pickle"
    output_image = OUT / "img_vec.pickle"
    manifest_path = OUT / "feature_manifest.json"

    for path in (output_text, output_image, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    entities = {}
    with (DATA / "ent_id").open(encoding="utf-8") as f:
        for line in f:
            uri, idx = line.rstrip("\n").split("\t")
            entities[uri] = int(idx)

    assert len(entities) == N_ENTITIES
    assert set(entities.values()) == set(range(N_ENTITIES))

    links = {}
    with (SOURCE / "DB15K_SameAsLink.txt").open(encoding="utf-8") as f:
        for line in f:
            fb, rel, db, dot = line.strip().split()
            assert rel == "<SameAs>" and dot == "."
            db = db.removeprefix("<").removesuffix(">")

            if fb in links and links[fb] != db:
                raise ValueError(f"Conflicting mapping: {fb}")
            links[fb] = db

    rng = np.random.default_rng(SEED)

    text = rng.standard_normal((N_ENTITIES, 384)).astype(np.float32)

    image = rng.standard_normal((N_ENTITIES, 383)).astype(np.float32)

    text_seen = set()

    with h5py.File(EMB / "MMKB_description_sentences.h5", "r") as f:
        for key in sorted(f.keys()):
            uri = "http://dbpedia.org/resource/" + key
            idx = entities.get(uri)
            if idx is None:
                continue

            vectors = np.asarray(f[key], dtype=np.float32)
            assert vectors.ndim == 2
            assert vectors.shape[1] == 384
            assert vectors.shape[0] > 0
            assert np.isfinite(vectors).all()

            # MMRNS keeps up to four sentence embeddings.
            # Pilot: collapse them to one entity vector.
            text[idx] = vectors[:4].mean(axis=0)
            text_seen.add(idx)

    image_sums = np.zeros((N_ENTITIES, 383), dtype=np.float64)
    image_counts = np.zeros(N_ENTITIES, dtype=np.int32)

    with h5py.File(EMB / "MMKB_img_BEIT_16-224.h5", "r") as f:
        for key in sorted(f.keys()):
            if not key.startswith("m."):
                continue

            freebase = "/m/" + key[2:]
            uri = links.get(freebase)
            idx = entities.get(uri)

            if idx is None:
                continue

            vectors = np.asarray(f[key], dtype=np.float32)
            assert vectors.shape == (24, 383)
            assert np.isfinite(vectors).all()

            # Pilot: mean-pool the 24 released image vectors.
            image_sums[idx] += vectors.mean(axis=0)
            image_counts[idx] += 1

    image_seen = np.flatnonzero(image_counts > 0)

    image[image_seen] = (
        image_sums[image_seen] / image_counts[image_seen, None]
    ).astype(np.float32)

    assert text.shape == (N_ENTITIES, 384)
    assert image.shape == (N_ENTITIES, 383)
    assert np.isfinite(text).all()
    assert np.isfinite(image).all()

    manifest = {
        "dataset": "MMKB-DB15K",
        "experiment": "T1_DB15K_H5_Pilot",
        "seed": SEED,
        "source": "MMRNS released H5 embeddings",
        "text_encoder": "MMRNS released text H5",
        "image_encoder": "MMRNS released BEiT H5",
        "text_pooling": "mean of first up to 4 rows",
        "image_pooling": "mean of 24 rows per H5 key, then mean across keys per entity",
        "text_shape": list(text.shape),
        "image_shape": list(image.shape),
        "text_matched_entities": len(text_seen),
        "image_matched_entities": len(image_seen),
        "image_h5_keys_matched": int(image_counts.sum()),
        "image_multi_key_entities": int(np.sum(image_counts > 1)),
        "missing_text_entities": N_ENTITIES - len(text_seen),
        "missing_image_entities": N_ENTITIES - len(image_seen),
        "missing_imputation": "numpy standard_normal seed 2022",
        "note": "H5 pilot, not equivalent to DSoM BERT/ViT",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    save_pickle(output_text, text)
    save_pickle(output_image, image)

    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, manifest_path)

    print("\n=== DB15K H5 PILOT FEATURES ===")
    print(json.dumps(manifest, indent=2))
    print("DB15K H5 FEATURE PREPARATION: PASS")


if __name__ == "__main__":
    main()
