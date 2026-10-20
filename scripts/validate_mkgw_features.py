#!/usr/bin/env python3
"""Read-only validation of MKG-W BERT and ViT matrices and provenance."""

import json
import pickle
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data/MKG-W"
FILES = {
    "text": ("dscp_vec.pickle", "dscp_features_manifest.json", (15000, 768)),
    "image": ("img_vec.pickle", "img_features_manifest.json", (15000, 1000)),
}
for name, (matrix_filename, manifest_filename, expected) in FILES.items():
    with (DATA / matrix_filename).open("rb") as f:
        vec = pickle.load(f)
    if not isinstance(vec, np.ndarray) or vec.shape != expected:
        raise RuntimeError(
            f"Invalid {name} matrix: {type(vec)} {getattr(vec, 'shape', None)}"
        )
    if vec.dtype != np.float32 or not np.isfinite(vec).all():
        raise RuntimeError(f"Invalid {name} dtype/nonfinite values")
    manifest = json.loads((DATA / manifest_filename).read_text(encoding="utf-8"))
    if manifest.get("dataset") != "MKG-W" or manifest.get("feature_shape") != list(
        expected
    ):
        raise RuntimeError(f"Invalid {name} manifest")
    print(
        f"{name.upper()}: PASS; shape={vec.shape}, dtype={vec.dtype}, mean={vec.mean():.6f}, std={vec.std():.6f}"
    )
    if name == "text":
        print("  Matched raw descriptions:", manifest["available_texts"])
    else:
        print("  Encoded raw images:", manifest["real_image_count"])
        print("  Missing or invalid:", manifest["missing_image_count"])
print("MKG-W MULTIMODAL FEATURES: PASS")
