#!/usr/bin/env python3
"""Portable input adapter around unchanged historical feature encoders.

Default audit checks paths/mapping without importing Torch or downloading models.
Extraction requires --mode extract; image extraction currently requires CUDA.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

from prepare_dataset import EXPECTED
from run_table2 import digest

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    ("FB15K-237", "text"): "src/build_text_features.py",
    ("FB15K-237", "image"): "src/build_image_features.py",
    ("WN18", "text"): "scripts/build_wn18_text_features.py",
    ("WN18", "image"): "scripts/build_wn18_image_features.py",
    ("MKG-Y", "text"): "scripts/build_mkgy_text_features.py",
    ("MKG-Y", "image"): "scripts/build_mkgy_image_features.py",
    ("MKG-W", "text"): "scripts/build_mkgw_text_features.py",
    ("MKG-W", "image"): "scripts/build_mkgw_image_features.py",
    ("MMKB-DB15K", "text"): "scripts/build_db15k_bert_features.py",
    ("MMKB-DB15K", "image"): "scripts/build_db15k_vit_features.py",
}


def module_for(dataset, modality):
    filename = ROOT / MODULES[dataset, modality]
    spec = importlib.util.spec_from_file_location("portable_feature_backend", filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=EXPECTED, required=True)
    p.add_argument("--modality", choices=("text", "image"), required=True)
    p.add_argument("--mode", choices=("audit", "extract", "verify"), default="audit")
    p.add_argument(
        "--input",
        type=Path,
        help="Description file, image ZIP, or image directory (FB/DB)",
    )
    p.add_argument("--links", type=Path, help="DB15K SameAs file or MKG-W ent_links")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--save-every", type=int, default=100)
    args = p.parse_args()
    data = ROOT / "data" / args.dataset
    output = data / ("dscp_vec.pickle" if args.modality == "text" else "img_vec.pickle")
    entries = [line.split("\t") for line in (data / "ent_id").read_text().splitlines()]
    if len(entries) != EXPECTED[args.dataset][0] or sorted(
        int(v) for _, v in entries
    ) != list(range(len(entries))):
        raise ValueError("Invalid prepared entity mapping")
    if args.mode == "verify":
        import numpy as np

        with output.open("rb") as stream:
            values = pickle.load(stream)
        dim = 768 if args.modality == "text" else 1000
        if (
            values.shape != (len(entries), dim)
            or values.dtype != np.float32
            or not np.isfinite(values).all()
        ):
            raise ValueError("Unexpected feature shape/dtype/values")
        print("PASS:", output, values.shape, digest(output))
        return
    if args.input is None or not args.input.exists():
        raise FileNotFoundError(
            "--input must name the downloaded source file/directory"
        )
    required_links = args.dataset == "MKG-W" or (
        args.dataset == "MMKB-DB15K" and args.modality == "image"
    )
    links = args.links or (
        ROOT / "src_data/MKG-W/ent_links" if args.dataset == "MKG-W" else None
    )
    if required_links and (links is None or not links.is_file()):
        raise FileNotFoundError("--links is required for this entity alignment")
    report = dict(
        dataset=args.dataset,
        modality=args.modality,
        input=str(args.input.resolve()),
        entities=len(entries),
        encoder_script=MODULES[args.dataset, args.modality],
        output=str(output),
        output_exists=output.exists(),
    )
    print(json.dumps(report, indent=2), flush=True)
    if args.mode == "audit":
        print(
            "Path/mapping preflight only; media coverage is checked by the extraction backend. No Torch import."
        )
        return
    if output.exists():
        raise FileExistsError(output)
    if args.batch_size < 1 or args.save_every < 1:
        raise ValueError("batch-size/save-every must be positive")
    if args.modality == "image" and args.device != "cuda":
        raise ValueError("Historical image backends require --device cuda")
    fingerprint = {
        "ent_id": digest(data / "ent_id"),
        "input_path": str(args.input.resolve()),
        "backend": digest(ROOT / MODULES[args.dataset, args.modality]),
    }
    fingerprint["packages"] = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "torchvision",
            "numpy",
            "transformers",
            "pytorch-pretrained-vit",
            "pillow",
        )
    }
    if args.input.is_file():
        fingerprint["input_sha256"] = digest(args.input)
    else:
        hasher = hashlib.sha256()
        for path in sorted(args.input.rglob("*")):
            if path.is_file():
                stat = path.stat()
                hasher.update(
                    f"{path.relative_to(args.input)}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
                )
        fingerprint["directory_metadata_sha256"] = hasher.hexdigest()
    if required_links:
        fingerprint["links_sha256"] = digest(links)
    pending = data / f"{args.modality}_portable_inputs.json"
    checkpoint = data / (
        "dscp_features_checkpoint.npz"
        if args.modality == "text"
        else "img_features_checkpoint.npz"
    )
    if pending.exists():
        if json.loads(pending.read_text()) != fingerprint:
            raise ValueError("Feature source/mapping changed since extraction started")
    elif checkpoint.exists():
        raise FileExistsError(
            "Legacy checkpoint has no portable input fingerprint; do not resume it with this adapter"
        )
    else:
        pending.write_text(json.dumps(fingerprint, indent=2) + "\n")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(ROOT))
    # Imports kept inside the explicit extract branch; original files are unchanged.
    if args.dataset == "MKG-W":
        from scripts import mkgw_alignment

        mkgw_alignment.LINKS_FILE = links.resolve()
    module = module_for(args.dataset, args.modality)
    module.TEXT_FILE = args.input.resolve()
    module.ARCHIVE = args.input.resolve()
    if args.dataset == "MMKB-DB15K" and args.modality == "image":
        module.LINKS_FILE = links.resolve()
    if args.dataset == "FB15K-237":
        module.ENT_ID_FILE = data / "ent_id"
        module.OUT_FILE = output
        module.ROOT = ROOT
        module.DATA = data
        module.ENT_FILE = data / "ent_id"
        module.OUTPUT = output
        module.CHECKPOINT = data / "img_features_checkpoint.npz"
        module.MANIFEST = data / "img_features_manifest.json"
        module.IMG_ROOT = args.input.resolve()
        module.BATCH_SIZE = args.batch_size
    delegated = ["feature-backend"]
    if (args.dataset, args.modality) != ("FB15K-237", "text"):
        delegated += [
            "--batch-size",
            str(args.batch_size),
            "--save-every",
            str(args.save_every),
        ]
    if args.modality == "text" and args.dataset != "FB15K-237":
        delegated += ["--device", args.device]
    if args.dataset == "MMKB-DB15K" and args.modality == "image":
        delegated += ["--image-root", str(args.input.resolve())]
    previous = sys.argv
    try:
        sys.argv = delegated
        module.main()
    finally:
        sys.argv = previous
    if not output.exists():
        raise RuntimeError(
            "Backend did not produce final features; inspect checkpoint/log"
        )
    provenance = dict(
        report,
        input_fingerprint=fingerprint,
        input_sha256=digest(args.input) if args.input.is_file() else None,
        encoder_script_sha256=digest(ROOT / MODULES[args.dataset, args.modality]),
        features_sha256=digest(output),
        device=args.device,
        note="Historical reconstructed encoder; exact author feature equivalence unverified",
    )
    (data / f"{args.modality}_portable_manifest.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
