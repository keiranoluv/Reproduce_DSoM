#!/usr/bin/env python3
"""Fetch public MMRNS DB15K triples and record hashes for the source audit."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parents[1]
FILE_IDS = {
    "train": "1XitpzW1UKeHYBEbZ7K22DIbx-TMcnknV",
    "valid": "17rUiNPHa-qnBQq7BzrySSfdH80Otr5Dk",
    "test": "1-sP4kIl3l7BHfvFyu7OKnIrCqEs236MG",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT.parent / "dataset_audit/MMRNS/MMKB-DB15K",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = {}
    for split, file_id in FILE_IDS.items():
        path = args.output / f"{split}.txt"
        gdown.download(id=file_id, output=str(path), quiet=False)
        content = path.read_bytes()
        # Reject error pages or an unexpected document format before recording it.
        rows = content.decode("utf-8").splitlines()
        if not rows or any(len(row.split("\t")) != 3 for row in rows):
            raise ValueError(f"Invalid triples in {path}")
        matches = content == (args.source / f"{split}.txt").read_bytes()
        files[split] = {
            "file_id": file_id,
            "url": f"https://drive.google.com/file/d/{file_id}/view",
            "sha256": hashlib.sha256(content).hexdigest(),
            "matches_downloaded_source": matches,
            "bytes": len(content),
        }
    record = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_repository": "https://github.com/quqxui/MMRNS",
        "source_folder": "https://drive.google.com/drive/folders/1eWW4QquPkZU2hEVONkJfLGpsBY_IpwR7",
        "files": files,
    }
    output = args.output / "upstream_downloads.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Retrieval record: {output}")
    if not all(item["matches_downloaded_source"] for item in files.values()):
        raise ValueError(
            "Public source differs from local raw files; inspect the saved copies"
        )
    print("PASS: all three public files match the local downloaded source")


if __name__ == "__main__":
    main()
