#!/usr/bin/env python3
"""Inspect MMKB raw-image URLs and downloaded image coverage, without network access."""

import argparse
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/MMKB-DB15K"
LINKS = ROOT.parent / "dataset_audit/MMRNS/MMKB-DB15K/DB15K_SameAsLink.txt"
DEFAULT_ARCHIVE = ROOT.parent / "mmkb/image-graph_urls.tar.gz"
DEFAULT_IMAGES = ROOT.parent / "mmkb/image-graph_images"
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def read_entities():
    entities = set()
    for line in (DATA / "ent_id").read_text(encoding="utf-8").splitlines():
        uri, idx = line.split("\t")
        entities.add(uri)
    if len(entities) != 12842:
        raise RuntimeError("Expected 12,842 structural entities")
    links = defaultdict(set)
    for line in LINKS.read_text(encoding="utf-8").splitlines():
        fb, relation, db, dot = line.split()
        if relation != "<SameAs>" or dot != ".":
            raise RuntimeError("Malformed SameAs row")
        uri = db[1:-1]
        if uri in entities:
            links["m." + fb.removeprefix("/m/")].add(uri)
    return entities, links


def inspect_archive(path, keys, full_scan, sample_lines):
    if not path.is_file():
        print("URL archive: NOT FOUND", path)
        return
    print("\n=== MMKB URL ARCHIVE ===")
    print("Archive:", path)
    matched_ids = set()
    parsed_lines = 0
    with tarfile.open(path, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.isfile()]
        print("Members:", len(members))
        print("First paths:", [m.name for m in members[:12]])
        targets = [m for m in members if m.name.rsplit("/", 1)[-1].startswith("URLS_")]
        print("URL list files:", [m.name for m in targets])
        for member in targets:
            file_obj = archive.extractfile(member)
            if file_obj is None:
                continue
            count = 0
            examples = []
            for data in file_obj:
                line = data.decode("utf-8", "replace").strip()
                if "\t" not in line:
                    continue
                url, value = line.rsplit("\t", 1)
                fb_id = value.split("/", 1)[0]
                if fb_id in keys:
                    matched_ids.add(fb_id)
                if len(examples) < 2:
                    examples.append((url[:90], value))
                count += 1
                if not full_scan and count >= sample_lines:
                    break
            parsed_lines += count
            print("List:", member.name, "scanned lines:", count, "examples:", examples)
    print("URLs sampled/scanned:", parsed_lines)
    print("DB15K Freebase IDs seen:", len(matched_ids))
    if not full_scan:
        print("Coverage from limited sample is NOT total URL coverage; use --full-scan")


def inspect_images(path, entities, links):
    print("\n=== DOWNLOADED RAW IMAGES ===")
    print("Raw images directory:", path)
    if not path.is_dir():
        print("NOT PRESENT. URL archive contains links, not downloaded JPGs")
        return
    matched_images = defaultdict(list)
    all_images = 0
    for file in path.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in EXTS:
            continue
        all_images += 1
        key = file.parent.name
        for uri in links.get(key, ()):
            matched_images[uri].append(file)
    print("All image files:", all_images)
    print("DB15K matched entities with raw-image candidates:", len(matched_images))
    print("DB15K without candidates:", len(entities) - len(matched_images))
    print(
        "Examples:",
        [str(paths[0]) for _, paths in list(sorted(matched_images.items()))[:5]],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url-archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--image-root", type=Path, default=DEFAULT_IMAGES)
    ap.add_argument("--full-scan", action="store_true", help="Scan all MMKB URL rows")
    ap.add_argument("--sample-lines", type=int, default=1000)
    ap.add_argument("--skip-archive", action="store_true")
    args = ap.parse_args()
    if args.sample_lines < 1:
        ap.error("--sample-lines must be >= 1")
    entities, links = read_entities()
    print("DB15K entities:", len(entities))
    print("Freebase IDs mapped to DB15K:", len(links))
    if not args.skip_archive:
        inspect_archive(args.url_archive, links, args.full_scan, args.sample_lines)
    inspect_images(args.image_root, entities, links)
    print("DB15K RAW IMAGE INVENTORY: COMPLETE")


if __name__ == "__main__":
    main()
