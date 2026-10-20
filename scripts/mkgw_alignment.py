#!/usr/bin/env python3
"""Align MKG-W Wikidata structural entity IDs to DBpedia raw media keys."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/MKG-W"
ENT_FILE = DATA / "ent_id"
LINKS_FILE = ROOT / "src_data/MKG-W/ent_links"
DBPEDIA_MARKER = "/resource/"


def normalize_uri(text):
    return text.strip().strip("<>").strip()


def dbpedia_key(text):
    text = normalize_uri(text)
    if "dbpedia.org" not in text.lower() or DBPEDIA_MARKER not in text:
        return None
    # Do not discard slash-bearing names by taking only the final path component.
    return text.split(DBPEDIA_MARKER, 1)[1]


def load_alignment():
    entities = {}
    for line_no, line in enumerate(
        ENT_FILE.read_text(encoding="utf-8").splitlines(), 1
    ):
        name, idx = line.split("\t")
        if name in entities:
            raise ValueError(f"Duplicate ent_id row {line_no}: {name}")
        entities[name] = int(idx)
    if len(entities) != 15000 or sorted(entities.values()) != list(range(15000)):
        raise ValueError("MKG-W ent_id is not a contiguous 15000-row mapping")

    by_id = [None] * 15000
    for name, idx in entities.items():
        by_id[idx] = name

    keys = [None] * 15000
    unknown_lines = []
    collisions = []
    linked_count = 0
    for line_no, line in enumerate(
        LINKS_FILE.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            unknown_lines.append((line_no, line[:180]))
            continue
        a, b = (normalize_uri(p) for p in parts)
        # Determine link orientation by matching the structural vocabulary.
        if a in entities and dbpedia_key(b) is not None:
            wiki, raw_key = a, dbpedia_key(b)
        elif b in entities and dbpedia_key(a) is not None:
            wiki, raw_key = b, dbpedia_key(a)
        else:
            unknown_lines.append((line_no, line[:180]))
            continue
        idx = entities[wiki]
        if keys[idx] is not None and keys[idx] != raw_key:
            collisions.append((wiki, keys[idx], raw_key))
            continue
        if keys[idx] is None:
            linked_count += 1
        keys[idx] = raw_key

    print("=== MKG-W WIKIDATA TO DBPEDIA ALIGNMENT ===")
    print("Structural entities:", len(entities))
    print("Mapped entities:", linked_count)
    print("Missing links:", 15000 - linked_count)
    print("Unrecognized link rows:", len(unknown_lines))
    print("Conflicting mappings:", len(collisions))
    if unknown_lines:
        print("First unrecognized link rows:", unknown_lines[:3])
    if collisions:
        print("First conflicting mappings:", collisions[:3])
    print(
        "Mapped examples:",
        [(i, by_id[i], keys[i]) for i in range(15000) if keys[i] is not None][:3],
    )

    # Fail safely rather than generating vectors with incorrect entity alignment.
    if linked_count < 14500 or collisions:
        raise RuntimeError(
            "MKG-W ent_links alignment needs investigation; no features should be generated"
        )
    return by_id, keys


if __name__ == "__main__":
    load_alignment()
