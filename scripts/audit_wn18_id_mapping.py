from collections import Counter
from pathlib import Path

BASE = Path("/workspace/workspace")
KG = BASE / "kg-bert/data/WN18"
KBC_ROOT = BASE / "dataset_audit/kbc"

matches = list(KBC_ROOT.rglob("WN/train"))
assert len(matches) == 1, matches
KBC = matches[0].parent

splits = {
    "train": ("train.tsv", "train"),
    "valid": ("dev.tsv", "valid"),
    "test": ("test.tsv", "test"),
}


def load(path):
    triples = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            parts = line.split("\t")
            assert len(parts) == 3, (path, line)
            triples.append(tuple(parts))
    return triples


kg_data = {split: load(KG / kg_file) for split, (kg_file, _) in splits.items()}

kbc_data = {split: load(KBC / kbc_file) for split, (_, kbc_file) in splits.items()}

entities = sorted(
    {e for triples in kg_data.values() for h, r, t in triples for e in (h, t)}
)

relations = sorted({r for triples in kg_data.values() for h, r, t in triples})

ent2id = {e: str(i) for i, e in enumerate(entities)}
rel2id = {r: str(i) for i, r in enumerate(relations)}

print("Entities:", len(ent2id))
print("Relations:", len(rel2id))

all_match = True

for split in splits:
    mapped = Counter((ent2id[h], rel2id[r], ent2id[t]) for h, r, t in kg_data[split])
    kbc = Counter(kbc_data[split])

    only_kg = mapped - kbc
    only_kbc = kbc - mapped
    match = not only_kg and not only_kbc
    all_match &= match

    print(f"\n{split.upper()}")
    print("Mapped KG-BERT:", sum(mapped.values()))
    print("KBC:", sum(kbc.values()))
    print("Only mapped KG-BERT:", sum(only_kg.values()))
    print("Only KBC:", sum(only_kbc.values()))
    print("MATCH:", match)

print("\nALL SPLITS MATCH:", all_match)
