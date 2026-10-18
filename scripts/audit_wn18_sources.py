from collections import Counter
from pathlib import Path

BASE = Path("/workspace/workspace")

KG = BASE / "kg-bert/data/WN18"
KBC_ROOT = BASE / "dataset_audit/kbc"

candidates = list(KBC_ROOT.rglob("WN/train"))
if len(candidates) != 1:
    raise RuntimeError(f"Expected one KBC WN/train, found {len(candidates)}")

KBC = candidates[0].parent

splits = {
    "train": ("train.tsv", "train"),
    "valid": ("dev.tsv", "valid"),
    "test": ("test.tsv", "test"),
}


def normalize_entity(x):
    x = x.strip()
    if len(x) == 9 and x[0] == "n" and x[1:].isdigit():
        return x[1:]
    return x


def normalize_relation(x):
    return x.strip().lstrip("_")


def load(path):
    triples = []

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue

        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"{path}:{number}: {line!r}")

        h, r, t = parts

        triples.append(
            (
                normalize_entity(h),
                normalize_relation(r),
                normalize_entity(t),
            )
        )

    return Counter(triples)


all_match = True

for split, (kg_name, kbc_name) in splits.items():
    a = load(KG / kg_name)
    b = load(KBC / kbc_name)

    only_kg = a - b
    only_kbc = b - a

    count_a = sum(a.values())
    count_b = sum(b.values())

    match = not only_kg and not only_kbc
    all_match &= match

    print(f"\n===== {split.upper()} =====")
    print("KG-BERT triples:", count_a)
    print("KBC triples    :", count_b)
    print("KG-BERT unique :", len(a))
    print("KBC unique     :", len(b))
    print("Only KG-BERT   :", sum(only_kg.values()))
    print("Only KBC       :", sum(only_kbc.values()))
    print("MATCH          :", match)

    if not match:
        print("Sample only KG-BERT:", list(only_kg.items())[:5])
        print("Sample only KBC    :", list(only_kbc.items())[:5])

print("\n================================")
print("ALL SPLITS EXACT MATCH:", all_match)
