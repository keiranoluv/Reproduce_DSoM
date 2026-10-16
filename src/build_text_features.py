import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


DSOM_ROOT = Path("/workspace/workspace/DSoM")
KG_BERT_ROOT = Path("/workspace/workspace/kg-bert")

ENT_ID_FILE = DSOM_ROOT / "data/FB15K-237/ent_id"
TEXT_FILE = KG_BERT_ROOT / "data/FB15k-237/entity2textlong.txt"

# DSoM default filename
OUT_FILE = DSOM_ROOT / "data/FB15K-237/dscp_vec.pickle"

MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 32
MAX_LENGTH = 512
SEED = 2022


def load_ent_id(path):
    mid_to_idx = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            mid, idx = line.rstrip("\n").split("\t")
            mid_to_idx[mid] = int(idx)

    return mid_to_idx


def load_descriptions(path):
    mid_to_text = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 1)

            if len(parts) != 2:
                continue

            mid, text = parts

            # KG-BERT stores literal "\n" inside descriptions
            text = text.replace("\\n", " ").strip()

            if text:
                mid_to_text[mid] = text

    return mid_to_text


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)

    summed = (last_hidden_state * mask).sum(dim=1)
    count = mask.sum(dim=1).clamp(min=1e-9)

    return summed / count


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print("Loading DSoM ent_id...")
    mid_to_idx = load_ent_id(ENT_ID_FILE)

    print("Loading KG-BERT descriptions...")
    mid_to_text = load_descriptions(TEXT_FILE)

    n_entities = len(mid_to_idx)

    print("DSoM entities       :", n_entities)
    print("KG-BERT descriptions:", len(mid_to_text))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device              :", device)

    print("Loading", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        output_hidden_states=True,
    )
    model.to(device)
    model.eval()

    hidden_size = model.config.hidden_size
    print("Hidden size         :", hidden_size)

    assert hidden_size == 768

    # DSoM specifies np.random.normal for missing descriptions.
    # Initialize everything randomly, then overwrite rows that have text.
    features = np.random.normal(
        loc=0.0,
        scale=1.0,
        size=(n_entities, hidden_size),
    ).astype(np.float32)

    available = []
    missing = []

    # IMPORTANT: order according to DSoM ent_id
    for mid, idx in sorted(mid_to_idx.items(), key=lambda x: x[1]):
        if mid in mid_to_text:
            available.append((mid, idx, mid_to_text[mid]))
        else:
            missing.append((mid, idx))

    print("Matched descriptions:", len(available))
    print("Missing descriptions:", len(missing))

    with torch.no_grad():
        for start in tqdm(range(0, len(available), BATCH_SIZE)):
            batch = available[start:start + BATCH_SIZE]

            texts = [x[2] for x in batch]

            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            outputs = model(**inputs)

            pooled = mean_pool(
                outputs.hidden_states[-2],
                inputs["attention_mask"],
            )

            pooled = pooled.cpu().numpy().astype(np.float32)

            for (_, idx, _), vec in zip(batch, pooled):
                features[idx] = vec

    assert features.shape == (14541, 768)
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()

    with open(OUT_FILE, "wb") as f:
        pickle.dump(features, f)

    print()
    print("Saved:", OUT_FILE)
    print("Shape:", features.shape)
    print("dtype:", features.dtype)

    print()
    print("Missing entities using random normal:")
    for mid, idx in missing:
        print(idx, mid)


if __name__ == "__main__":
    main()
