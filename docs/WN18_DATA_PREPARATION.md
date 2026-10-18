# DSoM Reproduction Notes — WN18 Dataset Preparation

**Date:** 18/10/2026  
**Project:** Reproduce DSoM (ACM MM 2025)  
**Dataset:** WN18 (không phải WN18RR)  
**Status:** Dataset preparation completed — Structural, Text, Image PASS.

## 1. Environment & Paths

Repository:

```text
/workspace/workspace/DSoM
```

Python environment:

```text
/workspace/workspace/DSoM/dsom/bin/python
```

Dataset sources:

```text
/workspace/workspace/kg-bert/data/WN18/
/workspace/workspace/fb15k_wn18_images.zip
/workspace/workspace/dataset_audit/kbc/data/WN/
```

DSoM dataset directories:

```text
DSoM/
├── src_data/WN18/      # Raw structural triples
│   ├── train
│   ├── valid
│   └── test
├── data/WN18/          # Processed dataset + modality features
└── scripts/
    ├── prepare_wn18_raw.py
    ├── preprocess_wn18.py
    ├── build_wn18_text_features.py
    └── build_wn18_image_features.py
```

**Reproduction rule:** Các thao tác tạo hoặc biến đổi model inputs phải được lưu thành Python scripts có thể commit lên GitHub. Các thao tác audit hoặc kiểm tra kết quả có thể chạy trực tiếp trong terminal.

---

## 2. Dataset Provenance — Nguồn dữ liệu

### 2.1. Structural triples và textual descriptions

Nguồn: [KG-BERT](https://github.com/yao8839836/kg-bert).

Repository đã được clone trong quá trình chuẩn bị FB15K-237:

```bash
cd /workspace/workspace
git clone https://github.com/yao8839836/kg-bert.git
```

Do clone toàn bộ repository, dataset WN18 đã có sẵn tại:

```text
kg-bert/data/WN18/
```

Các file sử dụng:

| File | Công dụng |
|---|---|
| `train.tsv` | Training triples |
| `dev.tsv` | Validation triples |
| `test.tsv` | Test triples |
| `entities.txt` | Danh sách entity IDs |
| `entity2text.txt` | Entity descriptions |

KG-BERT dùng WordNet synset IDs dạng 8 chữ số:

```text
03964744    _hyponym    04371774
```

Image folders lại có tiền tố `n`:

```text
wn18-images/n03964744/
```

Do đó khi mapping ảnh, áp dụng:

```python
synset = "n" + entity
```

### 2.2. Images

Nguồn: [MKGformer](https://github.com/zjunlp/MKGformer).

File đã được tải từ quá trình chuẩn bị FB15K-237:

```text
/workspace/workspace/fb15k_wn18_images.zip
```

Trong ZIP có hai nguồn ảnh FB15K và WN18, trong đó WN18 nằm tại:

```text
wn18-images/nXXXXXXXX/
```

Kết quả kiểm tra:

```text
WN18 image files: 70385
Synsets with images: 7801
Archive integrity: PASS
```

Kiểm tra ZIP:

```bash
unzip -t /workspace/workspace/fb15k_wn18_images.zip \
  > /workspace/workspace/wn18_zip_integrity.log 2>&1

tail -5 /workspace/workspace/wn18_zip_integrity.log
```

Kết quả xác nhận:

```text
No errors detected in compressed data
```

**Note:** Không cần tải lại hoặc giải nén toàn bộ ZIP. Image feature extraction đọc trực tiếp từ archive.

---

## 3. Verify Structural Triples Against KBC

DSoM tham chiếu structural data từ KBC. Để bảo đảm KG-BERT WN18 thực sự tương đương KBC, chúng ta đã đối chiếu toàn bộ triples.

Nguồn KBC:

[facebookresearch/kbc](https://github.com/facebookresearch/kbc)

Download:

```bash
cd /workspace/workspace

mkdir -p dataset_audit/kbc
cd dataset_audit/kbc

wget -c \
  https://dl.fbaipublicfiles.com/kbc/data.tar.gz \
  -O data.tar.gz
```

KBC sử dụng tên dataset `WN` thay cho `WN18`.

Extract:

```bash
tar -xzf data.tar.gz \
  --wildcards '*/WN/train' '*/WN/valid' '*/WN/test'
```

Đối chiếu bằng script đã tạo:

```bash
cd /workspace/workspace/DSoM

./dsom/bin/python scripts/audit_wn18_id_mapping.py
```

**Lưu ý quan trọng:**

- KG-BERT sử dụng WordNet synset IDs và relation names.
- KBC sử dụng numeric entity/relation IDs.
- So sánh trực tiếp chuỗi triple sẽ cho `MATCH=False`, dù hai dataset có thể tương đương.
- Phải map KG-BERT entities và relations sang numeric IDs bằng thứ tự từ điển trước khi so sánh.

Mapping:

```python
entities = sorted(all_entities)
relations = sorted(all_relations)

ent2id = {entity: str(i) for i, entity in enumerate(entities)}
rel2id = {relation: str(i) for i, relation in enumerate(relations)}
```

**Verified results:**

| Split | KG-BERT | KBC | Difference |
|---|---:|---:|---:|
| Train | 141.442 | 141.442 | 0 |
| Valid | 5.000 | 5.000 | 0 |
| Test | 5.000 | 5.000 | 0 |

```text
TRAIN MATCH: True
VALID MATCH: True
TEST MATCH: True

ALL SPLITS MATCH: True
```

**Conclusion:** KG-BERT WN18 và KBC/WN chứa cùng các triples ở cả ba splits sau mapping ID. Có thể dùng KG-BERT làm structural source để giữ nguyên WordNet synset IDs, thuận tiện cho việc căn chỉnh text và image features.

---

## 4. Prepare Raw Structural Dataset

Script:

```text
scripts/prepare_wn18_raw.py
```

Mục đích:

Chuyển triples KG-BERT thành định dạng input của DSoM.

| Input | Output |
|---|---|
| `train.tsv` | `src_data/WN18/train` |
| `dev.tsv` | `src_data/WN18/valid` |
| `test.tsv` | `src_data/WN18/test` |

Script:

- Kiểm tra input files.
- Parse từng triple dạng `head<TAB>relation<TAB>tail`.
- Kiểm tra số lượng bản ghi.
- Giữ nguyên WordNet synset IDs.
- Từ chối ghi đè thư mục output đã tồn tại.

Run:

```bash
cd /workspace/workspace/DSoM

./dsom/bin/python scripts/prepare_wn18_raw.py
```

Verified output:

```text
train: 141442 triples
valid: 5000 triples
test: 5000 triples

WN18 raw preparation: PASS
```

Kiểm tra:

```bash
ls -lh src_data/WN18/
head -3 src_data/WN18/train
```

**Không sử dụng trực tiếp numeric triples từ KBC ở bước này**, vì cần bảo toàn synset IDs để align modalities.

---

## 5. Preprocess Structural Dataset

Script:

```text
scripts/preprocess_wn18.py
```

Script gọi trực tiếp hàm preprocessing của DSoM:

```python
from process_datasets import prepare_dataset

prepare_dataset(str(source), "WN18")
```

Input:

```text
src_data/WN18/{train,valid,test}
```

Output:

```text
data/WN18/
├── ent_id
├── rel_id
├── train.pickle
├── valid.pickle
├── test.pickle
├── to_skip.pickle
└── probas.pickle
```

Run:

```bash
cd /workspace/workspace/DSoM

./dsom/bin/python scripts/preprocess_wn18.py
```

Expected vocabulary:

```text
40943 entities and 18 relations
```

Các file được sinh ra:

- `ent_id`: ánh xạ WordNet entity ID sang internal entity index.
- `rel_id`: ánh xạ relation name sang internal relation index.
- `train/valid/test.pickle`: triples đã được numeric encoding.
- `to_skip.pickle`: filtered evaluation lists, gồm cả reciprocal relation indexing.
- `probas.pickle`: train entity frequency distributions.

DSoM sử dụng 18 original relations; với reciprocal relations, mô hình có thể sử dụng 36 relation IDs.

**Important:** `data/WN18/ent_id` là mapping chuẩn để sắp xếp các hàng trong cả text và image feature matrices.

---

## 6. Structural Data Audit

Đã chạy:

```bash
./dsom/bin/python scripts/audit_wn18_dataset.py
```

Các kiểm tra:

1. Entity vocabulary và relation vocabulary.
2. Contiguous numeric IDs.
3. Train/valid/test shapes.
4. Raw triples và processed triples khớp chính xác.
5. Entity và relation IDs nằm trong khoảng hợp lệ.
6. Filtered evaluation (`lhs`, `rhs`) đúng theo train + valid + test.
7. `probas.pickle` đúng phân phối tần suất từ train.

Verified:

```text
Entities : 40943
Relations: 18

train: 141442 | dtype=uint64 | PASS
valid:    5000 | dtype=uint64 | PASS
test:     5000 | dtype=uint64 | PASS

FILTERED EVALUATION: PASS
TRAIN PROBABILITIES: PASS

WN18 STRUCTURAL AUDIT: PASS
```

**Structural preprocessing completed.**

---

## 7. WN18 Text Modality

### 7.1. Description source

```text
/workspace/workspace/kg-bert/data/WN18/entity2text.txt
```

Ví dụ:

```text
14854262    stool, solid excretory product evacuated from the bowels
00590383    chieftainship, the position of chieftain
```

Khác FB15K-237 sử dụng `entity2textlong.txt`, WN18 sử dụng `entity2text.txt`.

### 7.2. Text coverage

Đã đối chiếu keys trong `entity2text.txt` với `data/WN18/ent_id`.

Verified:

```text
WN18 entities:        40943
Valid descriptions:   40943
Coverage:             100%
Missing descriptions: 0
```

Không cần random imputation cho missing descriptions ở WN18.

### 7.3. BERT feature extraction

Script:

```text
scripts/build_wn18_text_features.py
```

Giữ nguyên thiết lập BERT của FB15K-237 reproduction:

| Parameter | Value |
|---|---|
| Encoder | `bert-base-uncased` |
| Hidden layer | `hidden_states[-2]` |
| Pooling | Attention-mask mean pooling |
| Max sequence length | 512 |
| Batch size | 32 |
| Seed | 2022 |
| Output dimension | 768 |
| Output dtype | float32 |

Mean pooling:

```python
hidden = outputs.hidden_states[-2]
mask = attention_mask.unsqueeze(-1).type_as(hidden)

pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
```

Output rows are aligned to `data/WN18/ent_id`.

Run:

```bash
cd /workspace/workspace/DSoM

./dsom/bin/python -u scripts/build_wn18_text_features.py \
  --device cuda \
  --batch-size 32 \
  --save-every 100 \
  2>&1 | tee data/WN18/build_text_features.log
```

Checkpoint:

```text
data/WN18/dscp_features_checkpoint.npz
```

Final output:

```text
data/WN18/dscp_vec.pickle
```

### 7.4. BERT verification

```bash
./dsom/bin/python - <<'PY'
import pickle
import numpy as np

with open("data/WN18/dscp_vec.pickle", "rb") as f:
    x = pickle.load(f)

print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("All finite:", np.isfinite(x).all())

assert x.shape == (40943, 768)
assert x.dtype == np.float32
assert np.isfinite(x).all()

print("TEXT FEATURES AUDIT: PASS")
PY
```

**Actual status: PASS.**

---

## 8. WN18 Image Modality

### 8.1. Image source and alignment

Source:

```text
/workspace/workspace/fb15k_wn18_images.zip
```

WN18 image directory format:

```text
wn18-images/nXXXXXXXX/image.JPEG
```

Image-to-entity alignment:

```python
synset = "n" + entity
```

Entity index lấy từ:

```text
data/WN18/ent_id
```

Verified image coverage:

| Metric | Result |
|---|---:|
| Total entities | 40.943 |
| Entities with images | 7.801 |
| Missing images | 33.142 |
| Image coverage | 19,05% |
| Missing image rate | 80,95% |
| Image files in archive | 70.385 |

### 8.2. Image feature extraction

Script:

```text
scripts/build_wn18_image_features.py
```

Cấu hình nhất quán với FB15K-237 reproduction:

| Parameter | Value |
|---|---|
| Encoder | `ViT("B_16_imagenet1k", pretrained=True)` |
| Library | `pytorch_pretrained_vit` |
| Input | RGB |
| Resize | 384 × 384 |
| Normalization | Mean = 0.5, Std = 0.5 |
| Image selection | First valid image by sorted filename |
| Minimum image size | 32 pixels |
| Missing images | Standard normal random vectors |
| Seed | 2022 |
| Feature dimension | 1000 |
| Dtype | float32 |

Image transform:

```python
transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
```

Random imputation:

```python
rng = np.random.default_rng(2022)
features = rng.standard_normal((40943, 1000)).astype(np.float32)
```

Với entities có ảnh hợp lệ, random row được thay bằng ViT feature.

**Lưu ý:** Đây là classification logits 1000 chiều từ implementation ViT đã chọn. Chưa xác minh feature layer và pretrained checkpoint này giống hệt thiết lập paper DSoM.

### 8.3. Smoke test

```bash
cd /workspace/workspace/DSoM

./dsom/bin/python -u scripts/build_wn18_image_features.py \
  --limit 1000 \
  --batch-size 8 \
  --save-every 100 \
  2>&1 | tee data/WN18/image_smoke.log
```

Smoke test xử lý tối đa 1.000 entities theo internal ID order, tạo checkpoint nhưng không xuất final pickle khi dataset chưa hoàn chỉnh.

### 8.4. Full feature extraction

```bash
./dsom/bin/python -u scripts/build_wn18_image_features.py \
  --batch-size 8 \
  --save-every 200 \
  2>&1 | tee data/WN18/image_full.log
```

Script hỗ trợ resume từ checkpoint:

```text
data/WN18/img_features_checkpoint.npz
```

Final outputs:

```text
data/WN18/img_vec.pickle
data/WN18/img_features_manifest.json
```

Image features được đọc trực tiếp từ ZIP, không cần tạo `data/WN18/raw_images/`.

### 8.5. Image verification

```bash
./dsom/bin/python - <<'PY'
import pickle
import json
import numpy as np

with open("data/WN18/img_vec.pickle", "rb") as f:
    x = pickle.load(f)

with open("data/WN18/img_features_manifest.json") as f:
    metadata = json.load(f)

print("Shape:", x.shape)
print("Dtype:", x.dtype)
print("Finite:", np.isfinite(x).all())
print("Real images:", metadata["real_image_count"])
print("Missing:", metadata["missing_image_count"])

assert x.shape == (40943, 1000)
assert x.dtype == np.float32
assert np.isfinite(x).all()
assert metadata["real_image_count"] == 7801
assert metadata["missing_image_count"] == 33142

print("WN18 IMAGE AUDIT: PASS")
PY
```

**Actual verified output:**

```text
Shape: (40943, 1000)
Dtype: float32
Finite: True
Real images: 7801
Missing: 33142

WN18 IMAGE AUDIT: PASS
```

---

## 9. Final Dataset Files

Sau khi chuẩn bị, các model inputs được lưu tại:

```text
data/WN18/
├── ent_id
├── rel_id
├── train.pickle
├── valid.pickle
├── test.pickle
├── to_skip.pickle
├── probas.pickle
├── dscp_vec.pickle
├── img_vec.pickle
├── img_features_manifest.json
└── *.log
```

Expected and verified principal shapes:

| Artifact | Shape |
|---|---|
| `train.pickle` | (141442, 3) |
| `valid.pickle` | (5000, 3) |
| `test.pickle` | (5000, 3) |
| `dscp_vec.pickle` | (40943, 768) |
| `img_vec.pickle` | (40943, 1000) |

All model feature arrays have finite values.

---

## 10. Reproduction Caveats

### Structural dataset

**Verified exactly equivalent to KBC/WN after entity/relation ID mapping.**

This verification compares all triples across train, valid, and test, not merely dataset sizes.

### Text embeddings

100% description coverage is verified.

However, using `bert-base-uncased` and mean pooling of the penultimate hidden layer does not prove the embeddings match the original authors' pretrained weights, pooling choices, or preprocessing exactly.

### Image embeddings

Image source, coverage, integrity, selection rule and output matrix shape are verified for this reproduction.

The original paper's exact image selection, image encoder checkpoint, preprocessing settings and feature extraction layer are not fully verified.

### Missing images

80.95% of WN18 entities have no image in this archive.

Random feature imputation is part of this reproduction setup; the high missing-image ratio should be documented when analyzing Teacher/Student performance.

### Reproducibility

For complete reruns, record:

- Git commit SHA.
- Library/model versions.
- Random seed.
- Dataset sources and archive checksums.
- BERT/ViT encoder configuration.
- Feature extraction scripts.
- WN18 modality coverage statistics.

---

## 11. Current Status and Next Step

**Completed:**

- [x] Identify WN18 dataset sources.
- [x] Verify KG-BERT and KBC triples equivalence.
- [x] Verify MKGformer ZIP integrity.
- [x] Prepare raw structural triples.
- [x] Preprocess structural dataset.
- [x] Audit structural mapping, filtered evaluation and probabilities.
- [x] Confirm 100% text coverage.
- [x] Extract and audit BERT text embeddings.
- [x] Confirm image coverage at 19.05%.
- [x] Extract and audit ViT image embeddings.

**Remaining:**

- [ ] Audit WN18 Teacher configuration and paper hyperparameters.
- [ ] Run WN18 Teacher smoke test.
- [ ] Train WN18 Teacher.
- [ ] Run WN18 Student distillation.
- [ ] Compare Student against DSoM Table 3.
- [ ] Document experimental deviations and ablations.

**Final conclusion:** WN18 model inputs are prepared and structurally audited. Dataset preparation is complete for the selected BERT/ViT reproduction pipeline. Training and experimental reproduction of Table 3 have not yet been performed.
