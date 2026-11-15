# Ghi chép reproduction DSoM — Chuẩn bị WN18

**Ngày:** 18/10/2026
**Dự án:** Reproduce DSoM (ACM MM 2025)
**Dataset:** WN18, không WN18RR
**Trạng thái:** Đã chuẩn bị dataset, Structural/Text/Image PASS.

## 1. Môi trường và đường dẫn

Repository:

```text
/workspace/workspace/DSoM
```

Môi trường Python:

```text
/workspace/workspace/DSoM/dsom/bin/python
```

Nguồn datasets:

```text
/workspace/workspace/kg-bert/data/WN18/
/workspace/workspace/fb15k_wn18_images.zip
/workspace/workspace/dataset_audit/kbc/data/WN/
```

Thư mục datasets DSoM:

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

## 2. Nguồn dữ liệu

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

**Ghi chú:** Không cần tải lại/giải nén toàn ZIP. Image extraction đọc trực tiếp archive.

---

## 3. Đối chiếu structural triples với KBC

DSoM tham chiếu structural data từ KBC. Để bảo đảm KG-BERT WN18 thực sự tương đương KBC, chúng ta đã đối chiếu toàn bộ triples.

Nguồn KBC:

[facebookresearch/kbc](https://github.com/facebookresearch/kbc)

Tải nguồn:

```bash
cd /workspace/workspace

mkdir -p dataset_audit/kbc
cd dataset_audit/kbc

wget -c \
  https://dl.fbaipublicfiles.com/kbc/data.tar.gz \
  -O data.tar.gz
```

KBC sử dụng tên dataset `WN` thay cho `WN18`.

Giải nén:

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

**Kết quả đã kiểm tra:**

| Split | KG-BERT | KBC | Chênh lệch |
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

**Kết luận:** KG-BERT WN18 và KBC/WN cùng triples ở cả ba splits sau ID mapping;
dùng KG-BERT giữ synset IDs để căn chỉnh text/images.

---

## 4. Chuẩn bị raw structural dataset

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

Chạy:

```bash
cd /workspace/workspace/DSoM

./dsom/bin/python scripts/prepare_wn18_raw.py
```

Output đã kiểm tra:

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

## 5. Preprocessing structural dataset

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

Chạy:

```bash
cd /workspace/workspace/DSoM

./dsom/bin/python scripts/preprocess_wn18.py
```

Vocabulary kỳ vọng:

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

**Lưu ý:** `data/WN18/ent_id` là mapping chuẩn cho rows cả text/image matrices.

---

## 6. Audit structural data

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

Đã kiểm tra:

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

**Structural preprocessing hoàn tất.**

---

## 7. Text modality WN18

### 7.1. Nguồn descriptions

```text
/workspace/workspace/kg-bert/data/WN18/entity2text.txt
```

Ví dụ:

```text
14854262    stool, solid excretory product evacuated from the bowels
00590383    chieftainship, the position of chieftain
```

Khác FB15K-237 sử dụng `entity2textlong.txt`, WN18 sử dụng `entity2text.txt`.

### 7.2. Độ phủ text

Đã đối chiếu keys trong `entity2text.txt` với `data/WN18/ent_id`.

Đã kiểm tra:

```text
WN18 entities:        40943
Valid descriptions:   40943
Coverage:             100%
Missing descriptions: 0
```

Không cần random imputation cho missing descriptions ở WN18.

### 7.3. Trích xuất BERT features

Script:

```text
scripts/build_wn18_text_features.py
```

Giữ nguyên thiết lập BERT của FB15K-237 reproduction:

| Tham số | Giá trị |
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

Output rows căn chỉnh theo `data/WN18/ent_id`.

Chạy:

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

### 7.4. Kiểm tra BERT

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

**Trạng thái thực tế: PASS.**

---

## 8. Image modality WN18

### 8.1. Nguồn ảnh và alignment

Nguồn:

```text
/workspace/workspace/fb15k_wn18_images.zip
```

Định dạng WN18 image directory:

```text
wn18-images/nXXXXXXXX/image.JPEG
```

Alignment ảnh→entity:

```python
synset = "n" + entity
```

Entity index lấy từ:

```text
data/WN18/ent_id
```

Độ phủ ảnh đã kiểm tra:

| Chỉ số | Result |
|---|---:|
| Total entities | 40.943 |
| Entities có ảnh | 7.801 |
| Missing images | 33.142 |
| Image coverage | 19,05% |
| Missing image rate | 80,95% |
| Image files in archive | 70.385 |

### 8.2. Trích xuất image features

Script:

```text
scripts/build_wn18_image_features.py
```

Cấu hình nhất quán với FB15K-237 reproduction:

| Tham số | Giá trị |
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

### 8.4. Trích xuất features đầy đủ

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

### 8.5. Kiểm tra image features

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

**Output thực tế đã kiểm tra:**

```text
Shape: (40943, 1000)
Dtype: float32
Finite: True
Real images: 7801
Missing: 33142

WN18 IMAGE AUDIT: PASS
```

---

## 9. Files dataset hoàn chỉnh

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

Các shape chính kỳ vọng và đã xác minh:

| Artifact | Shape |
|---|---|
| `train.pickle` | (141442, 3) |
| `valid.pickle` | (5000, 3) |
| `test.pickle` | (5000, 3) |
| `dscp_vec.pickle` | (40943, 768) |
| `img_vec.pickle` | (40943, 1000) |

Mọi model feature arrays đều có giá trị finite.

---

## 10. Các điểm cần lưu ý

### Structural dataset

**Đã xác minh tương đương chính xác KBC/WN sau entity/relation ID mapping.**

So toàn bộ triples của train/valid/test, không chỉ dataset sizes.

### Text embeddings

Đã xác minh descriptions phủ 100%.

Dùng `bert-base-uncased` và mean pooling hidden layer áp chót chưa chứng minh
embeddings khớp chính xác pretrained weights/pooling/preprocessing tác giả.

### Image embeddings

Đã kiểm tra nguồn ảnh, coverage, integrity, quy tắc chọn ảnh và output shape.

Chưa xác minh hoàn toàn image selection, encoder checkpoint, preprocessing
và feature layer đúng paper.

### Ảnh bị thiếu

80.95% WN18 entities không có ảnh trong archive này.

Random feature imputation thuộc setup reproduction; cần ghi tỷ lệ thiếu ảnh
cao khi phân tích teacher/student.

### Khả năng tái lập

Để chạy lại đầy đủ, ghi:

- Git commit SHA.
- Phiên bản thư viện/model.
- Random seed.
- Dataset sources/archive checksums.
- Cấu hình BERT/ViT.
- Feature extraction scripts.
- Thống kê modality coverage WN18.

---

## 11. Trạng thái và bước tiếp theo tại thời điểm viết

**Đã hoàn thành:**

- [x] Xác định nguồn WN18.
- [x] So triples KG-BERT/KBC.
- [x] Kiểm tra MKGformer ZIP.
- [x] Chuẩn bị raw structural triples.
- [x] Preprocess structural dataset.
- [x] Audit mappings/filtering/probabilities.
- [x] Xác nhận text phủ 100%.
- [x] Trích xuất/audit BERT embeddings.
- [x] Xác nhận ảnh phủ 19.05%.
- [x] Trích xuất/audit ViT embeddings.

**Còn lại tại thời điểm viết:**

- [ ] Audit WN18 teacher config/paper hyperparameters.
- [ ] Chạy teacher smoke.
- [ ] Train teacher.
- [ ] Distill student.
- [ ] So student với Table 3.
- [ ] Ghi deviations/ablations.

**Kết luận tại thời điểm viết:** Inputs WN18 đã chuẩn bị và structural audit
hoàn tất theo BERT/ViT pipeline này. Khi ghi hồ sơ này chưa train/reproduce
Table 3.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="1-environment--paths"></a>
<a id="10-reproduction-caveats"></a>
<a id="11-current-status-and-next-step"></a>
<a id="2-dataset-provenance--nguồn-dữ-liệu"></a>
<a id="3-verify-structural-triples-against-kbc"></a>
<a id="4-prepare-raw-structural-dataset"></a>
<a id="5-preprocess-structural-dataset"></a>
<a id="6-structural-data-audit"></a>
<a id="7-wn18-text-modality"></a>
<a id="71-description-source"></a>
<a id="72-text-coverage"></a>
<a id="73-bert-feature-extraction"></a>
<a id="74-bert-verification"></a>
<a id="8-wn18-image-modality"></a>
<a id="81-image-source-and-alignment"></a>
<a id="82-image-feature-extraction"></a>
<a id="84-full-feature-extraction"></a>
<a id="85-image-verification"></a>
<a id="9-final-dataset-files"></a>
<a id="dsom-reproduction-notes--wn18-dataset-preparation"></a>
<a id="missing-images"></a>
<a id="reproducibility"></a>
