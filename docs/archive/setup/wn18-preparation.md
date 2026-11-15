# Chuẩn bị WN18: ghi chép ngày 18/10/2026

Ghi chép này lưu nguồn và các bước tạo dữ liệu WN18 trên máy thực nghiệm cũ. Khi viết, graph cùng BERT/ViT features đã được kiểm tra; teacher/student chưa train. Kết quả về sau nằm trong [Table 3](../../research/tables/table-3-fb15k237-wn18.md). Cách chuẩn bị trên clone mới nằm ở [hướng dẫn dữ liệu](../../reproduction/data.md).

## 1. Đường dẫn lúc chạy

Repo là `/workspace/workspace/DSoM`, Python là `/workspace/workspace/DSoM/dsom/bin/python`. Nguồn dữ liệu được đặt ở:

```text
/workspace/workspace/kg-bert/data/WN18/
/workspace/workspace/fb15k_wn18_images.zip
/workspace/workspace/dataset_audit/kbc/data/WN/
```

Raw graph lưu trong `src_data/WN18/`; đầu ra preprocessing và features ở `data/WN18/`.

## 2. Nguồn graph, mô tả và ảnh

Graph và mô tả lấy từ [KG-BERT](https://github.com/yao8839836/kg-bert), đã được clone khi chuẩn bị FB15K-237:

```bash
cd /workspace/workspace
git clone https://github.com/yao8839836/kg-bert.git
```

| File | Công dụng |
|---|---|
| `train.tsv` | Training triples |
| `dev.tsv` | Validation triples |
| `test.tsv` | Test triples |
| `entities.txt` | Danh sách entity IDs |
| `entity2text.txt` | Entity descriptions |

WN18 nằm ở `kg-bert/data/WN18/`. Entity dùng synset tám chữ số, ví dụ `03964744`; thư mục ảnh có tiền tố `n`, ví dụ `wn18-images/n03964744/`. Mapping ảnh dùng `synset = "n" + entity`.

Ảnh lấy từ archive chung của [MKGformer](https://github.com/zjunlp/MKGformer), file `/workspace/workspace/fb15k_wn18_images.zip`, subtree `wn18-images/nXXXXXXXX/`. Archive có 70.385 ảnh thuộc 7.801 synset. Kiểm tra toàn ZIP bằng:

```bash
unzip -t /workspace/workspace/fb15k_wn18_images.zip \
  > /workspace/workspace/wn18_zip_integrity.log 2>&1

tail -5 /workspace/workspace/wn18_zip_integrity.log
```

Kết quả không có lỗi dữ liệu nén. Builder đọc trực tiếp ZIP nên không cần giải nén toàn bộ.

## 3. Đối chiếu KG-BERT với KBC/WN

KBC gọi dataset này là `WN`. Bản dùng để kiểm tra lấy từ [facebookresearch/kbc](https://github.com/facebookresearch/kbc):

```bash
cd /workspace/workspace

mkdir -p dataset_audit/kbc
cd dataset_audit/kbc

wget -c \
  https://dl.fbaipublicfiles.com/kbc/data.tar.gz \
  -O data.tar.gz
```

```bash
tar -xzf data.tar.gz \
  --wildcards '*/WN/train' '*/WN/valid' '*/WN/test'
```

KG-BERT dùng synset/relation name, KBC dùng numeric ID. Để so triples, sort vocabulary entity/relation rồi ánh xạ sang ID trước:

```python
entities = sorted(all_entities)
relations = sorted(all_relations)

ent2id = {entity: str(i) for i, entity in enumerate(entities)}
rel2id = {relation: str(i) for i, relation in enumerate(relations)}
```

```bash
cd /workspace/workspace/DSoM

./dsom/bin/python scripts/audit_wn18_id_mapping.py
```

| Split | KG-BERT | KBC | Chênh lệch |
|---|---:|---:|---:|
| Train | 141.442 | 141.442 | 0 |
| Valid | 5.000 | 5.000 | 0 |
| Test | 5.000 | 5.000 | 0 |

Cả train/valid/test khớp sau mapping. KG-BERT được dùng cho bước chuẩn bị để giữ synset ID, thuận tiện nối mô tả và ảnh.

## 4. Chuẩn bị và tiền xử lý graph

`scripts/prepare_wn18_raw.py` đọc triple tab-separated, kiểm tra số hàng và giữ nguyên ID nguồn:

| Input | Output |
|---|---|
| `train.tsv` | `src_data/WN18/train` |
| `dev.tsv` | `src_data/WN18/valid` |
| `test.tsv` | `src_data/WN18/test` |

```bash
cd /workspace/workspace/DSoM

./dsom/bin/python scripts/prepare_wn18_raw.py
```

Đầu ra có 141.442 train, 5.000 valid và 5.000 test triple. Kiểm tra file bằng:

```bash
ls -lh src_data/WN18/
head -3 src_data/WN18/train
```

`scripts/preprocess_wn18.py` gọi `process_datasets.prepare_dataset`:

```bash
cd /workspace/workspace/DSoM

./dsom/bin/python scripts/preprocess_wn18.py
```

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

`ent_id` có 40.943 entity và là thứ tự hàng chuẩn của cả hai mảng đặc trưng. `rel_id` có 18 base relation, mở rộng thành 36 khi thêm reciprocal. `to_skip.pickle` lưu bộ lọc lhs/rhs từ cả ba split; `probas.pickle` lưu tần suất entity từ training.

Kiểm tra vocabulary, ID liên tục, raw/processed triples, giới hạn ID, filters và tần suất bằng:

```bash
./dsom/bin/python scripts/audit_wn18_dataset.py
```

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

## 5. Đặc trưng văn bản

WN18 dùng `/workspace/workspace/kg-bert/data/WN18/entity2text.txt`. Cả 40.943 entity có mô tả hợp lệ, nên không cần điền vector cho text thiếu. Cấu hình BERT giữ như FB15K-237:

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

Attention-mask mean pooling được tính như sau:

```python
hidden = outputs.hidden_states[-2]
mask = attention_mask.unsqueeze(-1).type_as(hidden)

pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
```

Lệnh trích xuất:

```bash
cd /workspace/workspace/DSoM

./dsom/bin/python -u scripts/build_wn18_text_features.py \
  --device cuda \
  --batch-size 32 \
  --save-every 100 \
  2>&1 | tee data/WN18/build_text_features.log
```

Checkpoint trung gian là `data/WN18/dscp_features_checkpoint.npz`, đầu ra hoàn chỉnh là `data/WN18/dscp_vec.pickle`. Kiểm tra shape/dtype/finiteness:

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

## 6. Đặc trưng ảnh

Builder `scripts/build_wn18_image_features.py` tra ảnh theo `"n" + entity`, rồi đặt vector vào đúng hàng `data/WN18/ent_id`.

| Chỉ số | Result |
|---|---:|
| Total entities | 40.943 |
| Entities có ảnh | 7.801 |
| Missing images | 33.142 |
| Image coverage | 19,05% |
| Missing image rate | 80,95% |
| Image files in archive | 70.385 |

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

Transform và cách khởi tạo hàng thiếu:

```python
transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
```

```python
rng = np.random.default_rng(2022)
features = rng.standard_normal((40943, 1000)).astype(np.float32)
```

Hàng có ảnh hợp lệ được thay bằng ViT feature. Output là classifier logits 1.000 chiều của encoder này; feature layer/checkpoint và input 384 chưa được xác nhận trùng paper DSoM.

Smoke extraction xử lý tối đa 1.000 entity theo internal ID order:

```bash
cd /workspace/workspace/DSoM

./dsom/bin/python -u scripts/build_wn18_image_features.py \
  --limit 1000 \
  --batch-size 8 \
  --save-every 100 \
  2>&1 | tee data/WN18/image_smoke.log
```

Lần chạy nhỏ lưu checkpoint nhưng chưa xuất final pickle. Trích xuất đầy đủ bằng:

```bash
./dsom/bin/python -u scripts/build_wn18_image_features.py \
  --batch-size 8 \
  --save-every 200 \
  2>&1 | tee data/WN18/image_full.log
```

Builder có thể tiếp tục từ `data/WN18/img_features_checkpoint.npz`. Đầu ra gồm `img_vec.pickle` và `img_features_manifest.json`. Kiểm tra bằng:

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

```text
Shape: (40943, 1000)
Dtype: float32
Finite: True
Real images: 7801
Missing: 33142

WN18 IMAGE AUDIT: PASS
```

## 7. Bộ file hoàn chỉnh

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

| Artifact | Shape |
|---|---|
| `train.pickle` | (141442, 3) |
| `valid.pickle` | (5000, 3) |
| `test.pickle` | (5000, 3) |
| `dscp_vec.pickle` | (40943, 768) |
| `img_vec.pickle` | (40943, 1000) |

Các mảng đặc trưng đều finite. Graph đã khớp KBC/WN sau mapping, nhưng điều này chưa xác nhận BERT/ViT preprocessing trùng tác giả. Ảnh thiếu ở 80.95% entity là đặc điểm đáng chú ý của archive này.

Các bước sau ngày ghi chép là kiểm tra cấu hình teacher, smoke, train teacher/student và so Table 3. Những việc đó đã được tổng hợp ở [trang kết quả WN18](../../research/tables/table-3-fb15k237-wn18.md).

<!-- Liên kết từ các phiên bản tài liệu trước. -->
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
