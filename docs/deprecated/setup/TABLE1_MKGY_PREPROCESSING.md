# DSoM Table 1 — Preprocessing dataset MKG-Y

## Mục tiêu

Reproduce DSoM Table 1 trên MKG-Y.

## Nguồn dữ liệu

Dataset: MKG-Y

Nguồn: MMRNS
https://github.com/quqxui/MMRNS

Raw dataset files:
- train.txt
- valid.txt
- test.txt

Thống kê kỳ vọng:
- Entities: 15,000
- Relations: 28
- Train: 21,310
- Validation: 2,665
- Test: 2,663

Audit integrity:
- Không duplicate triples trong splits
- Không train/validation overlap
- Không train/test overlap
- Không validation/test overlap

## Bước 1: Chuẩn bị raw triples

Từ thư mục gốc repo:

```bash
mkdir -p src_data/MKG-Y

cp /workspace/workspace/dataset_audit/MMRNS/MKG-Y/train.txt \
   src_data/MKG-Y/train

cp /workspace/workspace/dataset_audit/MMRNS/MKG-Y/valid.txt \
   src_data/MKG-Y/valid

cp /workspace/workspace/dataset_audit/MMRNS/MKG-Y/test.txt \
   src_data/MKG-Y/test
```

Nguồn path trên là local staging directory; đổi nếu MMRNS nằm ở nơi khác.

Không thay dataset files đã có khi chưa audit.

## Bước 2: Preprocessing

```bash
cd /workspace/workspace/DSoM
./dsom/bin/python -m scripts.prepare_mkgy_dataset
```

Gọi hàm gốc `src.process_datasets.prepare_dataset()` của DSoM.

Script từ chối ghi đè output directory đã tồn tại.

## Outputs kỳ vọng

```text
data/MKG-Y/
├── ent_id
├── rel_id
├── train.pickle
├── valid.pickle
├── test.pickle
├── to_skip.pickle
└── probas.pickle
```

Entity/relation IDs được gán bằng cách sort vocabulary trên train/validation/test.

## Ghi chú

- Giữ MMRNS splits gốc.
- Không deduplicate/chia lại.
- Tách structural preprocessing và feature extraction.
- Không train teacher/student ở bước này.
- Validate artifacts trước encoding.

## Trạng thái reproduction

Structural preprocessing hoàn tất local.
Artifact validation còn chờ xác nhận tại thời điểm viết.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="data-provenance"></a>
<a id="dsom-table-1--mkg-y-dataset-preprocessing"></a>
<a id="expected-output"></a>
<a id="notes"></a>
<a id="objective"></a>
<a id="reproduction-status"></a>
<a id="step-1-prepare-raw-triples"></a>
<a id="step-2-preprocess"></a>
