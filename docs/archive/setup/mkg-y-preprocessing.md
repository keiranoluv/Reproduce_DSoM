# Chuẩn bị graph MKG-Y cho Exp01

Trang này ghi bước tiền xử lý MKG-Y đầu tiên. Lúc viết, preprocessing đã xong và đang chờ kiểm tra file đầu ra. Kết quả teacher/student hoàn tất sau đó nằm trong [Table 1](../../research/tables/table-1-mkg-y-mkg-w.md).

## Dữ liệu nguồn

MKG-Y lấy từ [MMRNS](https://github.com/quqxui/MMRNS), dùng `train.txt`, `valid.txt`, `test.txt` nguyên bản. Dataset có 15.000 entity, 28 relation, 21.310 train, 2.665 validation và 2.663 test triple. Các split được kiểm tra không có duplicate hoặc exact overlap.

## Các bước đã dùng

Đưa raw triples vào repo từ local staging:

```bash
mkdir -p src_data/MKG-Y

cp /workspace/workspace/dataset_audit/MMRNS/MKG-Y/train.txt \
   src_data/MKG-Y/train

cp /workspace/workspace/dataset_audit/MMRNS/MKG-Y/valid.txt \
   src_data/MKG-Y/valid

cp /workspace/workspace/dataset_audit/MMRNS/MKG-Y/test.txt \
   src_data/MKG-Y/test
```

Gọi script tiền xử lý:

```bash
cd /workspace/workspace/DSoM
./dsom/bin/python -m scripts.prepare_mkgy_dataset
```

Script gọi `src.process_datasets.prepare_dataset()` và từ chối output đã tồn tại. ID entity/relation được gán theo vocabulary đã sort từ cả ba split.

## Đầu ra

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

Bước này chỉ tạo dữ liệu graph. Trích xuất văn bản/ảnh và training được làm sau khi kiểm tra đầu ra. Để chuẩn bị lại với đường dẫn cụ thể, xem [hướng dẫn dữ liệu hiện tại](../../reproduction/data.md).

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="data-provenance"></a>
<a id="dsom-table-1--mkg-y-dataset-preprocessing"></a>
<a id="expected-output"></a>
<a id="notes"></a>
<a id="objective"></a>
<a id="reproduction-status"></a>
<a id="step-1-prepare-raw-triples"></a>
<a id="step-2-preprocess"></a>
