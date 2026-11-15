# Chuẩn bị MKG-W cho Exp01

Ghi chép này thuộc lần chạy `T1_MKGW_Exp01`. Cách chuẩn bị trên clone mới nằm trong [hướng dẫn dữ liệu](../../reproduction/data.md); số liệu sau training nằm trong [kết quả Table 1](../../research/tables/table-1-mkg-y-mkg-w.md).

## Dữ liệu và ánh xạ

MKG-W lấy từ [MMRNS](https://github.com/quqxui/MMRNS), giữ nguyên 34.196/4.276/4.274 train/valid/test triple, 15.000 entity và 169 relation. Teacher ComplExMDR dùng ba modality, student dùng cấu trúc với RC `source_frozen`. Các tham số optimizer, scheduler, KD, early stopping và encoder theo baseline MKG-Y.

Graph dùng URI Wikidata, còn mô tả và ảnh dùng khóa DBpedia. Builder đọc `data/MKG-W/ent_id`, nối qua `src_data/MKG-W/ent_links`, rồi tìm mô tả hoặc thư mục ảnh. Hàng thiếu modality giữ Gaussian random vector có seed.

Archive ảnh gốc có khoảng 9.301 thư mục entity. Một số entity có HDF5 image feature MMRNS nhưng không có ảnh trong archive. Pipeline dùng ViT output `(15000, 1000)`, nên HDF5 vector được giữ riêng.

## Lệnh đã dùng

```bash
cd /workspace/workspace/DSoM

# Dataset stage/preprocessing: runs only when data/MKG-W is absent.
./dsom/bin/python -m scripts.prepare_mkgw_dataset

# CPU-only integrity and alignment audits. Stop and investigate if either fails.
./dsom/bin/python -m scripts.mkgw_alignment
./dsom/bin/python -m scripts.build_mkgw_text_features --audit-only
./dsom/bin/python -m scripts.build_mkgw_image_features --audit-only

# Independent GPU feature extractions; run one at a time.
./dsom/bin/python -m scripts.build_mkgw_text_features --device cuda --batch-size 16 --save-every 100
./dsom/bin/python -m scripts.build_mkgw_image_features --batch-size 8 --save-every 300

./dsom/bin/python -m scripts.validate_mkgw_features

# Enable MKG-W in the Teacher CLI, commit the entire setup, then smoke.
./dsom/bin/python -m scripts.enable_mkgw_teacher
bash scripts/smoke_test_mkgw.sh

# After smoke PASS, run full Teacher -> Student (logs to W&B).
bash scripts/run_mkgw_exp01_full.sh
```

Các đường dẫn trong lệnh thuộc bố trí dữ liệu lúc chạy. Structural preprocessing và feature extraction là hai bước riêng.

## File cần giữ cùng kết quả

`data/MKG-W/structural_manifest.json` lưu split checksum; text/image manifest lưu độ phủ và cách biểu diễn. Teacher/student nằm dưới `runs/teacher/MKG-W/` và `runs/student/MKG-W/`. Full run ghi commit SHA trong `result.json`/W&B và dùng chung W&B group.

Khi tổng hợp Table 1, đọc student `result.json` ở trường `final_result`. Giá trị paper tham chiếu: MRR 40.90%, H@1 33.56%, H@3 43.58%, H@10 55.28%. Đặc trưng BERT/ViT của lần chạy này được dựng lại và chưa xác nhận trùng assets tác giả.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="commands-repository-root"></a>
<a id="dsom-table-1--mkg-w-exp01-setup"></a>
<a id="mkg-w-specific-alignment"></a>
<a id="purpose-and-provenance"></a>
<a id="records-and-metadata"></a>
