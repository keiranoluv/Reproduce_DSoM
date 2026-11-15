# Kết quả MKG-Y Exp01: bản ghi đầu tiên

Bản ghi này được viết khi MKG-Y hoàn tất và chưa chạy MKG-W. Trang [Table 1 hiện tại](../../research/tables/table-1-mkg-y-mkg-w.md) đã gộp cả hai dataset.

## Cấu hình và dữ liệu

`T1_MKGY_Exp01` dùng RTX 5090 32 GB, seed 2022, commit `39c7886f8bfd728e9c79afb11a0800ae0ac4fad7`. Teacher là ComplExMDR ba modality; student chỉ dùng cấu trúc với RC `source_frozen`.

MKG-Y lấy từ [MMRNS](https://github.com/quqxui/MMRNS), gồm 15.000 entity, 28 relation, 21.310/2.665/2.663 train/valid/test triple. Ảnh ViT có shape `(15000, 1000)`, text BERT `(15000, 768)`.

| Tham số | Giá trị |
|---|---|
| Rank | 2000 |
| Batch size | 1000 |
| Optimizer | Adagrad |
| Learning rate | 0.1 |
| Scheduler | min |
| Regularizer | N3, weight 0.01 |
| Max epochs | 200 |
| Validation interval | 3 |
| Early stopping | 10 |
| Student KD loss | kdrl |
| Teacher strategy | rc |
| Policy optimizer | source_frozen |
| KD temperature | 4 |
| CE weight | 1 |
| KL weight | 2 |
| RC hidden dimension | 1024 |
| RC reward / penalty | 1 / 10 |
| RC reward mode | binary |

Teacher chọn epoch 36 theo validation Hits@10 và chạy 47 epoch. Student chọn epoch 42 theo validation MRR 0.38296 và chạy 53 epoch. Lần full dùng toàn bộ dữ liệu, load teacher và lưu checkpoint thành công.

## Kết quả student

Lấy từ `final_result` của selected checkpoint:

| Chỉ số | DSoM Paper | Exp01 | Chênh lệch |
|---|---:|---:|---:|
| MR | Không ghi tại đây | 468.4587 | N/A |
| MRR (%) | 40.02 | 40.69 | +0.67 pp |
| Hits@1 (%) | 35.67 | 36.01 | +0.34 pp |
| Hits@3 (%) | 41.70 | 42.41 | +0.71 pp |
| Hits@10 (%) | 47.71 | 48.99 | +1.28 pp |

Giá trị đầy đủ: MR 468.45867919921875, MRR 0.4069194197654724, H@1 0.36012017726898193, H@3 0.4241456985473633, H@10 0.48986104130744934.

Run ID là `20261020_174808_2562445`; [W&B student](https://wandb.ai/keiranoluv-hcmus/DSoM-Reproduction/runs/2a9t5tjk). Teacher/student lưu dưới `runs/teacher/MKG-Y/full/` và `runs/student/MKG-Y/full/`. Log test cuối có thể khác chỉ số selected checkpoint.

Đặc trưng được dựng lại, policy giữ trọng số source-frozen và mỗi cấu hình chỉ có một lần chạy. Teacher dùng oracle modality ranking, được ghi riêng với student. Các khác biệt này vẫn cần nêu khi so kết quả với paper.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="1-experiment-information"></a>
<a id="3-training-configuration"></a>
<a id="4-training-summary"></a>
<a id="5-student-test-results"></a>
<a id="6-experiment-tracking"></a>
<a id="7-reproduction-limitations"></a>
<a id="8-conclusion"></a>
<a id="dsom-table-1--mkg-y-exp01-reproduction"></a>
