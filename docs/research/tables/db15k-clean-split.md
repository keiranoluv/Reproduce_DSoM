# DB15K: kiểm tra độ nhạy với bộ chia CleanV1

Validation của DB15K công khai có nhiều triple trùng train hoặc test. CleanV1 tạo một bộ chia riêng để xem kết quả thay đổi thế nào khi dùng validation tách biệt. Teacher và student được train mới; dữ liệu và kết quả benchmark gốc được giữ để so sánh.

## Dữ liệu gốc có gì?

Ngày 2026-10-23, ba file tải lại từ [MMRNS/Drive](https://drive.google.com/drive/folders/1eWW4QquPkZU2hEVONkJfLGpsBY_IpwR7) được so từng byte với bản đã tải ở `../dataset_audit/MMRNS/MMKB-DB15K/` và bản trong `src_data/MMKB-DB15K/`. Khi encode theo mapping hiện có, chúng tạo đúng processed triples, kể cả thứ tự hàng.

| Original split | Rows | Unique triples | Duplicate rows |
|---|---:|---:|---:|
| Train | 79,222 | 79,222 | 0 |
| Validation | 9,904 | 7,481 | 2,423 |
| Test | 9,902 | 9,902 | 0 |

Có 6.584 unique triple trùng train/validation, 824 trùng validation/test và không có exact overlap train/test. Chỉ 73 validation triple vắng mặt ở cả train lẫn test. Ngoài ra, 131 training fact là chiều đảo của test fact dưới cùng relation ID.

Các overlap này có trong file công khai trước khi tiền xử lý. Kết quả kiểm tra chưa cho biết tác giả có dùng chính các file đó cho số liệu paper hay không. [Source trace](../provenance/db15k_source_trace.json) lưu file ID, URL, SHA256 và số lượng; bản tải và biên nhận nằm ở `runs/audit/db15k_raw_source_20261023/`.

## CleanV1 được tạo thế nào?

Dataset mới có tên `MMKB-DB15K-CleanV1`, lưu riêng trong `src_data/`, `data/` và `runs/clean_sensitivity/`.

1. Giữ nguyên official test, gồm thứ tự và nội dung raw/processed file.
2. Bỏ validation cũ; 73 fact riêng của nó cũng không được thêm vào train hoặc filters.
3. Loại 131 training fact cùng nhóm `(min(head,tail), relation, max(head,tail))` với test, lưu các fact loại riêng.
4. Từ 79.091 fact còn lại, chọn 9.904 fact làm validation bằng NumPy `default_rng(2022)`. Shuffle các nhóm đã sort, chia gần theo tỷ lệ relation rồi lấp phần còn lại. Hai hướng cùng relation được giữ trong một split.
5. Chỉ giữ một nhóm làm validation nếu mọi entity/relation của pool vẫn xuất hiện ít nhất một lần trong train.
6. Dùng nguyên mapping và BERT/ViT features để giữ row alignment.
7. Tạo evaluation filters từ ba split mới. NDKD neighbor vẫn chỉ lấy từ train và reciprocals.

| CleanV1 split | Triples | Entities represented | Relations represented |
|---|---:|---:|---:|
| Train | 69,187 | 12,783 | 273 |
| Validation | 9,904 | 7,893 | 196 |
| Test | 9,902 | 7,989 | 191 |

Ba cặp split không có exact overlap, same-relation undirected-group overlap hoặc duplicates. Mapping vẫn có 12.842 entity và 279 relation. Train gốc có 12.785 entity/273 relation; sau khi loại test reversals, hai entity mất occurrence duy nhất, nên số test entity vắng trong train tăng từ 57 lên 59.

`data/MMKB-DB15K-CleanV1/split_manifest.json` lưu quy tắc, số lượng từng relation và hash input/output. Builder từ chối ghi đè dataset đã có.

## Cấu hình train và kết quả

Teacher/student dùng cấu hình Exp01: ComplExMDR rank 2000, batch 1000, Adagrad LR 0.1, N3 0.01, seed 2022, tối đa 200 epoch, validation mỗi 3 epoch và early stopping gốc. Scheduler giữ `min` trên validation Hits@10.

Student dùng NDKD, RC `source_frozen`, temperature 4, KD weight 2, alpha/beta 1/1, bảy strategy, hidden 1024 và binary reward +1/-10. Nó chỉ load teacher mới của CleanV1.

Full suite: `runs/clean_sensitivity/MMKB-DB15K-CleanV1/full/20261023_043918_3369511/`. Teacher chạy 65 epoch trong 763.98 giây, chọn epoch 54 theo validation Hits@10; tại đó oracle validation MRR là 0.422351, Hits@10 là 0.575323. Student chạy 38 epoch trong khoảng 1.523,26 giây và chọn epoch 27 theo clean validation MRR. Tổng train/eval khoảng 38.1 phút.

| Student protocol | Epoch tốt nhất | Selected validation MRR | Test MRR | Test H@1 | Test H@3 | Test H@10 |
|---|---:|---:|---:|---:|---:|---:|
| Original Exp01, contaminated validation | 63 | 0.941991 | 0.413046 | 0.331701 | 0.454656 | 0.565744 |
| CleanV1 sensitivity | 27 | 0.384404 | 0.380746 | 0.301101 | 0.418754 | 0.532721 |

Test MRR giảm 0.032300, tương ứng 3.2300 pp; MR tăng từ 326.1553 lên 386.2926. Hai cấu hình đánh giá cùng official test nhưng dùng filters của bộ chia tương ứng. Teacher validation MRR ở benchmark gốc là 0.944744, cao hơn nhiều so với CleanV1.

## Cách chạy và kiểm tra lại

Các lệnh dưới đây thuộc workflow sensitivity đã dùng. `./dsom/bin/python` là virtual environment của máy chạy cũ; thay bằng Python trong môi trường hiện tại nếu cần.

```bash
# Optional fresh re-download into a NEW directory; requires the existing gdown package.
./dsom/bin/python scripts/download_db15k_raw_source.py \
  --output runs/audit/db15k_raw_source_fresh

# First-time preparation; refuses to overwrite an existing dataset.
./dsom/bin/python scripts/prepare_db15k_clean_split.py \
  --upstream-record runs/audit/db15k_raw_source_20261023/upstream_downloads.json

# Verify original artifacts, deterministic regeneration, disjointness, and filters.
./dsom/bin/python scripts/prepare_db15k_clean_split.py --verify-only

./dsom/bin/python -m pytest -q tests/test_db15k_clean_split.py
bash scripts/run_db15k_clean_sensitivity.sh --smoke
bash scripts/run_db15k_clean_sensitivity.sh --full
```

Hai source tool nhận `--source` để chọn raw source khác. Nếu tải mới, truyền `upstream_downloads.json` của lần tải đó thay record ngày 2026-10-23.

Smoke hoàn tất ở `runs/clean_sensitivity/MMKB-DB15K-CleanV1/smoke/20261023_043857_3369024/`; log full ở `logs/db15k_clean_v1_full.log`. Kiểm tra trước full bao gồm tái tạo deterministic split, overlap, hash, official test, filters và training-only neighbors. Dataset loader trả shape `(12842, 558, 12842)` với 138.374 reciprocal training row.

Trong thư mục full có:

- `summary.json`: so sánh original/CleanV1 và đường dẫn kết quả hai stage.
- `teacher/checkpoints/0/result.json`, `student/checkpoints/0/result.json`: cấu hình và chỉ số.
- `checkpoint_verification.json`: đánh giá độc lập best student, khớp `final_result`.
- `split_manifest.json`: hash dữ liệu mới và gốc.
- `environment.json`: Python 3.12.3, NumPy 2.5.3, PyTorch 2.7.0+cu128, CUDA 12.8, RTX 5090; split/training seed 2022.
- `source_snapshot.tar.gz`, SHA256, Git HEAD và working-tree patch: mã nguồn lúc chạy.

Launcher kiểm tra các file benchmark gốc sau training và xác nhận features, mapping, split, result JSON và Exp01 checkpoint còn nguyên. W&B tùy chọn qua `DSOM_WANDB=1`; mặc định lưu log trên máy.

## Diễn giải kết quả

CleanV1 đồng thời giảm số training fact, loại test reversal, thay phân phối validation và thay all-true filters. Vì nhiều yếu tố cùng đổi, mức giảm test MRR chưa thể quy riêng cho validation overlap. Bộ chia vẫn giữ cold test entity và chưa xử lý semantic inverse relation khác ID.

Đây là một lần kiểm tra độ nhạy với một seed. Khi dùng số liệu, giữ nhãn `Original Exp01` và `CleanV1` để người đọc biết kết quả thuộc bộ chia nào.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="cleanv1-protocol"></a>
<a id="commands"></a>
<a id="db15k-raw-source-trace-and-clean-split-sensitivity"></a>
<a id="full-results"></a>
<a id="interpretation-limits"></a>
<a id="retraining-and-result-selection"></a>
<a id="source-finding"></a>
