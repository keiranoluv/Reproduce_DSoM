# DB15K — Truy nguồn raw data và sensitivity trên clean split

Ngày 2026-10-23: source audit, clean split, smoke và full teacher→student
retraining **PASS**. Hash artifacts gốc giữ nguyên sau training. Reload
selected student độc lập, evaluate full test, metrics khớp chính xác.

## Phát hiện từ nguồn

Validation contamination có ngay trong **MMRNS raw files đang công bố**, trước
preprocessing repo này. Kết luận chỉ áp dụng public files kiểm tra ngày này,
không xác định private files tác giả dùng để tạo số liệu paper.

[MMRNS README](https://github.com/quqxui/MMRNS#datasets) dẫn tới
[Google Drive datasets](https://drive.google.com/drive/folders/1sFC-P9RKnikqNXjmLcj0IX7x5zvRs-Yj),
thư mục DB15K là [MMKB-DB15K](https://drive.google.com/drive/folders/1eWW4QquPkZU2hEVONkJfLGpsBY_IpwR7).

Ba giai đoạn được đối chiếu độc lập đều khớp:

1. Bản tải mới từ public folder khớp từng byte `../dataset_audit/MMRNS/MMKB-DB15K/{train,valid,test}.txt` đã có.
2. Files đó khớp từng byte `src_data/MMKB-DB15K/{train,valid,test}`.
3. Encode theo `ent_id`/`rel_id` hiện có tạo đúng `data/MMKB-DB15K/{train,valid,test}.pickle`, kể cả thứ tự rows.

Bản tải mới/retrieval record ở `runs/audit/db15k_raw_source_20261023/`.
Public IDs, URLs, SHA256 và counts trong [source-trace report](../audits/db15k_source_trace.json).
`src/process_datasets.py` ánh xạ từng row một-một, không resample validation.

| Original split | Rows | Unique triples | Duplicate rows |
|---|---:|---:|---:|
| Train | 79,222 | 79,222 | 0 |
| Validation | 9,904 | 7,481 | 2,423 |
| Test | 9,902 | 9,902 | 0 |

Unique exact overlaps gốc: train/validation **6,584**, validation/test **824**,
train/test **0**. Chỉ **73** validation triples không có ở cả train/test.
Ngoài ra **131** training facts đảo chiều official test fact dưới *cùng* relation ID.

## Protocol CleanV1

Giữ benchmark data, features, result JSONs, baseline checkpoints gốc. Files mới
dùng `MMKB-DB15K-CleanV1` riêng trong `data/`, `src_data/`, `runs/clean_sensitivity/`.

1. Giữ official test triples/thứ tự; copy raw/processed test từng byte.
2. Bỏ hẳn validation gốc, không thêm 73 novel facts vào training pool/filters.
3. Loại 131 training facts cùng `(min(head,tail), relation, max(head,tail))` với test; lưu facts loại riêng.
4. Từ 79,091 training facts còn lại, giữ lại 9,904 validation facts xác định bằng NumPy `default_rng(2022)`. Shuffle sorted groups, phân quota gần theo tỷ lệ relations rồi lấp phần còn lại; hai hướng cùng relation vào cùng split.
5. Chỉ chấp nhận holdout group nếu mọi entity/relation của eligible pool vẫn có ít nhất một occurrence training.
6. Giữ mappings đầy đủ, copy BERT/ViT features không đổi values/row alignment.
7. Tạo filters từ **chỉ ba splits mới**. NDKD masks vẫn training triples/reciprocals qua `Dataset.train_to_skip`.

| CleanV1 split | Triples | Entities represented | Relations represented |
|---|---:|---:|---:|
| Train | 69,187 | 12,783 | 273 |
| Validation | 9,904 | 7,893 | 196 |
| Test | 9,902 | 7,989 | 191 |

Cả ba split pairs không có exact-triple hoặc same-relation undirected-group
overlap, không duplicates. Mapping vẫn 12,842 entities/279 relations, khớp
features gốc. Training gốc chứa 12,785 entities/273 relations; hai entities
mất occurrence duy nhất khi loại test reversals. Test entities không có trong
training tăng từ 57 lên 59.

Protocol, per-relation counts, input/output và original hashes nằm trong
`data/MMKB-DB15K-CleanV1/split_manifest.json`. Builder không ghi đè output directories.

## Retraining và chọn kết quả

Teacher/student đều train mới; student chỉ load **teacher mới**. Cấu hình
khớp Exp01: seed 2022, ComplExMDR rank 2000, batch 1000, Adagrad LR 0.1,
N3 0.01, tối đa 200 epochs, validation mỗi 3 epochs, giữ early stopping gốc.

Student: NDKD, source-frozen RC, temperature 4, KD weight 2, alpha/beta 1/1,
bảy strategies, hidden 1024, binary reward +1/-10. Giữ scheduler `min` theo dõi
validation Hits@10 dù ngược chiều; đổi ở đây sẽ thêm yếu tố gây nhiễu.

Chọn teacher theo validation Hits@10 với oracle minimum modality rank hiện có;
chọn student bằng validation MRR. Dùng student `final_result` so test; teacher
oracle không phải deployable student score. Code log test trong training nhưng
không dùng chọn checkpoint.

Bật local logging; W&B opt-in `DSOM_WANDB=1`. Lưu exact commands, source archive
và digest, Git HEAD, working-tree patch, manifest, logs, checkpoints, comparison
summary. Archive cần thiết vì code mới có thể chưa commit.

## Lệnh chạy

Từ thư mục gốc repository:

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

Hai source tools nhận `--source` trỏ bản raw source khác. Nếu chuẩn bị từ bản
tải mới, truyền `upstream_downloads.json` của bản đó thay retrieval record October 23.

Smoke PASS tại `runs/clean_sensitivity/MMKB-DB15K-CleanV1/smoke/20261023_043857_3369024/`.
Subsets nhỏ chỉ kiểm tra thực thi, không benchmark. Full output còn lưu trong
`logs/db15k_clean_v1_full.log`.

Các bước kiểm chứng trước full retraining:

- Downloader chạy thành công với ba public file IDs.
- Hai focused tests PASS: reversal grouping/determinism/partition coverage và giữ rare entities/relations trong train.
- Generated data qua checks hash preservation, deterministic reconstruction, duplicates/overlaps, exact test preservation, filters.
- Dataset loader thật trả shape `(12842, 558, 12842)` và 138,374 reciprocal training rows. Mọi held-out answers không có trong training-neighbor masks ở cả hai chiều.
- Teacher→student smoke PASS, gồm source-frozen optimizer assertion và load teacher mới.
- Python lint, shell syntax, `git diff --check` PASS cho workflow mới.

## Giới hạn diễn giải

Đây là **sensitivity analysis**, không ước lượng tác động nhân quả riêng của
validation contamination. Nó đồng thời giảm train, loại test reversals, đổi
validation distribution và all-true filters do bỏ facts chỉ có validation cũ.
Giữ cold test entities, chưa phát hiện semantic inverse relations khác IDs.
Features tái dựng và một seed vẫn là giới hạn. Không thay original paper-protocol
score bằng clean score nếu không ghi nhãn split khác.

## Kết quả full

Full run: `runs/clean_sensitivity/MMKB-DB15K-CleanV1/full/20261023_043918_3369511/`.

Teacher chạy 65 epochs trong 763.98 giây, chọn epoch 54 theo validation Hits@10.
Tại đó valid MRR 0.422351, Hits@10 0.575323, là oracle metrics hiện có.
Teacher validation contamination gốc có selected valid MRR 0.944744; tương
phản này không phải causal estimate có kiểm soát.

Student chạy 38 epochs trong khoảng 1,523.26 giây, chọn epoch 27 theo clean
valid MRR. Hai stages tổng khoảng 38.1 phút train/eval. Không reuse teacher
checkpoint gốc để train student này.

| Student protocol | Epoch tốt nhất | Selected validation MRR | Test MRR | Test H@1 | Test H@3 | Test H@10 |
|---|---:|---:|---:|---:|---:|---:|
| Original Exp01, contaminated validation | 63 | 0.941991 | 0.413046 | 0.331701 | 0.454656 | 0.565744 |
| CleanV1 sensitivity | 27 | 0.384404 | 0.380746 | 0.301101 | 0.418754 | 0.532721 |

CleanV1 test MRR chênh **-0.032300**, tức **-3.2300 điểm phần trăm**. Test MR
đổi từ 326.1553 sang 386.2926. Cùng official test triples, mỗi protocol dùng
filters tương ứng.

Artifacts tương đối từ full run folder trên:

- `summary.json`: original/clean comparison và stage result paths.
- `teacher/checkpoints/0/result.json`: metrics/config teacher hoàn thành.
- `student/checkpoints/0/result.json`: metrics/config student hoàn thành.
- `checkpoint_verification.json`: reload best student đánh giá test độc lập, khớp chính xác `final_result`.
- `split_manifest.json`: input/output/original hashes.
- `environment.json`: Python 3.12.3, NumPy 2.5.3, PyTorch 2.7.0+cu128, CUDA 12.8, RTX 5090; training/split seeds đều 2022.
- `source_snapshot.tar.gz`, SHA256, Git HEAD, working-tree patch: provenance source lúc launch.

Launcher hoàn tất kiểm tra original artifacts, in `CLEAN SENSITIVITY PIPELINE: PASS`,
bao gồm bảo toàn features, mappings, split files, result JSONs và Exp01 checkpoints.

Validation phản ánh disjoint holdout và thấp hơn nhiều contaminated scores.
Test cho thấy pipeline nhạy với protocol thay đổi, nhưng **không quy toàn bộ
mức giảm cho validation contamination**: training ít hơn, loại reversals,
validation distribution và filtering cùng đổi. So một seed là exploratory,
giữ cả hai bộ kết quả có nhãn.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="cleanv1-protocol"></a>
<a id="commands"></a>
<a id="db15k-raw-source-trace-and-clean-split-sensitivity"></a>
<a id="full-results"></a>
<a id="interpretation-limits"></a>
<a id="retraining-and-result-selection"></a>
<a id="source-finding"></a>
