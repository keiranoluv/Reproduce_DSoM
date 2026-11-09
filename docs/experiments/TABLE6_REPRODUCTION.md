# Table 6 — Ablation teacher một modality

Trạng thái: người dùng hoàn tất full training ngày 2026-10-23; kiểm tra
CPU/files **PASS**, đủ 12 ô. Ghi nhận tạm thời kết quả dưới đây. Trợ lý không
chạy GPU; benchmark artifacts cũ được bảo toàn.

## Full run đã ghi nhận — 2026-10-23

`runs/table6/full/20261023_110036_163721/` **PASS**, đủ **12/12 ô**: 7 students
train mới và 5 kết quả reuse đã kiểm tra. Training commit:
`f717d2444d05bbbe3782c78f5cb8728c6fe7812b`, seed 2022. Bảy jobs mới mất
khoảng 33 phút. Chỉ số snapshot, artifact hashes và result paths nằm trong
[table6_results_20261023.json](table6_results_20261023.json).

| Dataset | Phương pháp | Reuse | Epoch tốt nhất | Test MRR | H@1 | H@3 | H@10 | MRR paper | MRR delta (pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| MKG-Y | full | True | 42 | 0.406919 | 0.360120 | 0.424146 | 0.489861 | 0.4002 | +0.672 |
| MKG-Y | structural | False | 18 | 0.382687 | 0.349230 | 0.400864 | 0.441795 | 0.3833 | -0.061 |
| MKG-Y | visual | False | 33 | 0.375854 | 0.344536 | 0.389974 | 0.429215 | 0.3392 | +3.665 |
| MKG-Y | description | False | 27 | 0.384521 | 0.333646 | 0.403305 | 0.472212 | 0.3729 | +1.162 |
| MKG-W | full | True | 54 | 0.405840 | 0.335400 | 0.431914 | 0.537319 | 0.4090 | -0.316 |
| MKG-W | structural | False | 15 | 0.366692 | 0.308961 | 0.386640 | 0.478708 | 0.3681 | -0.141 |
| MKG-W | visual | False | 3 | 0.295746 | 0.247309 | 0.315863 | 0.384183 | 0.3118 | -1.605 |
| MKG-W | description | False | 30 | 0.400567 | 0.326158 | 0.428638 | 0.542583 | 0.3923 | +0.827 |
| MMKB-DB15K | full | True | 63 | 0.413046 | 0.331701 | 0.454656 | 0.565744 | 0.4268 | -1.375 |
| MMKB-DB15K | structural | True | 48 | 0.412724 | 0.333922 | 0.451878 | 0.562563 | 0.4134 | -0.068 |
| MMKB-DB15K | visual | False | 45 | 0.381125 | 0.306201 | 0.420521 | 0.525449 | 0.3736 | +0.753 |
| MMKB-DB15K | description | True | 18 | 0.381768 | 0.304938 | 0.418552 | 0.527924 | 0.4130 | -3.123 |

Review CPU/files tái tạo summary, kiểm tra lại input/reference SHA256 và source
snapshots, đủ bảy full configs và checkpoint files khác rỗng. Selected epoch là
maximum validation MRR toàn cục đầu tiên; test curve entry bằng `final_result`.
Mọi training losses finite; cả bảy logs có W&B links. Không inference checkpoint
mới hoặc chạy GPU. Bằng chứng trong `cpu_review.json` của suite.

Kết quả quan sát:

- Structural-only MRR gần paper trên ba datasets: sai số tuyệt đối trung bình 0.090 điểm phần trăm.
- Chênh single-teacher lớn nhất: MKG-Y visual +3.665 pp, DB15K description -3.123 pp, MKG-W visual -1.605 pp; chưa rõ nguyên nhân.
- Full-modality có MRR cao nhất cả ba datasets; hơn best single teacher +2.240 pp MKG-Y, +0.527 pp MKG-W, chỉ +0.032 pp DB15K. Một seed chưa chứng minh ý nghĩa thống kê.
- Full-modality không thắng mọi Hits metric: DB15K structural H@1 cao hơn; MKG-W description H@10 cao hơn.
- MKG-W visual chọn epoch 3, dừng sau 14 epochs; bảy runs đều early stopping, không phát hiện lỗi config/thực thi.

Kết luận tạm thời: đủ lưới Table 6 theo protocol đã ghi nhận; mức khớp paper
không đồng đều. Full rows vẫn `source_frozen`, không chứng minh lợi ích learned
RC. Giới hạn feature equivalence, SUM/MEAN, DB15K validation và một seed còn
nguyên. Không khởi chạy corrective retraining.

## Phạm vi

[DSoM v1, Table 6 và Appendix E](https://arxiv.org/html/2507.20738v1#A5)
so full DSoM với structural (S), visual (V), description (D) teachers trên
DB15K, MKG-W, MKG-Y: 4 rows × 3 datasets = 12 ô metrics. Targets MRR/H@1/3/10
trong `table6_paper_targets.json`.

Launcher reuse Exp01 multimodal teacher từng dataset và đóng băng parameters.
Single-teacher run chỉ chọn raw logits modality tương ứng; vẫn train structural
ComplEx student từ đầu với CE + N3 + NDKD. Không train visual student, xóa features
hoặc retrain teacher. Load đủ modalities vì fixed-strategy distiller tính cả ba outputs.

Core changes lúc triển khai chỉ bổ sung `visual: (0,1,0)` trong
`KD_FixedStrategy.STRATEGIES` và lựa chọn `visual` trong CLI. Strategies,
defaults, optimizer branches và loss functions hiện có không đổi.

## Reuse mặc định lịch sử và chạy mới

| Dataset | Full-modality row | S teacher | V teacher | D teacher |
|---|---|---|---|---|
| MKG-Y | Reuse Exp01 | New | New | New |
| MKG-W | Reuse Exp01 | New | New | New |
| MMKB-DB15K | Reuse Exp01 | Reuse A1 | New | Reuse A2 |

Legacy full suite đã ghi nhận chạy **7 student jobs mới** tuần tự theo thứ tự
trên, reuse 5 ô. Chỉ reuse DB15K A1/A2 sau khi kiểm tra teacher, features,
matched hyperparameters, full-data limits, checkpoint và selected test metrics.
Reference không duy nhất bị từ chối; không tự chọn run. Không tự reuse
Table 6 runs bất kỳ phát sinh sau đó.

`--rerun-existing` chạy **12 students mới**, gồm ba full-modality source-frozen
baselines; vẫn reuse teachers. Tạo thư mục mới và giữ kết quả gốc. Với legacy
lookup đó, smoke chạy 7 jobs còn thiếu: 1 epoch / 1000 examples / 64 queries
mỗi chiều. Full results reuse có nhãn reference; smoke không có paper delta
và không xác nhận full table.

Configs chép Exp01 của dataset: rank 2000, Adagrad .1, batch 1000, N3 .01,
temperature 4, CE/KD weights 1/2, NDKD alpha/beta 1/1, tối đa 200 epochs,
validation mỗi 3 epochs, early stopping 10, seed 2022. Single-modality jobs
dùng `fixed_ndkd`, không RC policy. Giữ scheduler `min` để so với runs cũ.
Training masks chỉ dùng training triples/reciprocals; validation/test dùng
benchmark filters hiện có.

## Giới hạn diễn giải

Full-modality references là **source_frozen RC** của Tables 1/2, không chứng
minh hiệu quả learned RL. A5 learned DB15K/Table 4 là protocol khác, không thay
vào đây. `--rerun-existing` vẫn giữ full row `source_frozen`. Fresh/reused có
cùng cấu hình dự định, nhưng policy initialization có thể đổi quỹ đạo RNG
minibatch so với fixed-teacher job.

NDKD giữ neighbor SUM của release, paper mô tả MEAN. DB15K overlap, features
tái dựng và một seed còn nguyên. Đủ 12 ô là hoàn tất lưới theo protocol/released
code đã ghi nhận, không phải tái lập chính xác private runs tác giả.

## Hướng dẫn thực thi

Lệnh chạy hiện tại, chọn từng ô/toàn bảng và prerequisites: [RUNNING.md](../reproduction/RUNNING.md).

Fresh registry mặc định khác suite lịch sử 7-new/5-reused; RUNNING.md phân
biệt 9-new/3-reused và bắt buộc train mới một ô.

## Artifacts và kiểm chứng

Mỗi lần gọi tạo `runs/table6/{plan,smoke,full}/<UTC timestamp>/` MỚI.
`--output` chỉ nhận thư mục mới trong `runs/table6/`. Full cần source commit
và tracked tree sạch. Lưu plan, configs/commands, source snapshot, input SHA256
(teacher/features/data/maps/reference results), `train.log` từng job, checkpoints,
`summary.json`, `table6_results.md`. Kiểm tra lại original artifacts khi summarize.
Test metrics lấy `final_result` của best-validation-MRR checkpoint, không W&B cuối.

Cách tái tạo summary được ghi trong RUNNING.md.

`full_table6_grid_completed=true` yêu cầu đủ 12 full-data cells, gồm references
reuse đã kiểm tra; smoke/partial không bật cờ này.
`exact_author_protocol_reproduced=false` giữ rõ các giới hạn trên. Summarize
đọc paper targets từ source snapshot của suite.

CPU tests thực hiện fixed-NDKD optimizer updates thật cho S/V/D và pair/average,
kiểm tra chọn logits, frozen teacher, CLI variants, reuse validation, bảy jobs
mặc định/mười hai jobs mới, input integrity, selected metrics, dừng khi lỗi và
completion flags. Synthetic optimizer tests chuyển legacy `.cuda()` sang CPU
và chặn khởi tạo CUDA.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="artifacts-and-verification"></a>
<a id="historical-default-reuse-and-fresh-runs"></a>
<a id="interpretation-limits"></a>
<a id="recorded-full-run--2026-10-23"></a>
<a id="scope"></a>
<a id="table-6--single-modality-teacher-ablation"></a>
