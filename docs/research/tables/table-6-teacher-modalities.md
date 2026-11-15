# Table 6: chọn từng modality của teacher

Table 6 so sánh full DSoM với teacher chỉ dùng cấu trúc (S), ảnh (V) hoặc mô tả (D) trên DB15K, MKG-W và MKG-Y. Student của cả bốn phương pháp vẫn chỉ dùng cấu trúc.

## Kết quả

Suite `runs/table6/full/20261023_110036_163721/` đã hoàn tất 12 ô vào ngày 2026-10-23: 7 student train mới và 5 kết quả dùng lại. Commit training là `f717d2444d05bbbe3782c78f5cb8728c6fe7812b`, seed 2022; bảy student mới mất khoảng 33 phút.

[Bản tổng hợp JSON](reference/table6_results_20261023.json) lưu chỉ số, hash và đường dẫn result. Chỉ số dùng thang 0-1; Δ MRR trong bảng tính bằng điểm phần trăm.

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

Full có MRR cao nhất trên cả ba dataset, hơn single teacher tốt nhất lần lượt 2.240 pp ở MKG-Y, 0.527 pp ở MKG-W và 0.032 pp ở DB15K. Full không dẫn đầu mọi chỉ số: DB15K structural có H@1 cao hơn, MKG-W description có H@10 cao hơn.

Structural-only gần paper nhất, sai số MRR tuyệt đối trung bình 0.090 pp. Chênh lệch lớn hơn nằm ở MKG-Y visual (+3.665 pp), DB15K description (-3.123 pp) và MKG-W visual (-1.605 pp); chưa xác định nguyên nhân. MKG-W visual chọn epoch 3 và dừng sau 14 epoch. Cả bảy lần chạy mới dừng bằng early stopping.

## Cấu hình

Runner dùng lại multimodal teacher Exp01 và giữ trọng số đóng băng. Single-teacher student chọn raw logits của modality tương ứng, rồi train mới bằng CE + N3 + NDKD. Distiller vẫn nạp đủ ba nhánh để tính output.

Các tham số được chép từ baseline của dataset: rank 2000, batch 1000, Adagrad LR 0.1, N3 0.01, temperature 4, CE/KD weights 1/2, alpha/beta 1/1, tối đa 200 epoch, validation mỗi 3 epoch, early stopping 10, seed 2022. Scheduler giữ `min`. Neighbor mask chỉ dùng training triples và reciprocals.

S/V/D dùng `fixed_ndkd`, không có policy. Full dùng `source_frozen`, như Table 1 và 2. A5 learned-RC của Table 4 là cấu hình khác. NDKD giữ neighbor SUM của code công bố.

## Dùng lại kết quả và train mới

Suite lịch sử dùng các kết quả sau:

| Dataset | Full-modality row | S teacher | V teacher | D teacher |
|---|---|---|---|---|
| MKG-Y | Reuse Exp01 | New | New | New |
| MKG-W | Reuse Exp01 | New | New | New |
| MMKB-DB15K | Reuse Exp01 | Reuse A1 | New | Reuse A2 |

Hai kết quả DB15K A1/A2 chỉ được dùng lại sau khi kiểm tra teacher, features, cấu hình, checkpoint và chỉ số. Runner từ chối reference không duy nhất.

Với registry mới, mặc định dùng lại 3 full baseline và train 9 S/V/D student. Thêm `--rerun-existing` để train mới cả 12 student, vẫn dùng lại teacher. Xem [lệnh Table 6](../../reproduction/running.md#table-6) để chạy một ô hoặc toàn bảng.

## Đọc và kiểm tra đầu ra

Mỗi suite nằm ở `runs/table6/{plan,smoke,full}/<UTC timestamp>/`. Bên trong có plan, source snapshot, cấu hình, lệnh, hash input/reference, log, checkpoint, `summary.json` và `table6_results.md`. Số liệu lấy từ `final_result` của checkpoint đầu tiên đạt validation MRR lớn nhất.

Lần kiểm tra lại đã đối chiếu SHA256, source, cả bảy full config, checkpoint và test curve với `final_result`; kết quả nằm trong `cpu_review.json`. Cờ `full_table6_grid_completed=true` yêu cầu đủ 12 ô full, kể cả reference đã kiểm tra. `summarize` đọc giá trị paper từ source snapshot của suite.

Đặc trưng được dựng lại, DB15K có validation overlap và mỗi cấu hình chỉ có một seed. Những chênh lệch nhỏ trong bảng chưa đủ để kết luận phương pháp nào tốt hơn một cách ổn định. Xem thêm [các khác biệt với paper](../methodology/protocol.md).

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="artifacts-and-verification"></a>
<a id="historical-default-reuse-and-fresh-runs"></a>
<a id="interpretation-limits"></a>
<a id="recorded-full-run--2026-10-23"></a>
<a id="scope"></a>
<a id="table-6--single-modality-teacher-ablation"></a>
