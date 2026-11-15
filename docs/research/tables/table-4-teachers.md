# Table 4: kết hợp điểm của các teacher

Trang này tổng hợp hai component student và ba cách đánh giá trực tiếp teacher: Average, ML trên validation và RC trên validation. Kết hợp với A4/A5/A9/A10, repo đã có kết quả cho đủ chín dòng Table 4.

## Kết quả student NEKD/NNKD

Suite `runs/table4_components/full/20261023_075850_052725/` đã được đối chiếu lại cấu hình, hash, checkpoint và chỉ số tại epoch được chọn:

| Component | Epoch tốt nhất | Test MRR | H@1 | H@3 | H@10 | MRR paper | MRR delta (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RC + NEKD | 75 | .411020 | .333115 | .450010 | .560038 | .4100 | +.102 |
| RC + NNKD | 42 | .412105 | .332256 | .452030 | .563775 | .4233 | -1.120 |

NNKD cao hơn NEKD 0.1084 pp MRR trong lần chạy này, nhỏ hơn chênh lệch 1.33 pp của paper. Hai cấu hình dùng learned RC với Adam và neighbor SUM của code công bố. Xem [cách tách loss](table-4-students.md).

## Teacher Average được tính thế nào?

[DSoM §5.3/Table 4](https://arxiv.org/html/2507.20738v1#S5.SS3) mô tả “average performance” của ba modal teacher. Runner lưu hai cách tính để người đọc đối chiếu:

- `teacher_mean_metrics`: trung bình chỉ số của ba modality, dùng làm kết quả chính theo cách diễn đạt của paper.
- `teacher_mean_logits`: lấy trung bình raw logits rồi xếp hạng, lưu để tham khảo.

Runner cũng lưu chỉ số từng modality. Cả hai cách trên khác oracle minimum rank trong evaluator teacher gốc.

## Meta learner trên validation

Kiến trúc theo [MoSE meta learner](https://github.com/OreOZhao/MoSE4MKGC/blob/192d9ff2882b5be6ef8239a37cbd90c5dd102b5a/src/meta_learner.py): `Linear(3,32) → ReLU → Linear(32,1) → ReLU` trên ba raw modality score của mỗi entity. Train bằng validation triples hai chiều, Adagrad LR 0.1, batch 256, trong 20 epoch. Bản cài đặt giữ chiều batch khi chỉ có một phần tử.

MoSE gốc chọn checkpoint theo test Hits@10. Ở đây số epoch được cố định trước và dùng final epoch, sau đó mới đánh giá test. Vì có thay đổi này, kết quả mang nhãn ML-on-val adaptation. Commit và hash nguồn được lưu trong `reference/table4_ensemble_sources.json`.

## RC trên validation

Source DSoM chưa đủ chi tiết để tái tạo độc lập RC-on-validation. Cấu hình `RC-on-val-adapted` ở đây dùng:

- Exp01 teacher đóng băng, không train thêm structural student.
- Raw score `[batch, entities, S/V/D]` qua flatten, dropout 0.1, `Linear(3*entities,1024)`, ReLU và `Linear(1024,7)`.
- Bảy action S, V, D, SV, SD, VD, SVD; tổ hợp nhiều modality lấy trung bình logits.
- Khi train, lấy mẫu action. Reward +1 nếu unfiltered target CE thấp hơn average-teacher CE, ngược lại -10. Reward reference là -10, nên advantage là 11 hoặc 0. Dùng Categorical log probability, không thêm entropy term.
- Adam LR `1e-5`, 20 epoch cố định, seed 2022; mỗi epoch chạy validation RHS rồi LHS, giữ duplicates và shuffle từng chiều.
- Khi đánh giá, tắt dropout, chọn argmax không dùng nhãn rồi mới áp dụng filtered ranking.

Reward reference và số epoch là lựa chọn của lần tái lập này. Policy A5 được giữ riêng. Validation dùng đủ dữ liệu gốc có overlap; chỉ số trên chính validation sau fitting được dùng để theo dõi, chưa phản ánh khả năng tổng quát hóa.

## Kết quả teacher ensemble

Suite `runs/table4_ensembles/full/20261023_103437_794958/` đã hoàn tất:

| Cách tính | Test MRR |
|---|---:|
| Teacher Average: mean metrics | 0.242729 |
| Mean logits, tham khảo | 0.329994 |
| ML-on-val | 0.408904 |
| RC-on-val | 0.400841 |

Đây là điểm đánh giá trực tiếp teacher ensemble. Các student row được lấy từ những lần train đã nêu ở trên.

## Chạy lại và tìm đầu ra

Xem [lệnh và thứ tự Table 4](../../reproduction/running.md#table-4). Chọn full component-pair suite bằng `--student-suite`; mặc định lịch sử là `20261023_075850_052725`.

Mỗi lần gọi tạo thư mục mới trong `runs/table4_ensembles/{mode}/`. Runner lưu source, input hash, cấu hình, lịch sử và head/optimizer state. Đầu ra gồm `result.json` từng method, `final_head.pth` cho ML/RC, `history.json`, `summary.json` và `table4_results.md`. `--output` cũng cần trỏ tới thư mục chưa tồn tại.

`all_nine_rows_have_results=true` xác nhận đủ chín dòng; `exact_author_protocol_reproduced=false` ghi nhận các adaptation ML/RC và cách hiểu Teacher Average còn chưa chắc chắn. Kết quả mới có một seed, dùng đặc trưng dựng lại và DB15K split gốc.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="completed-student-audit"></a>
<a id="sources-and-interpretation"></a>
<a id="table-4--remaining-teacher-rows-and-protocol-limits"></a>
