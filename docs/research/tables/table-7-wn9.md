# Table 7: DSoM trên WN9

Teacher và student WN9 đã train xong; selected student đã được nạp lại để đánh giá raw và filtered. Dòng DSoM hoàn tất với policy `source_frozen`. Repo chưa chạy lại các mô hình đối chứng khác trong Table 7.

<a id="recorded-full-run--2026-10-24"></a>

## Kết quả ngày 2026-10-24

Suite `runs/table7/full/20261023_173538_401388/` dùng timestamp UTC, commit `cc00d7e94c57266a2e5ebfd6df56e483f17ef45a` và seed 2022. [Bản tổng hợp JSON](reference/table7_results_20261024.json) lưu kết quả và hash các file.

| Thiết lập | MR | MRR | H@1 | H@3 | H@10 |
|---|---:|---:|---:|---:|---:|
| Paper DSoM | n/a | 0.921300 | 0.905200 | n/a | 0.955600 |
| Reproduction, filtered | 6.131160 | 0.919634 | 0.903336 | 0.928355 | 0.949583 |
| Reproduction, raw | 14.506444 | 0.618763 | 0.520470 | 0.669447 | 0.810083 |

So với paper, filtered MRR/H@1/H@10 thấp hơn lần lượt 0.167/0.186/0.602 điểm phần trăm. Raw MRR thấp hơn 30.254 pp. Raw và filtered dùng cùng checkpoint được chọn theo filtered validation MRR; chưa rõ tác giả dùng cách đánh giá nào cho Table 7. [DySarl §4.1.3](https://openreview.net/pdf?id=Gf94NMabeh) mô tả raw, còn evaluator DSoM công bố dùng filtered.

Teacher chọn epoch 33, dừng epoch 44. Student chọn epoch 15, dừng epoch 26, selected validation MRR 0.916170. Đánh giá độc lập có đủ 1.337 validation và 1.319 test query mỗi chiều RHS/LHS. W&B: [teacher](https://wandb.ai/keiranoluv-hcmus/DSoM-Reproduction/runs/cj92k47u), [student](https://wandb.ai/keiranoluv-hcmus/DSoM-Reproduction/runs/8k36o3o8).

## Dữ liệu và cấu hình

WN9 có 6.555 entity và 18 relation sau reciprocal augmentation. Graph lấy từ MoSE đã ghim commit; đặc trưng BERT/ViT lấy theo synset ID từ frozen encoder WN18. Teacher/student graph được train mới trên WN9. Xem [nguồn dữ liệu WN9](../provenance/wn9-sources.md).

Teacher và student dùng ComplEx rank 2000, batch 1000, Adagrad LR 0.1, N3 0.01, init 0.001, scale 20, tối đa 200 epoch, validation mỗi 3 epoch, early stopping 10, seed 2022 và scheduler `min`.

Teacher ba modality S/V/D, alpha 1, chọn checkpoint theo filtered validation Hits@10. Evaluator teacher lấy minimum rank giữa các modality; MRR tại epoch này không nhất thiết là MRR cao nhất. Các oracle teacher metric được giữ riêng với kết quả student.

Student chỉ dùng cấu trúc, CE/KD weights 1/2, temperature 4, NDKD alpha/beta 1/1. RC có bảy action, hidden 1024, policy temperature 1, binary reward +1/-10, không normalize input. Teacher đóng băng khi distill.

Policy mặc định `source_frozen` được khởi tạo và lấy mẫu nhưng không cập nhật. Tùy chọn `--policy-optimizer Adam --policy-lr 0.00001` tạo một learned-RC adaptation riêng; full Adam kiểm tra policy có weight update khác không.

## Cách đánh giá raw và filtered

Evaluator nạp lại structural student, kiểm tra đúng shape WN9, rồi chấm valid/test ở cả RHS và reciprocal LHS. Nó dùng pessimistic ties `>=` và loại target hiện tại khi đếm đối thủ. Raw giữ mọi entity khác; filtered loại thêm known answer từ filters của ba split gốc.

Ranking dùng batch 2000, `get_queries @ get_rhs` và `score` như legacy student. NumPy unsigned IDs được đổi sang long indices. Kiểm tra full yêu cầu số query đầy đủ và filtered metrics khớp training checkpoint trong sai số `2e-6`.

Smoke chạy mỗi stage một epoch trên 1.000 reciprocal training example, đánh giá 64 query mỗi chiều và vẫn dùng full training-neighbor map. Subset của evaluator cố định bằng seed 2022, có thể khác subset ngẫu nhiên của trainer nên smoke không yêu cầu metrics trùng nhau. Kết quả smoke chỉ dùng để kiểm tra thực thi.

## Chạy lại và tìm file

[Hướng dẫn Table 7](../../reproduction/running.md#table-7) có lệnh chạy cả hai stage hoặc từng stage:

- `run_wn9_teacher.py` gọi `learn.py` để train teacher.
- `run_wn9_student.py` gọi `kd_learn.py`, nhận thư mục WN9 teacher stage đã hoàn tất.
- `run_table7.py` chạy teacher, student và tổng hợp kết quả.
- `eval_wn9_student.py` / `src/wn9_evaluation.py` nạp checkpoint để đánh giá độc lập.
- `wn9_workflow.py` quản lý cấu hình, lệnh, kiểm tra file và summary.

Mỗi stage chạy process riêng. Nếu teacher lỗi, runner dừng trước student; nếu student/evaluation lỗi, log và `failure.json` được giữ lại. Mặc định `plan` chỉ kiểm tra file và hash; với cả hai stage, checkpoint teacher được điền vào student command sau khi teacher hoàn tất.

Kết quả nằm ở `runs/table7/{plan,smoke,full}/<UTC timestamp>/`; stage độc lập ở `runs/table7/{teacher,student}/{mode}/<UTC timestamp>/`. Suite gồm source, plan, `command.sh`, log, checkpoint, `summary.json`, `table7_results.md` và student `evaluation.json`. Mỗi lần chạy tạo thư mục mới, kể cả khi chỉ định `--output`.

Student plan/full cần full WN9 teacher với data/feature/checkpoint hash khớp. Smoke student có thể dùng full teacher. Checkpoint WN18 không đáp ứng yêu cầu này.

## Kiểm tra lại và khác biệt còn lại

Ngày 2026-10-24, summary và các file gốc đã được đối chiếu lại: hash input/checkpoint/result/evaluation, liên kết teacher và student, epoch được chọn, độ phủ query và chỉ số reload. Việc kiểm tra file này không train hoặc inference lại.

`full_table7_dsom_row_completed=true` xác nhận đủ hai stage full và đánh giá checkpoint. `exact_author_protocol_reproduced=false` giữ vì cách đánh giá paper còn chưa rõ. Ngoài ra còn có inverse facts giữa split, tám vector ảnh thiếu kế thừa từ WN18, đặc trưng dựng lại, NDKD SUM thay vì MEAN và một seed.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="default-protocol"></a>
<a id="independent-stages-and-e2e-composition"></a>
<a id="outputs-and-rechecking"></a>
<a id="raw-and-filtered-evaluation"></a>
<a id="table-7--wn9-teacherstudent-workflow"></a>
