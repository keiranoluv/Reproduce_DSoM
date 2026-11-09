# Table 7 — Workflow teacher/student WN9

Trạng thái: **người dùng hoàn tất full WN9 teacher/student và checkpoint
evaluation; kiểm tra CPU/files PASS ngày 2026-10-24**. Đủ dòng DSoM theo
source-frozen protocol; chưa train competitors. Trợ lý không chạy GPU.
Chuẩn bị dữ liệu ở [WN9_SOURCE_AUDIT.md](WN9_SOURCE_AUDIT.md).

## Full run đã ghi nhận — 2026-10-24

Suite: `runs/table7/full/20261023_173538_401388/`, timestamp thư mục là UTC.
Training source `cc00d7e94c57266a2e5ebfd6df56e483f17ef45a`, seed 2022,
policy optimizer `source_frozen`. Hai stages/E2E summary đều **PASS**;
`full_table7_dsom_row_completed=true`, `exact_author_protocol_reproduced=false`.
Bằng chứng: [table7_results_20261024.json](table7_results_20261024.json).

| Thiết lập | MR | MRR | H@1 | H@3 | H@10 |
|---|---:|---:|---:|---:|---:|
| Paper DSoM | — | 0.921300 | 0.905200 | — | 0.955600 |
| Reproduction, filtered | 6.131160 | 0.919634 | 0.903336 | 0.928355 | 0.949583 |
| Reproduction, raw | 14.506444 | 0.618763 | 0.520470 | 0.669447 | 0.810083 |

So [paper Table 7](https://arxiv.org/html/2507.20738v1#A4), filtered MRR/H@1/H@10
chênh **-0.167 / -0.186 / -0.602 điểm phần trăm**. Mức gần này phụ thuộc filtered
evaluation. Raw MRR chênh -30.254 pp; evaluation setting ảnh hưởng lớn hơn
nhiều gap filtered so paper. Hai setting dùng cùng checkpoint chọn bằng
filtered valid MRR, không tìm checkpoint riêng cho raw. Chưa giải quyết
protocol paper, chứng minh thống kê hoặc xác minh learned RC.

Teacher chọn epoch **33**, dừng epoch 44; student chọn epoch **15**, dừng
epoch 26. Selected validation MRR student .916170. Independent checkpoint
evaluation lưu đủ **1,337 validation và 1,319 test queries mỗi chiều** RHS/LHS
reciprocal. Không đưa teacher oracle scores vào bảng so sánh. Tracking:
[teacher](https://wandb.ai/keiranoluv-hcmus/DSoM-Reproduction/runs/cj92k47u),
[student](https://wandb.ai/keiranoluv-hcmus/DSoM-Reproduction/runs/8k36o3o8).

CPU review chạy lại summarizer validation nhưng chặn ghi file, yêu cầu summary
text sinh ra khớp chính xác JSON/Markdown cũ. Kiểm tra input hashes, teacher/student
checkpoint/result hashes, liên kết teacher→student, chọn epoch theo validation,
evaluation hash/coverage và metrics reloaded đã lưu so training (tolerance 2e-6).
Không import Torch, inference checkpoint hoặc training; artifacts gốc giữ
nguyên từng byte. Giới hạn inverse facts, features tái dựng, SUM NDKD, scheduler
và một seed tiếp tục áp dụng.

## Stages độc lập và ghép E2E

- `scripts/run_wn9_teacher.py`: train multimodal WN9 teacher mới qua `learn.py`.
- `scripts/run_wn9_student.py`: train structural student mới qua `kd_learn.py`, nhận completed/validated WN9 teacher stage.
- `scripts/run_table7.py`: chạy hai stages tuần tự, tổng hợp raw/filtered evaluation của student reload độc lập.
- `scripts/eval_wn9_student.py` / `src/wn9_evaluation.py`: chấm selected student raw/filtered độc lập, không teacher minimum/oracle ranks.
- `scripts/wn9_workflow.py`: metadata, configs, subprocess, checkpoint/provenance validation và summary dùng chung; plan không import Torch.

Không sửa implementation training, loss, evaluator dataset hay model hiện có.
Teacher/student chạy process riêng, giải phóng GPU giữa stages. Evaluator chạy
process riêng sau khi student train xong. Teacher lỗi thì không launch student;
student/eval lỗi thì không có E2E PASS, giữ logs và `failure.json` để kiểm tra.

Standalone và E2E mặc định **plan**: kiểm tra inputs/hashes WN9, không Torch/CUDA
hoặc training. E2E plan đánh dấu student checkpoint pending; chỉ tạo student
command thực sau khi chọn/validate teacher checkpoint mới. `plan.json` và
`command.sh` mỗi executed stage ghi lệnh thực tế.

## Protocol mặc định

Full teacher: WN9 6555 entities / 18 reciprocal relations, rank 2000, Adagrad LR
.1, batch 1000, N3 .01, init .001, modality split S/V/D, alpha 1, scale 20,
tối đa 200 epochs, validation mỗi 3, early stopping 10, seed 2022. Giữ scheduler
`min`. Chọn teacher bằng **filtered validation Hits@10 lớn nhất**, dùng minimum
rank giữa modalities (oracle diagnostic) như released code. `best_mrr` là MRR
của epoch được chọn, không nhất thiết maximum MRR. Teacher metrics không phải
final prediction performance Table 7.

Student giữ structural ComplEx rank/optimizer/regularizer/schedule, CE/KD weights
1/2, temperature 4, NDKD alpha/beta 1/1, RC bảy actions binary +1/-10, hidden
1024, policy temperature 1, không normalize policy input. Teacher đóng băng.
Chọn student checkpoint bằng **filtered validation MRR lớn nhất**.

Default policy optimizer **source_frozen** khớp main baseline trước đó;
khởi tạo/sample policy nhưng không optimize, không xác minh learned RL.
`--policy-optimizer Adam --policy-lr 0.00001` là learned-RC adaptation riêng,
giữ suite riêng. Full Adam cần ghi nonzero policy updates. Smoke chỉ kiểm tra
instrumentation; zero update hợp lệ nếu run nhỏ không có binary reward
advantage khác không.

Smoke giữ rank 2000 cùng mọi maps/features; mỗi stage 1 epoch trên 1000
reciprocal-augmented training examples, validation mỗi epoch, evaluation
64 queries mỗi chiều. Masks vẫn dùng full training-only neighbor map. Smoke
chỉ kiểm tra thực thi, không phải score Table 7. Full E2E luôn train **full
teacher và student mới**, không resume/reuse smoke checkpoint.

## Đánh giá raw và filtered

Reload student, kiểm tra unimodal `ComplExMDR` đúng WN9 shape, evaluate
valid/test RHS và reciprocal LHS. Hai protocol cùng queries, pessimistic ties
`>=`, loại target hiện tại. Raw giữ mọi entity khác; filtered loại thêm các
known answers trong split-union filters gốc. Đổi unsigned NumPy IDs sang long
indices. Batch 2000 và `get_queries @ get_rhs` / `score` khớp legacy student
ranking. Không dùng oracle minimum hoặc target-aware action selection.

Full student kiểm tra đủ valid/test queries, filtered validation/test MRR
reload khớp selected training checkpoint, cùng test Hits@1/3/10 (tolerance 2e-6).
Smoke dùng evaluation subset seed 2022 cố định, khác random smoke subsets
của trainer legacy nên không bắt buộc metrics bằng nhau.

[DSoM Table 7](https://arxiv.org/html/2507.20738v1#A4) báo MRR .9213, H@1 .9052,
H@10 .9556. Protocol chính xác chưa rõ: [DySarl §4.1.3](https://openreview.net/pdf?id=Gf94NMabeh)
báo raw, còn DSoM released evaluator dùng filtered. Workflow báo cả hai trên
**cùng checkpoint chọn bằng filtered validation**; không chọn checkpoint khác
theo raw test hoặc tuyên bố raw dùng raw-validation checkpoint selection.

Giới hạn còn giữ: inverse facts WN9, reconstructed frozen-encoder features và
tám missing-image vectors kế thừa, SUM NDKD so với MEAN paper, một seed và
scheduler gốc. Không nhận WN18 graph checkpoint. Teacher phải là completed
stage với WN9 data/feature hashes khớp, checkpoint hash verified. Full student
từ chối smoke teacher; standalone smoke student có thể dùng full teacher.

## Hướng dẫn thực thi

Lệnh chạy hiện tại, chọn từng ô/toàn bảng và prerequisites: [RUNNING.md](../reproduction/RUNNING.md).

Standalone student nhận completed teacher stage directory, không nhận
checkpoint bất kỳ; source-frozen/learned-RC adaptations giữ suite riêng.

## Outputs và kiểm tra lại

E2E: `runs/table7/{plan,smoke,full}/<UTC timestamp>/`, gồm `teacher/`, `student/`,
`summary.json`, `table7_results.md`, root/stage plans, source snapshots, commands,
training/eval logs, selected checkpoints, hashes. Independent stages tại
`runs/table7/{teacher,student}/{mode}/<UTC timestamp>/`. Mỗi lần gọi tạo thư mục
MỚI, từ chối ghi đè. `--output` chọn thư mục mới trong `runs/table7/`. Không
resume optimizer hoặc ngầm launch GPU từ plan. Verify datasets/features trước
sau, không ghi lại. Bảo vệ teacher checkpoint/results khi student training.

Cách tái tạo summary được ghi trong RUNNING.md.

`full_table7_dsom_row_completed=true` cần đủ hai full stages, liên kết fresh
WN9 checkpoints, evaluation đầy đủ và metrics khớp. Vẫn giữ
`exact_author_protocol_reproduced=false`. Smoke luôn full-row flag false,
không có paper deltas.

CPU tests gồm optimizer step multimodal teacher thật cỡ nhỏ, save/load teacher,
frozen teacher/student distillation, bảo toàn source-frozen policy, save/load
student, so raw/filtered với legacy `get_ranking`, uint64 filters, parse CLI
không CUDA setup, orchestration standalone/E2E với mocked training processes.
Tests không thay smoke GPU thực tế của người dùng.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="default-protocol"></a>
<a id="independent-stages-and-e2e-composition"></a>
<a id="outputs-and-rechecking"></a>
<a id="raw-and-filtered-evaluation"></a>
<a id="recorded-full-run--2026-10-24"></a>
<a id="table-7--wn9-teacherstudent-workflow"></a>
