# Table 3: FB15K-237 và WN18

Hai cấu hình dùng RC `source_frozen`, seed 2022 và GPU RTX 5090 32 GB. Student được chọn theo validation MRR; các số dưới đây lấy từ `final_result` của checkpoint đó.

## So sánh với paper

| Dataset | Nguồn | MR ↓ | MRR ↑ | H@1 (%) | H@3 (%) | H@10 (%) |
|---|---|---:|---:|---:|---:|---:|
| FB15K-237 | Paper | 116.0000 | NR | 27.50 | 41.10 | 56.40 |
| FB15K-237 | Source-Frozen RC | 116.5569 | 0.373564 | 27.79 | 41.05 | 56.67 |
| WN18 | Paper | 7.0000 | NR | 95.00 | 96.20 | 97.40 |
| WN18 | Source-Frozen RC | 8.9588 | 0.955870 | 94.87 | 95.92 | 96.89 |

Paper Table 3 chỉ công bố MR và Hits@K; `NR` là không công bố. MR càng thấp càng tốt, MRR/Hits càng cao càng tốt. Δ = tái lập − paper, `pp` là điểm phần trăm:

| Dataset | Delta MR | Delta H@1 (pp) | Delta H@3 (pp) | Delta H@10 (pp) |
|---|---:|---:|---:|---:|
| FB15K-237 | +0.5569 | +0.29 | -0.05 | +0.27 |
| WN18 | +1.9588 | -0.13 | -0.28 | -0.51 |

Các chỉ số Hits gần paper trên cả hai dataset. Mỗi cấu hình chỉ có một seed và giữ các khác biệt về policy, đặc trưng mô tả bên dưới.

## Cấu hình và checkpoint

| Tham số | Giá trị |
|---|---|
| Student model | ComplExMDR, chỉ structure |
| Teacher modalities | Structure, Image, Description |
| Student rank | 2000 |
| Batch size | 1000 |
| Optimizer | Adagrad |
| Learning rate | 0.1 |
| Scheduler mode | min |
| Regularizer | N3 (0.01) |
| KD loss | kdrl |
| Teacher strategy | rc |
| RC strategy count | 7 |
| KD temperature | 4 |
| KD CE weight | 1 |
| KD KL weight | 2 |
| DKD alpha / beta | 1 / 1 |
| RC policy optimizer | source_frozen |
| RC policy temperature | 1 |
| RC hidden size | 1024 |
| RC reward / penalty | 1 / 10 |
| RC reward mode | binary |
| Validation interval | 3 epochs |
| Max epochs | 200 |
| Early stopping | 10 |
| Checkpoint selection | Best Validation MRR |

| Dataset | Epoch tốt nhất | Total epochs | Best Valid MRR |
|---|---:|---:|---:|
| FB15K-237 | 60 | 71 | 0.379939 |
| WN18 | 21 | 32 | 0.953347 |

Cả hai dừng bằng early stopping. Nhánh thực nghiệm là `experiment/wn18-table3`, dựa trên baseline `experiment/source-frozen-rc`; tag lưu kết quả là `table3-source-frozen-20261020`. Commit baseline FB15K-237: `1dfabf9b41a3a5ef700170f4227ca2c4c48a0e36`.

## Policy source-frozen hoạt động thế nào?

Paper mô tả policy học bằng REINFORCE để chọn tổ hợp teacher. Trong source công bố, `Distiller.get_learnable_parameters()` chỉ trả student parameters và `KD_RL` không override hàm này, nên policy không nằm trong optimizer dù có gradient.

Fork biểu diễn hành vi đó bằng `--policy_optimizer source_frozen`: vẫn lấy mẫu bảy strategy, tính reward và REINFORCE loss, nhưng chỉ cập nhật student. Smoke WN18 xác nhận policy có gradient khác không và weight update bằng zero. Muốn đánh giá đóng góp của learned RC cần dùng cấu hình policy có optimizer riêng và đối chứng tương ứng.

## Teacher WN18

Teacher ba modality dùng ComplExMDR, rank 2000, batch 1000, Adagrad LR 0.1, N3 và scheduler `min`. Nó chọn epoch 39 theo validation Hits@10, chạy tổng cộng 50 epoch. Tại checkpoint đó:

| Chỉ số | Validation | Test |
|---|---:|---:|
| MR | n/a | 7.4043 |
| MRR | 0.959601 | 0.962368 |
| Hits@1 | n/a | 95.48% |
| Hits@3 | n/a | 96.73% |
| Hits@10 | 97.49% | 97.62% |

Evaluator teacher lấy minimum ground-truth rank giữa các modality. Cần đọc đây là oracle metric; nó dùng nhãn để biết nhánh nào xếp hạng đúng nhất. Chỉ số student trong bảng so sánh dùng inference của structural student.

## Dữ liệu

WN18 có 40.943 entity, 18 base relation và 36 relation khi thêm reciprocal. Triples đã được đối chiếu với KBC/WN sau mapping ID. Archive có ảnh cho 7.801 entity (19.05%); các hàng thiếu dùng vector ngẫu nhiên. Chi tiết nằm trong [ghi chép chuẩn bị WN18](../../archive/setup/wn18-preparation.md).

WN18 có nhiều inverse facts giữa các split. Đặc trưng BERT/ViT được dựng lại và chưa xác nhận trùng quy trình tác giả. Kết quả trên trang này dùng WN18, không phải WN18RR.

## File kết quả trên máy

FB15K-237:

- Teacher: `runs/teacher/FB15K-237/full/20261017_000249/checkpoints/0/m-20261017_0002-n-teacher_fb15k237_full_20261017_000249.pth`.
- Student: `runs/student/FB15K-237/table3_source_frozen_rc/full/20261020_121832/checkpoints/0/student-20261020_1218-n-table3_source_frozen_rc_full_20261020_121832.pth`.
- Result: `runs/student/FB15K-237/table3_source_frozen_rc/full/20261020_121832/checkpoints/0/result.json`.

WN18:

- Teacher: `runs/teacher/WN18/full/scheduler-min/20261020_190522/checkpoints/0/m-20261020_1905-n-teacher_wn18_full_scheduler-min_20261020_190522.pth`.
- Teacher result: `runs/teacher/WN18/full/scheduler-min/20261020_190522/checkpoints/0/result.json`.
- Student: `runs/student/WN18/table3_source_frozen_rc/full/20261020_201526/checkpoints/0/student-20261020_2015-n-table3_source_frozen_rc_full_20261020_201526.pth`.
- Student result: `runs/student/WN18/table3_source_frozen_rc/full/20261020_201526/checkpoints/0/result.json`.

Các file này được giữ ngoài Git. Để chạy mới, xem [lệnh Table 3](../../reproduction/running.md#table-3).

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="1-experiment-information"></a>
<a id="2-student-table-3-comparison"></a>
<a id="3-training-configuration"></a>
<a id="5-paper-intended-vs-released-implementation"></a>
<a id="6-data-and-evaluation-limitations"></a>
<a id="7-local-experiment-artifacts"></a>
<a id="8-conclusion"></a>
<a id="differences-reproduction-minus-paper"></a>
<a id="dsom-table-3-reproduction--source-frozen-rc"></a>
<a id="paper-intended-algorithm"></a>
<a id="released-implementation"></a>
<a id="reproduction-modification"></a>
<a id="training-summary"></a>
