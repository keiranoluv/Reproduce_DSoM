# DSoM Table 3 — Reproduction với Source-Frozen RC

## 1. Thông tin thí nghiệm

- Paper: Dark Side of Modalities: Reinforced Multimodal Distillation for Multimodal Knowledge Graph Reasoning (ACM MM 2025).
- Paper: https://arxiv.org/abs/2507.20738
- Repo công bố: https://github.com/OreOZhao/DSoM
- Repo reproduction: https://github.com/keiranoluv/Reproduce_DSoM
- Nhánh reproduction: `experiment/wn18-table3`
- Nguồn Source-Frozen RC baseline: `experiment/source-frozen-rc`
- Commit baseline (FB15K-237 Source-Frozen RC): `1dfabf9b41a3a5ef700170f4227ca2c4c48a0e36`
- Nhánh reproduction: `experiment/wn18-table3`
- Tag reproduction: `table3-source-frozen-20261020`
- Seed: 2022
- GPU: NVIDIA RTX 5090 32GB
- Datasets: FB15K-237 và WN18, không WN18RR
- Student policy optimizer: `source_frozen`

## 2. So sánh student Table 3

Kết quả reproduction lấy từ `final_result` trong training `result.json`,
thuộc checkpoint có validation MRR tốt nhất.

| Dataset | Nguồn | MR ↓ | MRR ↑ | H@1 (%) | H@3 (%) | H@10 (%) |
|---|---|---:|---:|---:|---:|---:|
| FB15K-237 | Paper | 116.0000 | NR | 27.50 | 41.10 | 56.40 |
| FB15K-237 | Source-Frozen RC | 116.5569 | 0.373564 | 27.79 | 41.05 | 56.67 |
| WN18 | Paper | 7.0000 | NR | 95.00 | 96.20 | 97.40 |
| WN18 | Source-Frozen RC | 8.9588 | 0.955870 | 94.87 | 95.92 | 96.89 |

Paper Table 3 không báo MRR cho các dataset này. `NR` nghĩa là không công bố.

### Chênh lệch: reproduction trừ paper

| Dataset | Delta MR | Delta H@1 (pp) | Delta H@3 (pp) | Delta H@10 (pp) |
|---|---:|---:|---:|---:|
| FB15K-237 | +0.5569 | +0.29 | -0.05 | +0.27 |
| WN18 | +1.9588 | -0.13 | -0.28 | -0.51 |

- MR: thấp hơn tốt hơn.
- Hits@K: cao hơn tốt hơn.
- `pp`: điểm phần trăm.
- Paper báo MR dưới dạng số nguyên.

## 3. Cấu hình training

Áp dụng cùng cấu hình student baseline cho cả hai datasets.

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

### Tóm tắt training

| Dataset | Epoch tốt nhất | Total epochs | Best Valid MRR |
|---|---:|---:|---:|
| FB15K-237 | 60 | 71 | 0.379939 |
| WN18 | 21 | 32 | 0.953347 |

Cả hai thí nghiệm dừng bằng early stopping.

## 4. WN18 Teacher

WN18 teacher dùng ComplExMDR ba modalities, Adagrad, batch size 1000,
rank 2000, learning rate 0.1, N3 regularization và scheduler `min`.

- Epoch tốt nhất: 39
- Tổng epochs: 50
- Best Validation Hits@10: 97.49%
- Validation MRR at selected epoch: 0.959601
- Test MR: 7.4043
- Test MRR: 0.962368
- Test Hits@1: 95.48%
- Test Hits@3: 96.73%
- Test Hits@10: 97.62%

Teacher checkpoint được chọn bằng validation Hits@10.

Lưu ý: teacher evaluation công bố lấy minimum ground-truth rank giữa các
modalities. Đây là oracle metric, không phải score-fusion inference có thể
triển khai. Không so sánh trực tiếp với chất lượng student inference.

## 5. Paper so với implementation công bố

### Thuật toán theo paper

Paper mô tả RC policy học bằng REINFORCE để chọn động tổ hợp multimodal teachers hữu ích.

Student học từ multimodal teacher logits qua knowledge distillation, gồm neighbor-decoupled KD.

### Implementation công bố

Trong source công bố gốc, `Distiller.get_learnable_parameters()` chỉ trả student parameters.

`KD_RL` công bố không override phương thức này.

Vì vậy policy gradient loss có thể được tính nhưng RC policy parameters
không nằm trong optimizer parameter list tương ứng.

### Điều chỉnh trong reproduction

Fork hỗ trợ tường minh:

`--policy_optimizer source_frozen`

Mode này giữ sampling bảy strategies, reward calculation và REINFORCE loss,
nhưng chỉ cập nhật student parameters, không cập nhật policy network.

WN18 smoke diagnostics ghi nhận policy gradients khác không nhưng policy weight updates bằng zero.

Kết quả Source-Frozen RC gần Table 3 không chứng minh tác giả đã train frozen
policy, cũng không xác nhận hiệu quả learned RC.

Cần learned-policy ablations có kiểm soát để đánh giá đóng góp của reinforcement learning.

## 6. Giới hạn dữ liệu/evaluation

- Audit WN18 structure so với KBC/WN.
- WN18 có 40,943 entities, 18 base relations, mở rộng thành 36 với reciprocals.
- Độ phủ ảnh WN18: 7,801 / 40,943 entities (19.05%).
- Missing image features được impute ngẫu nhiên.
- Chưa xác nhận image/text features tương đương chính xác quy trình tác giả.
- WN18 có inverse-relation leakage, làm link prediction dễ hơn đáng kể.
- Cả hai student results dùng một seed, chưa có ước lượng phương sai giữa seeds.
- W&B `test/*` ở bước log cuối không nhất thiết là evaluation của best checkpoint.
- Báo cáo dùng `final_result`.

Nguồn dữ liệu WN18 được ghi trong [hồ sơ chuẩn bị WN18](../deprecated/setup/WN18_DATA_PREPARATION.md).

## 7. Artifacts thí nghiệm local

### FB15K-237

- Checkpoint teacher: `runs/teacher/FB15K-237/full/20261017_000249/checkpoints/0/m-20261017_0002-n-teacher_fb15k237_full_20261017_000249.pth`
- Checkpoint student: `runs/student/FB15K-237/table3_source_frozen_rc/full/20261020_121832/checkpoints/0/student-20261020_1218-n-table3_source_frozen_rc_full_20261020_121832.pth`
- Kết quả student: `runs/student/FB15K-237/table3_source_frozen_rc/full/20261020_121832/checkpoints/0/result.json`

### WN18

- Checkpoint teacher: `runs/teacher/WN18/full/scheduler-min/20261020_190522/checkpoints/0/m-20261020_1905-n-teacher_wn18_full_scheduler-min_20261020_190522.pth`
- Kết quả teacher: `runs/teacher/WN18/full/scheduler-min/20261020_190522/checkpoints/0/result.json`
- Checkpoint student: `runs/student/WN18/table3_source_frozen_rc/full/20261020_201526/checkpoints/0/student-20261020_2015-n-table3_source_frozen_rc_full_20261020_201526.pth`
- Kết quả student: `runs/student/WN18/table3_source_frozen_rc/full/20261020_201526/checkpoints/0/result.json`

Các đường dẫn này là training artifacts local, được chủ đích loại khỏi Git tracking.

## 8. Kết luận

Source-Frozen RC baseline cho kết quả student gần Table 3 đã công bố
trên cả FB15K-237 và WN18.

Điều này hỗ trợ khả năng tái lập hành vi released code, nhưng không độc lập
xác minh cơ chế học REINFORCE policy mà paper mô tả.

Các đối chứng tiếp theo được đề xuất:
- Structural student không KD.
- Fixed teacher-strategy KD.
- Learned-policy RC với matched training settings.
- Nhiều seeds và phân tích biến thiên thống kê.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
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
