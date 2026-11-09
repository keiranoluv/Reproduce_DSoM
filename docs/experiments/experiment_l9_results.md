# Thí nghiệm L9 — Student fixed Structural+Description

## Mục tiêu

Đánh giá fixed Structural+Description có cải thiện distillation so Experiment K baselines không.

## Dataset

FB15K-237

## Teacher

Pretrained MIN checkpoint đóng băng:

runs/teacher/FB15K-237/full/20261017_000249/checkpoints/0/m-20261017_0002-n-teacher_fb15k237_full_20261017_000249.pth

## Cấu hình

- Teacher strategy: Structural+Description (0.5, 0, 0.5)
- Student: unimodal ComplExMDR
- Distillation: fixed_ndkd
- Mẫu training: 544230 reciprocal queries
- Số epochs: 5
- Batch size: 16
- Rank: 2000
- Optimizer: Adagrad
- Learning rate: 0.1
- Regularizer: N3
- Trọng số regularization: 0.01
- CE weight: 1
- KD weight: 2
- Temperature: 4
- Scheduler: ReduceLROnPlateau, mode=min

## Đánh giá filtered đầy đủ

| Split | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|---:|
| Validation | 112.2216 | 0.362118 | 0.264271 | 0.399829 | 0.558968 |
| Test | 117.0304 | 0.357728 | 0.260945 | 0.393873 | 0.554481 |

## So sánh với K trên test

| Student | MR | MRR | Hits@1 | Hits@10 |
|---|---:|---:|---:|---:|
| K1 - Description | 119.17 | 0.35342 | 0.25731 | 0.54666 |
| K2 - Learned RC | 154.85 | 0.34586 | 0.25164 | 0.53765 |
| L9 - Structural+Description | 117.03 | 0.357728 | 0.260945 | 0.554481 |

## So sánh Table 3 trên test

| Chỉ số | Paper | L9 | Chênh lệch |
|---|---:|---:|---:|
| MR | 116 | 117.0304 | +1.0304 |
| Hits@1 | 0.275 | 0.260945 | -0.014055 |
| Hits@3 | 0.411 | 0.393873 | -0.017127 |
| Hits@10 | 0.564 | 0.554481 | -0.009519 |

## Phát hiện

1. Trong các runs đã ghi nhận, L9 hơn K1/K2 về test MRR, Hits@1 và Hits@10.

2. Fixed Structural+Description cho student tốt hơn learned-RC config được đánh giá.

3. L9 gần Table 3 nhưng không phải tái lập trung thực DSoM RC.

## Giới hạn

- Mỗi cấu hình so sánh một run.
- Không có confidence intervals nhiều seeds.
- Chưa xác minh năm training epochs khớp đầy đủ paper protocol.
- Chiến lược cố định, không học.
- Teacher checkpoint/modalities còn cần xác minh protocol.

## Kết luận

Kết thúc nhánh diagnostics/ablations L1-L9.

Tiếp theo trở lại reproduction Table 3, tập trung căn chỉnh code và paper.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="comparison-with-dsom-table-3---test"></a>
<a id="comparison-with-experiment-k---test"></a>
<a id="conclusion"></a>
<a id="configuration"></a>
<a id="experiment-l9---fixed-structuraldescription-student"></a>
<a id="findings"></a>
<a id="full-filtered-evaluation"></a>
<a id="limitations"></a>
<a id="objective"></a>
