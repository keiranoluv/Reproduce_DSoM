# Chẩn đoán RC policy DSoM — FB15K-237

## Phạm vi

Các thí nghiệm khảo sát RC policy khi distill student; đây là diagnostics/
ablations, không tuyên bố tái lập mọi chi tiết của paper.

Nhánh baseline: student-pipeline-fixes

Nhánh thí nghiệm: experiment/rc-policy-stabilization

Teacher: FB15K-237 pretrained MIN checkpoint.

## Thí nghiệm E — Training quy mô vừa

Cấu hình:

- Optimizer student: Adagrad, learning rate 0.1
- Optimizer policy: Adam, learning rate 1e-5
- Mẫu training: 20,000
- Batch size: 16
- Số epochs: 3
- Queries evaluation: 1,024

Kết quả:

| Epoch | Valid MRR | Test MRR | Test Hits@10 |
|---|---:|---:|---:|
| 1 | 0.238054 | 0.226723 | 0.367188 |
| 2 | 0.225990 | 0.241456 | 0.380859 |
| 3 | 0.257907 | 0.263478 | 0.409668 |

Policy entropy ở minibatch cuối:

| Epoch | Entropy | Near-deterministic |
|---|---:|---:|
| 1 | 0.081332 | 13/16 |
| 2 | 0.000929 | 15/16 |
| 3 | 0.000003 | 16/16 |

Diễn giải: student tốt hơn nhưng policy gần deterministic.

## Thí nghiệm F — Audit chất lượng strategy

Checkpoint: best distiller Experiment E.

Validation: 1,024 queries gồm reciprocals. Policy chọn argmax.

- Policy chọn Structural-only: 100%
- Độ đồng thuận policy/oracle: 2.6367%
- Student mean CE: 6.519696
- Policy mean CE: 6.272866
- Oracle mean CE: 4.651416
- Policy mean CE regret: 1.621450

Phân phối oracle strategies:

| Chiến lược | Frequency |
|---|---:|
| Structural | 2.64% |
| Image | 5.47% |
| Description | 50.78% |
| Structural+Image | 3.71% |
| Structural+Description | 12.11% |
| Image+Description | 20.61% |
| All three | 4.69% |

Oracle cần ground-truth labels, chỉ dùng diagnostic.

## Thí nghiệm G — Audit reward signal

Cùng checkpoint và validation setup của Experiment F.

Reward +1 khi strategy CE < student CE, ngược lại -10.
Baseline reward từ mean của ba teacher modalities.

| Chiến lược | Mean CE | Positive reward |
|---|---:|---:|
| Structural | 6.2729 | 55.76% |
| Image | 6.3664 | 55.47% |
| Description | 4.9021 | 88.87% |
| Structural+Image | 5.6744 | 74.41% |
| Structural+Description | 5.2009 | 85.64% |
| Image+Description | 5.2135 | 85.16% |
| All three | 5.2242 | 86.52% |

Phát hiện bổ sung:

- Mọi strategies cùng reward: 43.55% queries
- Ít nhất hai strategies nhận +1: 91.89%
- Mean CE gap giữa các strategies reward dương: 1.411396
- Policy reward tệ hơn best available reward: 38.09%

Quan sát gợi ý reward phân biệt hạn chế; chưa đủ xác lập nguyên nhân collapse.

## Thí nghiệm H — Audit hướng update

Student: Adagrad 0.1.
Policy: Adam 1e-5.
Training: 2,048 examples, một epoch.

Batch 3:
- Một advantage âm
- P(Description) thay đổi: -0.091313
- P(Structural) thay đổi: +0.002867

Batch 8:
- Hai advantages âm
- P(Image) thay đổi: -0.108906
- P(Structural) thay đổi: +0.039095

So probabilities trên cùng minibatch trước/sau optimizer updates, tắt policy dropout.

## Thí nghiệm I — Phân tích advantage

20 batches đầu, 320 examples:

- Samples advantage dương: 0
- Samples advantage âm: 7
- Samples advantage zero: 313

Actions có advantage âm:

- Image: 5
- Description: 1
- Image+Description: 1

Batch 3: Description nhận advantage -11.
Batch 8: hai Image selections nhận advantage -11 mỗi selection.

Phù hợp REINFORCE phạt sampled strategies; chưa chứng minh early updates
đơn lẻ gây ra Structural-only policy cuối cùng.

## Giới hạn

- E chỉ dùng 20,000 examples, 3 epochs.
- Evaluation/audits dùng sampled queries.
- F/G đánh giá trained policy trên validation.
- CE oracle cần nhãn.
- H/I chỉ đo 20 batches đầu.
- Chưa thể kết luận full Table 3 reproduction từ những kiểm tra này.

## Tiếp theo: thí nghiệm J

So distillation với các fixed strategies:

- J0: Structural-only
- J1: Description-only
- J2: trung bình mọi modalities
- J3: learned RC (Adam 1e-5)

Giữ student initialization, training subset, optimizer, NDKD config, epochs
và evaluation protocol giống nhau trong khả năng có thể.

Dùng validation metrics để so model, không chọn cấu hình bằng test metrics.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="dsom-rc-policy-diagnostics--fb15k-237"></a>
<a id="experiment-e--medium-training"></a>
<a id="experiment-f--strategy-quality-audit"></a>
<a id="experiment-g--reward-signal-audit"></a>
<a id="experiment-h--update-direction-audit"></a>
<a id="experiment-i--advantage-attribution"></a>
<a id="limitations"></a>
<a id="next-experiment-j"></a>
<a id="scope"></a>
