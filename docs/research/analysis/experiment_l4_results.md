# Thí nghiệm L4 — Ablation policy learning rate

## Mục tiêu

Kiểm tra giảm RC policy learning rate có giảm early concentration và cải
thiện teacher strategy selection dưới continuous CE reward hay không.

## Thiết lập thí nghiệm

Dataset: FB15K-237
Mẫu training: 2048
Số epochs: 1
Batch size: 16
Số optimizer steps: 128

Reward: Continuous CE
Optimizer policy: Adam
Optimizer student: Adagrad, LR 0.1

Các cấu hình:
- L3a: raw input, policy LR 1e-5
- L4a: raw input, policy LR 1e-6
- L3b: normalized input, policy LR 1e-5
- L4b: normalized input, policy LR 1e-6

## Chẩn đoán training

| Chỉ số | L3a | L4a | L3b | L4b |
|---|---:|---:|---:|---:|
| Entropy cuối | 0.056885 | 0.401380 | 0.227821 | 0.650564 |
| Xác suất cực đại cuối | 0.981713 | 0.852131 | 0.911043 | 0.785581 |
| Logit range cuối | 21.9802 | 7.0258 | 12.7707 | 5.2009 |
| Gradient trung bình norm | 1.1808 | 4.3228 | 1.1928 | 3.5271 |
| Advantage khác không | 99.76% | 97.46% | 99.12% | 95.65% |

Gradient norm là của RC MLP fc2.weight.

## Audit validation đối chiếu

1024 validation queries, policy argmax, seed 2022.

| Chỉ số | L3a | L4a | L3b | L4b |
|---|---:|---:|---:|---:|
| Policy entropy | 0.022224 | 0.284409 | 0.109288 | 0.569393 |
| Độ đồng thuận oracle | 2.6367% | 3.8086% | 2.7344% | 4.1992% |
| Policy CE | 6.233175 | 6.206656 | 6.223771 | 6.159985 |
| CE regret | 1.581759 | 1.555240 | 1.572355 | 1.508569 |
| Tỷ lệ chọn Structural | 96.97% | 95.90% | 95.31% | 92.38% |

## Phát hiện

1. Giảm LR từ 1e-5 xuống 1e-6 tăng entropy ở cả hai input configs.
2. Logit range policy giảm mạnh.
3. Độ đồng thuận oracle tăng nhẹ.
4. CE regret giảm nhẹ.
5. Structural vẫn chiếm argmax, hơn 92% validation queries.
6. Chỉ giảm policy LR chưa giải quyết selection mismatch.

## Giới hạn

- Một run ngắn mỗi cấu hình.
- Chỉ 128 optimizer steps.
- Evaluation dùng 1024 sampled validation queries.
- Chưa xác lập nguyên nhân policy collapse.
- Chưa có full-data link prediction cho L4.

## Thí nghiệm tiếp theo

L5: audit hướng policy gradient.

Kiểm tra:
- Advantage distribution theo strategy được chọn.
- Policy probability thay đổi sau optimizer step.
- Teacher CE từng strategy trên training minibatches.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="experiment-l4---policy-learning-rate-ablation"></a>
<a id="experimental-setup"></a>
<a id="findings"></a>
<a id="limitations"></a>
<a id="matched-validation-audit"></a>
<a id="next-experiment"></a>
<a id="objective"></a>
<a id="training-diagnostics"></a>
