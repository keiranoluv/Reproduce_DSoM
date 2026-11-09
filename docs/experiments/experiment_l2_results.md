# Thí nghiệm L2 — Chuẩn hóa RC policy input

## Thiết lập

Dataset: FB15K-237
Mẫu training: 2048
Số epochs: 1
Batch size: 16
Số optimizer steps: 128
Optimizer policy: Adam, LR 1e-5
Reward: original binary +1/-10
Temperature policy: 1

L0: raw teacher logits vào RC MLP
L2: teacher logits chuẩn hóa theo query vào RC MLP

Teacher logits dùng cho reward/KD giữ nguyên.

## Chẩn đoán training

| Chỉ số | L0 | L2 |
|---|---:|---:|
| Entropy ban đầu | 1.864328 | 1.928094 |
| Entropy cuối | 0.169272 | 0.401179 |
| Xác suất cực đại cuối | 0.944511 | 0.861394 |
| Logit range cuối | 17.169900 | 10.566500 |

## Audit validation đối chiếu 1024 queries

| Chỉ số | L0 | L2 |
|---|---:|---:|
| Policy entropy | 0.140282 | 0.424915 |
| Độ đồng thuận policy/oracle | 9.6680% | 10.9375% |
| Student CE | 7.139151 | 7.153788 |
| Policy CE | 5.539306 | 5.647519 |
| Oracle CE | 4.651416 | 4.651416 |
| Policy CE regret | 0.887890 | 0.996103 |
| Tỷ lệ reward dương | 75.3906% | 77.4414% |

## Phân phối strategies

| Chiến lược | L0 | L2 |
|---|---:|---:|
| Structural | 29.20% | 47.36% |
| Image | 0.00% | 0.00% |
| Description | 1.95% | 0.39% |
| Structural+Image | 0.00% | 0.00% |
| Structural+Description | 67.19% | 45.41% |
| Image+Description | 1.66% | 6.84% |
| All three | 0.00% | 0.00% |

## Phát hiện

1. Normalize input giảm early policy concentration.
2. L2 entropy cao hơn nhưng mean CE regret tệ hơn L0.
3. Cả hai ít chọn Description-only.
4. Policy đa dạng hơn không đảm bảo selection tốt hơn.

## Giới hạn

- Một run ngắn mỗi cấu hình.
- Chỉ 128 optimizer steps.
- Audit dùng 1024 sampled validation queries.
- Không so full-data link prediction.
- Reward design không đổi.

## Thí nghiệm tiếp theo

L3a: raw RC input + continuous CE reward.
L3b: normalized RC input + continuous CE reward.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="experiment-l2---normalized-rc-policy-input"></a>
<a id="findings"></a>
<a id="limitations"></a>
<a id="matched-1024-query-validation-audit"></a>
<a id="next-experiment"></a>
<a id="setup"></a>
<a id="strategy-distribution"></a>
<a id="training-diagnostics"></a>
