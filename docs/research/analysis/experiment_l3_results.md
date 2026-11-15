# Thí nghiệm L3 — Ablation continuous CE reward

## Thiết lập

Dataset: FB15K-237
Mẫu training: 2048
Số epochs: 1
Batch size: 16
Số training steps: 128
Optimizer policy: Adam, LR 1e-5

Các cấu hình:
- L0: binary reward, raw RC input
- L2: binary reward, normalized RC input
- L3a: continuous CE reward, raw RC input
- L3b: continuous CE reward, normalized RC input

Continuous reward:
R(s) = CE(Student) - CE(Teacher_s)

Advantage:
A(s) = CE(Teacher_average) - CE(Teacher_s)

## Chẩn đoán training

| Chỉ số | L0 | L2 | L3a | L3b |
|---|---:|---:|---:|---:|
| Entropy cuối | 0.169272 | 0.401179 | 0.056885 | 0.227821 |
| Xác suất cực đại cuối | 0.944511 | 0.861394 | 0.981713 | 0.911043 |
| Logit range cuối | 17.1699 | 10.5665 | 21.9802 | 12.7707 |
| Advantage khác không | 1.81% | 1.42% | 99.76% | 99.12% |

## Audit validation đối chiếu (1024 queries)

| Chỉ số | L0 | L2 | L3a | L3b |
|---|---:|---:|---:|---:|
| Policy entropy | 0.140282 | 0.424915 | 0.022224 | 0.109288 |
| Độ đồng thuận oracle | 9.6680% | 10.9375% | 2.6367% | 2.7344% |
| Policy CE | 5.539306 | 5.647519 | 6.233175 | 6.223771 |
| CE regret | 0.887890 | 0.996103 | 1.581759 | 1.572355 |
| Tỷ lệ chọn Structural | 29.20% | 47.36% | 96.97% | 95.31% |

## Chẩn đoán policy gradient

Đo gradient norm trên RC MLP fc2.weight.

| Chỉ số | L0 | L2 | L3a | L3b |
|---|---:|---:|---:|---:|
| Gradient trung bình | 2.8690 | 2.0686 | 1.1808 | 1.1928 |
| Gradient trung vị | 0.0000 | 0.0000 | 0.2814 | 0.6525 |
| Gradient cực đại | 38.6851 | 24.4666 | 18.8057 | 9.5310 |
| Batches gradient khác không | 24.22% | 19.53% | 100% | 100% |

## Phát hiện

1. Continuous CE reward gần như loại bỏ advantage sparsity.
2. L3a/L3b tập trung mạnh vào Structural.
3. Hai L3 variants có CE regret cao hơn binary baselines.
4. Continuous reward có mean fc2 gradient norm thấp hơn.
5. Normalize input giảm concentration ở cả hai reward settings.

## Giới hạn

- Một run mỗi cấu hình.
- Chỉ 128 optimizer steps.
- Evaluation dùng 1024 sampled validation queries.
- Chưa kiểm soát độc lập reward scale và policy learning rate.
- Chưa xác lập nguyên nhân policy collapse.
- Chưa có full-data link prediction cho L3.

## Thí nghiệm tiếp theo

L4: ablation policy learning rate với continuous CE reward.
So policy_lr=1e-5 với 1e-6.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="experiment-l3---continuous-ce-reward-ablation"></a>
<a id="findings"></a>
<a id="limitations"></a>
<a id="matched-validation-audit-1024-queries"></a>
<a id="next-experiment"></a>
<a id="policy-gradient-diagnostics"></a>
<a id="setup"></a>
<a id="training-diagnostics"></a>
