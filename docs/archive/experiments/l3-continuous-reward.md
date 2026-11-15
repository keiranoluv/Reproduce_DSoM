# L3: đổi binary reward sang continuous CE reward

L3 kiểm tra reward liên tục trên FB15K-237 với 2.048 training example, batch 16, một epoch/128 step và policy Adam LR `1e-5`.

L0/L2 dùng binary reward với raw/normalized input. L3a/L3b giữ hai kiểu input đó và dùng `R(s) = CE(student) - CE(teacher_s)`. Khi trừ baseline, advantage là `CE(teacher_average) - CE(teacher_s)`.

## Trong training

| Chỉ số | L0 | L2 | L3a | L3b |
|---|---:|---:|---:|---:|
| Entropy cuối | 0.169272 | 0.401179 | 0.056885 | 0.227821 |
| Xác suất cực đại cuối | 0.944511 | 0.861394 | 0.981713 | 0.911043 |
| Logit range cuối | 17.1699 | 10.5665 | 21.9802 | 12.7707 |
| Advantage khác không | 1.81% | 1.42% | 99.76% | 99.12% |

Continuous reward gần như loại bỏ advantage bằng zero, nhưng L3a/L3b tập trung mạnh hơn vào structural.

## Trên validation

Cùng 1.024 sampled query:

| Chỉ số | L0 | L2 | L3a | L3b |
|---|---:|---:|---:|---:|
| Policy entropy | 0.140282 | 0.424915 | 0.022224 | 0.109288 |
| Độ đồng thuận oracle | 9.6680% | 10.9375% | 2.6367% | 2.7344% |
| Policy CE | 5.539306 | 5.647519 | 6.233175 | 6.223771 |
| CE regret | 0.887890 | 0.996103 | 1.581759 | 1.572355 |
| Tỷ lệ chọn Structural | 29.20% | 47.36% | 96.97% | 95.31% |

CE regret của hai cấu hình continuous reward cao hơn binary baseline. Normalize input giúp entropy cao hơn ở cả hai cách reward.

## Policy gradient

Norm được đo trên `RC MLP fc2.weight`:

| Chỉ số | L0 | L2 | L3a | L3b |
|---|---:|---:|---:|---:|
| Gradient trung bình | 2.8690 | 2.0686 | 1.1808 | 1.1928 |
| Gradient trung vị | 0.0000 | 0.0000 | 0.2814 | 0.6525 |
| Gradient cực đại | 38.6851 | 24.4666 | 18.8057 | 9.5310 |
| Batches gradient khác không | 24.22% | 19.53% | 100% | 100% |

Continuous reward có gradient khác không ở mọi batch, nhưng mean norm thấp hơn binary reward. Lần thử này chưa tách riêng reward scale và policy learning rate, mới có một seed và chưa đánh giá full-data link prediction.

[L4](l4-policy-learning-rate.md) giảm policy LR từ `1e-5` xuống `1e-6` để kiểm tra tiếp.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="experiment-l3---continuous-ce-reward-ablation"></a>
<a id="findings"></a>
<a id="limitations"></a>
<a id="matched-validation-audit-1024-queries"></a>
<a id="next-experiment"></a>
<a id="policy-gradient-diagnostics"></a>
<a id="setup"></a>
<a id="training-diagnostics"></a>
