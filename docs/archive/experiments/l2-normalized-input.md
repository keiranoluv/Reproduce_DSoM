# L2: chuẩn hóa đầu vào policy

L2 so raw teacher logits của L0 với logits đã chuẩn hóa theo query trước khi đưa vào RC MLP. Logits dùng để tính reward và KD giữ nguyên.

Hai cấu hình dùng FB15K-237, 2.048 training example, batch 16, một epoch/128 step. Policy là Adam LR `1e-5`, temperature 1, binary reward +1/-10.

## Trong training

| Chỉ số | L0 | L2 |
|---|---:|---:|
| Entropy ban đầu | 1.864328 | 1.928094 |
| Entropy cuối | 0.169272 | 0.401179 |
| Xác suất cực đại cuối | 0.944511 | 0.861394 |
| Logit range cuối | 17.169900 | 10.566500 |

Chuẩn hóa input giúp entropy cuối cao hơn và khoảng logits nhỏ hơn trong lần chạy này.

## Trên validation

Đối chiếu cùng 1.024 sampled validation query:

| Chỉ số | L0 | L2 |
|---|---:|---:|
| Policy entropy | 0.140282 | 0.424915 |
| Độ đồng thuận policy/oracle | 9.6680% | 10.9375% |
| Student CE | 7.139151 | 7.153788 |
| Policy CE | 5.539306 | 5.647519 |
| Oracle CE | 4.651416 | 4.651416 |
| Policy CE regret | 0.887890 | 0.996103 |
| Tỷ lệ reward dương | 75.3906% | 77.4414% |

| Chiến lược | L0 | L2 |
|---|---:|---:|
| Structural | 29.20% | 47.36% |
| Image | 0.00% | 0.00% |
| Description | 1.95% | 0.39% |
| Structural+Image | 0.00% | 0.00% |
| Structural+Description | 67.19% | 45.41% |
| Image+Description | 1.66% | 6.84% |
| All three | 0.00% | 0.00% |

L2 chọn strategy đa dạng hơn nhưng CE regret cao hơn L0. Cả hai ít chọn description-only. Entropy cao hơn chưa đồng nghĩa với chọn teacher có CE tốt hơn.

Đây là một lần chạy ngắn cho mỗi cấu hình, chưa đánh giá full-data link prediction. [L3](l3-continuous-reward.md) tiếp tục thử continuous CE reward với cả raw và normalized input.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="experiment-l2---normalized-rc-policy-input"></a>
<a id="findings"></a>
<a id="limitations"></a>
<a id="matched-1024-query-validation-audit"></a>
<a id="next-experiment"></a>
<a id="setup"></a>
<a id="strategy-distribution"></a>
<a id="training-diagnostics"></a>
