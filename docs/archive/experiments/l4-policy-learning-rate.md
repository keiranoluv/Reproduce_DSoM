# L4: giảm learning rate policy

L4 giữ continuous CE reward của L3 và giảm policy Adam LR từ `1e-5` xuống `1e-6`. L3a/L4a dùng raw input; L3b/L4b dùng normalized input.

Các cấu hình chạy trên FB15K-237 với 2.048 training example, batch 16 và một epoch/128 step. Student dùng Adagrad LR 0.1.

## Trong training

| Chỉ số | L3a | L4a | L3b | L4b |
|---|---:|---:|---:|---:|
| Entropy cuối | 0.056885 | 0.401380 | 0.227821 | 0.650564 |
| Xác suất cực đại cuối | 0.981713 | 0.852131 | 0.911043 | 0.785581 |
| Logit range cuối | 21.9802 | 7.0258 | 12.7707 | 5.2009 |
| Gradient trung bình norm | 1.1808 | 4.3228 | 1.1928 | 3.5271 |
| Advantage khác không | 99.76% | 97.46% | 99.12% | 95.65% |

Gradient norm được đo ở `RC MLP fc2.weight`. LR nhỏ hơn cho entropy cao hơn và khoảng logits thấp hơn ở cả hai kiểu input.

## Trên validation

Đánh giá policy argmax trên cùng 1.024 query, seed 2022:

| Chỉ số | L3a | L4a | L3b | L4b |
|---|---:|---:|---:|---:|
| Policy entropy | 0.022224 | 0.284409 | 0.109288 | 0.569393 |
| Độ đồng thuận oracle | 2.6367% | 3.8086% | 2.7344% | 4.1992% |
| Policy CE | 6.233175 | 6.206656 | 6.223771 | 6.159985 |
| CE regret | 1.581759 | 1.555240 | 1.572355 | 1.508569 |
| Tỷ lệ chọn Structural | 96.97% | 95.90% | 95.31% | 92.38% |

Oracle agreement tăng nhẹ và CE regret giảm nhẹ, nhưng structural vẫn chiếm hơn 92% lựa chọn. Giảm LR chưa giải quyết được chênh lệch giữa lựa chọn của policy và CE oracle.

Mỗi cấu hình chỉ chạy 128 step, chưa có full-data link prediction. [L5](l5-strategy-advantage.md) kiểm tra advantage từng strategy ở các training batch đầu để hiểu lựa chọn này.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="experiment-l4---policy-learning-rate-ablation"></a>
<a id="experimental-setup"></a>
<a id="findings"></a>
<a id="limitations"></a>
<a id="matched-validation-audit"></a>
<a id="next-experiment"></a>
<a id="objective"></a>
<a id="training-diagnostics"></a>
