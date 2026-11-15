# Thử optimizer cho RC policy trên FB15K-237

Các lần thử này kiểm tra policy tập trung nhanh tới mức nào sau optimizer update. Student dùng pretrained MIN teacher, Adagrad LR 0.1; policy có mạng `43623 → 1024 → 7`, KD temperature 4, reward +1/-10.

Mỗi cấu hình chạy 2.048 training example, batch 16, một epoch; evaluation giới hạn 64 query mỗi chiều.

## Probe cùng batch

Trước update ở batch 3, cả bốn cấu hình có entropy 1.857928, hidden RMS 0.829591 và logit range 1.246729.

| Chỉ số | A: Adagrad 1e-3 | B: Adam 1e-4 | C: Adam 1e-3 | D: Adam 1e-5 |
|---|---:|---:|---:|---:|
| Entropy sau update | 0.000378 | 0.845803 | 0.010450 | 1.745250 |
| Hidden RMS sau update | 21.264126 | 1.627662 | 12.482269 | 0.842773 |
| Logit range sau update | 213.857391 | 13.630247 | 120.607147 | 2.191428 |
| Batches có policy gradient khác không | 22 | 34 | 19 | 31 |
| Batches cập nhật fc2 weights | 16 | 125 | 125 | 125 |

Adagrad `1e-3` và Adam `1e-3` làm policy tập trung nhanh. Adam `1e-4` giảm mức tăng activation ban đầu nhưng policy về sau vẫn gần deterministic. Adam `1e-5` giữ ổn định ban đầu tốt nhất trong các cấu hình đã thử.

## Kết quả student ở smoke

| Chỉ số | B: Adam 1e-4 | C: Adam 1e-3 | D: Adam 1e-5 |
|---|---:|---:|---:|
| Valid MRR | 0.1572 | 0.1765 | 0.1617 |
| Valid Hits@10 | 0.2344 | 0.2500 | 0.2578 |
| Test MRR | 0.1138 | 0.1219 | 0.1223 |
| Test Hits@10 | 0.1641 | 0.1797 | 0.1797 |

Entropy cao hơn chưa bảo đảm student MRR tốt hơn. Các số này thuộc smoke với ít dữ liệu, chưa dùng để so Table 3. Đổi optimizer policy là một biến thể thực nghiệm, chưa được xác nhận là thiết lập của paper.

[Thí nghiệm E–I](e-i-policy-diagnostics.md) tiếp tục với Adam `1e-5`, 20.000 training example và ba epoch.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="findings"></a>
<a id="fixed-batch-probe-batch-3"></a>
<a id="limitations"></a>
<a id="next-experiment"></a>
<a id="objective"></a>
<a id="rc-policy-optimizer-ablation--fb15k-237"></a>
<a id="setup"></a>
<a id="student-results-smoke-tests"></a>
