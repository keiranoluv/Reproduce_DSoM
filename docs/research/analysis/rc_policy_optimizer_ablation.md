# Ablation RC policy optimizer — FB15K-237

## Mục tiêu

Khảo sát RC policy saturation khi train DSoM student.

## Thiết lập

- Teacher: FB15K-237 pretrained MIN checkpoint
- Optimizer student: Adagrad, LR=0.1
- Mẫu training: 2048
- Batch size: 16
- Số epochs: 1
- Queries evaluation: 64
- Policy network: 43623 -> 1024 -> 7
- KD temperature: 4
- RC reward: +1 / -10

## Probe trên batch cố định (batch 3)

| Chỉ số | A: Adagrad 1e-3 | B: Adam 1e-4 | C: Adam 1e-3 | D: Adam 1e-5 |
|---|---:|---:|---:|---:|
| Entropy sau update | 0.000378 | 0.845803 | 0.010450 | 1.745250 |
| Hidden RMS sau update | 21.264126 | 1.627662 | 12.482269 | 0.842773 |
| Logit range sau update | 213.857391 | 13.630247 | 120.607147 | 2.191428 |
| Batches có policy gradient khác không | 22 | 34 | 19 | 31 |
| Batches cập nhật fc2 weights | 16 | 125 | 125 | 125 |

Cả bốn thí nghiệm có cùng pre-update probe:
- Entropy: 1.857928
- Hidden RMS: 0.829591
- Logit range: 1.246729

## Kết quả student (smoke tests)

| Chỉ số | B: Adam 1e-4 | C: Adam 1e-3 | D: Adam 1e-5 |
|---|---:|---:|---:|
| Valid MRR | 0.1572 | 0.1765 | 0.1617 |
| Valid Hits@10 | 0.2344 | 0.2500 | 0.2578 |
| Test MRR | 0.1138 | 0.1219 | 0.1223 |
| Test Hits@10 | 0.1641 | 0.1797 | 0.1797 |

## Phát hiện

1. Adagrad policy LR=1e-3 saturation nặng ngay sau update đáng kể đầu tiên.
2. Adam LR=1e-3 cũng saturation nhanh.
3. Adam LR=1e-4 giảm initial activation explosion nhưng về sau policy gần deterministic.
4. Adam LR=1e-5 ổn định ban đầu nhất trong các cấu hình được thử.
5. Entropy cao hơn không đảm bảo student MRR tốt hơn.

## Giới hạn

- Chỉ 2048 training examples và một epoch.
- Evaluation giới hạn 64 queries mỗi chiều.
- Không so trực tiếp với Table 3.
- Policy optimizer experiments sửa reproduction implementation, không khẳng định khớp paper.
- Cần training dài hơn để đánh giá ổn định lâu dài.

## Thí nghiệm tiếp theo

Adam policy LR=1e-5, 20000 training examples, 3 epochs, mở rộng validation evaluation.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="findings"></a>
<a id="fixed-batch-probe-batch-3"></a>
<a id="limitations"></a>
<a id="next-experiment"></a>
<a id="objective"></a>
<a id="rc-policy-optimizer-ablation--fb15k-237"></a>
<a id="setup"></a>
<a id="student-results-smoke-tests"></a>
