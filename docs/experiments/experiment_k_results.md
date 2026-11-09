# Thí nghiệm K — Ablation student trên full data

## Thiết lập

- Dataset: FB15K-237
- GPU: NVIDIA RTX 5090
- Teacher: pretrained MIN
- Mẫu training: 544,230, gồm reciprocal triples
- Batch size: 16
- Số epochs: 5
- Best checkpoint: epoch 4 cho cả hai runs
- Optimizer student: Adagrad, LR 0.1
- K2 policy optimizer: Adam, LR 1e-5
- Validation trong training: 1,024 queries mỗi chiều
- Đánh giá cuối: full filtered validation/test

## Phương pháp

- K1: teacher chỉ dùng description với NDKD
- K2: learned RC với NDKD và REINFORCE

## Validation đầy đủ

| Chỉ số | K1 | K2 |
|---|---:|---:|
| MR | 115.219845 | 143.361359 |
| MRR | 0.359133 | 0.353590 |
| Hits@1 | 0.264015 | 0.259338 |
| Hits@3 | 0.394668 | 0.387111 |
| Hits@10 | 0.550157 | 0.544454 |

## Test đầy đủ

| Chỉ số | Paper DSoM | K1 | K2 |
|---|---:|---:|---:|
| MR | 116 | 119.167278 | 154.851830 |
| Hits@1 | 0.275 | 0.257305 | 0.251637 |
| Hits@3 | 0.411 | 0.389011 | 0.379507 |
| Hits@10 | 0.564 | 0.546663 | 0.537648 |

Test MRR:
- K1: 0.353423
- K2: 0.345857

## Thời gian chạy

- K1: 478.88 giây/epoch (~39.9 phút tổng cộng)
- K2: 603.36 giây/epoch (~50.3 phút tổng cộng)
- Full evaluation: ~7 giây cho cả hai models/splits

## Phát hiện

1. K1 tốt hơn K2 trên mọi full validation/test metrics.
2. K1 train nhanh hơn K2 khoảng 26%.
3. Không run nào khớp toàn bộ published Table 3 metrics.
4. Cả hai dùng một seed và năm epochs.
5. Chọn checkpoint bằng sampled validation MRR.
6. Khác teacher selection và RL optimization, không cô lập tác động nhân quả của RC collapse.

## Các bước tiếp theo

1. Audit K2 strategy distribution và policy entropy.
2. So selected strategy CE với oracle CE.
3. Benchmark batch sizes lớn hơn.
4. Xác minh training/evaluation protocol paper.
5. Chạy thêm seeds trước kết luận tổng quát.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="experiment-k---full-data-student-ablation"></a>
<a id="findings"></a>
<a id="full-test"></a>
<a id="full-validation"></a>
<a id="methods"></a>
<a id="next-steps"></a>
<a id="runtime"></a>
<a id="setup"></a>
