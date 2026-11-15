# K: description teacher và learned RC trên full data

Thí nghiệm K kiểm tra hai cách distill student FB15K-237: K1 lấy teacher description với NDKD; K2 dùng learned RC, NDKD và REINFORCE. Cả hai train năm epoch với cùng pretrained MIN teacher trên RTX 5090.

## Cấu hình

Dữ liệu có 544.230 training query sau reciprocal augmentation, batch 16. Student dùng Adagrad LR 0.1; K2 policy dùng Adam LR `1e-5`. Trong training, mỗi lần validation lấy 1.024 query mỗi chiều. Cả hai chọn checkpoint epoch 4, sau đó đánh giá filtered trên full validation/test.

## Kết quả

Validation đầy đủ:

| Chỉ số | K1 | K2 |
|---|---:|---:|
| MR | 115.219845 | 143.361359 |
| MRR | 0.359133 | 0.353590 |
| Hits@1 | 0.264015 | 0.259338 |
| Hits@3 | 0.394668 | 0.387111 |
| Hits@10 | 0.550157 | 0.544454 |

Test đầy đủ:

| Chỉ số | Paper DSoM | K1 | K2 |
|---|---:|---:|---:|
| MR | 116 | 119.167278 | 154.851830 |
| Hits@1 | 0.275 | 0.257305 | 0.251637 |
| Hits@3 | 0.411 | 0.389011 | 0.379507 |
| Hits@10 | 0.564 | 0.546663 | 0.537648 |

Test MRR của K1 là 0.353423, K2 là 0.345857. K1 tốt hơn ở mọi chỉ số validation/test đã đo. K1 mất 478.88 giây/epoch, khoảng 39.9 phút tổng cộng; K2 mất 603.36 giây/epoch, khoảng 50.3 phút. Đánh giá đầy đủ hai model/split mất khoảng 7 giây.

## Cách đọc kết quả

Đây là full-data pilot năm epoch, chọn checkpoint bằng sampled validation MRR. Hai cấu hình đổi cả cách chọn teacher lẫn policy optimization, nên chưa tách được tác động riêng của RC. Mỗi cấu hình có một seed và chưa khớp toàn bộ chỉ số Table 3.

Các lần kiểm tra tiếp theo nằm trong [L0/L1](l0-l1-policy-momentum.md), [L2](l2-normalized-input.md) và chuỗi L3-L9 ở [mục lục](README.md). Baseline Table 3 sau đó được ghi riêng trong [kết quả chính](../../research/tables/table-3-fb15k237-wn18.md).

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="experiment-k---full-data-student-ablation"></a>
<a id="findings"></a>
<a id="full-test"></a>
<a id="full-validation"></a>
<a id="methods"></a>
<a id="next-steps"></a>
<a id="runtime"></a>
<a id="setup"></a>
