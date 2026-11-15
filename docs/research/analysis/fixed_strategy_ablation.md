# Thí nghiệm J — Fixed strategy so với learned RC

Dataset: FB15K-237

Cấu hình:
- 20,000 training examples
- 3 epochs
- Batch size 16
- Student: Adagrad, LR 0.1
- RC policy: Adam, LR 1e-5
- Đánh giá: 1,024 queries mỗi split
- Chọn checkpoint bằng validation MRR

| Run | Valid MRR | Valid H@10 | Test MRR | Test H@10 |
|---|---:|---:|---:|---:|
| J0 Structural | 0.250256 | 0.399414 | 0.250696 | 0.393066 |
| J1 Description | 0.264426 | 0.408203 | 0.261604 | 0.416016 |
| J2 Average | 0.249199 | 0.387207 | 0.249323 | 0.391113 |
| J3 Learned RC | 0.257907 | 0.407715 | 0.263478 | 0.409668 |

## Phát hiện

Description-only có validation MRR cao nhất.

Learned RC hơn Structural-only trên validation dù diagnostics trước đó cho thấy policy collapse về Structural-only.

Điều này gợi ý policy decisions cuối chưa mô tả hết tác động teacher selection động trong training.

## Giới hạn

Kết quả từ medium run: 20,000 examples, 3 epochs và sampled evaluation queries.

Cần kiểm tra tương đương initialization/minibatch order.

Tại thời điểm thí nghiệm chưa thể khẳng định full Table 3 reproduction.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="experiment-j--fixed-strategy-vs-learned-rc"></a>
<a id="findings"></a>
<a id="limitations"></a>
