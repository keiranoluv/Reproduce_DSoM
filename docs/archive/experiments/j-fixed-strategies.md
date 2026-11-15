# J: teacher cố định và learned RC

Thí nghiệm J so bốn cách chọn teacher khi distill student FB15K-237: structural, description, average và learned RC. Mỗi cấu hình dùng 20.000 training example, batch 16, ba epoch; validation/test lấy 1.024 query mỗi split.

Student dùng Adagrad LR 0.1, policy RC dùng Adam LR `1e-5`. Checkpoint được chọn theo validation MRR.

| Run | Valid MRR | Valid H@10 | Test MRR | Test H@10 |
|---|---:|---:|---:|---:|
| J0 Structural | 0.250256 | 0.399414 | 0.250696 | 0.393066 |
| J1 Description | 0.264426 | 0.408203 | 0.261604 | 0.416016 |
| J2 Average | 0.249199 | 0.387207 | 0.249323 | 0.391113 |
| J3 Learned RC | 0.257907 | 0.407715 | 0.263478 | 0.409668 |

Description có validation MRR cao nhất. Learned RC tốt hơn fixed structural trên validation, dù phép kiểm tra policy cuối trước đó cho thấy nó tập trung vào structural. Lựa chọn ở checkpoint cuối có thể chưa mô tả hết quá trình teacher selection khi train.

Các kết quả dùng tập nhỏ và sampled evaluation, cần kiểm tra thêm initialization/minibatch order trước khi quy chênh lệch cho strategy. [Thí nghiệm K](k-full-data.md) mở rộng description và learned RC sang full data.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="experiment-j--fixed-strategy-vs-learned-rc"></a>
<a id="findings"></a>
<a id="limitations"></a>
