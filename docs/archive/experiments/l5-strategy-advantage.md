# L5: vì sao policy ưu tiên structural ở đầu training?

L5 lấy cấu hình L3a: continuous CE reward, raw policy input, FB15K-237, pretrained MIN teacher đóng băng. Lần chạy có 2.048 training example, batch 16 và một epoch/128 step; phép kiểm tra tập trung vào 20 batch đầu, gồm 320 query.

Với mỗi query, tính cả bảy strategy. `Advantage(s) = CE(average teacher) - CE(teacher_s)`; giá trị dương nghĩa là strategy có CE tốt hơn average teacher.

## Kết quả trên training query

| Chiến lược | Mean CE | Mean Advantage | Oracle % | Sampled % |
|---|---:|---:|---:|---:|
| Structural | 2.7072 | +1.1887 | 79.38% | 80.94% |
| Image | 6.4623 | -2.5664 | 0.94% | 1.56% |
| Description | 4.7788 | -0.8829 | 0.94% | 4.69% |
| Structural+Image | 3.7623 | +0.1336 | 7.50% | 1.56% |
| Structural+Description | 3.2780 | +0.6179 | 7.50% | 3.75% |
| Image+Description | 5.1580 | -1.2622 | 0.31% | 5.00% |
| All three | 3.8959 | 0.0000 | 3.44% | 2.50% |

Structural có mean CE thấp nhất, là oracle của 79.38% query và được policy sample 80.94%. Ưu tiên structural ở những batch này phù hợp với continuous reward đang dùng.

Trong lần kiểm tra 1.024 validation query trước đó, structural chỉ là oracle ở 2.64%, còn description ở 50.78%. Chênh lệch gợi ý chất lượng tương đối của teacher thay đổi giữa train và validation.

Training state thay đổi giữa các batch, còn validation được đo ở checkpoint cố định. [L6](l6-teacher-generalization.md) kiểm tra lại bằng cùng frozen teacher trên toàn bộ train/validation/test.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="comparison-with-validation"></a>
<a id="counterfactual-advantage"></a>
<a id="experiment-l5---counterfactual-teacher-strategy-audit"></a>
<a id="findings"></a>
<a id="limitations"></a>
<a id="next-step"></a>
<a id="objective"></a>
<a id="results"></a>
<a id="setup"></a>
