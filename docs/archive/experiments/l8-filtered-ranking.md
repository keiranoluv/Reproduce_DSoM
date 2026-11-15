# L8: filtered ranking của bảy teacher strategy

L8 đánh giá cùng frozen pretrained MIN teacher trên FB15K-237 validation: 17.535 triple, tương ứng 35.070 query RHS/LHS. Filtering dùng `dataset.to_skip`.

| Chiến lược | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|---:|
| Structural | 152.03 | 0.3643 | 0.2714 | 0.3980 | 0.5515 |
| Image | 960.54 | 0.2193 | 0.1641 | 0.2392 | 0.3284 |
| Description | 152.32 | 0.3394 | 0.2566 | 0.3681 | 0.5045 |
| Structural+Image | 197.61 | 0.3205 | 0.2357 | 0.3513 | 0.4889 |
| Structural+Description | 110.80 | 0.3790 | 0.2855 | 0.4159 | 0.5663 |
| Image+Description | 193.01 | 0.3098 | 0.2329 | 0.3366 | 0.4606 |
| All three | 130.62 | 0.3498 | 0.2618 | 0.3825 | 0.5246 |

Structural+Description đứng đầu ở MR, MRR và cả ba Hits. Structural có filtered MRR cao hơn description, khác kết quả unfiltered CE/MRR trong L6/L7. Cách filtering ảnh hưởng rõ tới thứ hạng teacher trong lần kiểm tra này.

Một teacher có filtered MRR cao hơn chưa chắc distill student tốt hơn. [L9](l9-structural-description.md) train student với fixed Structural+Description để so với K1 description và K2 learned RC. Tất cả phép kiểm tra ở đây dùng cùng một pretrained teacher checkpoint.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="experiment-l8---filtered-teacher-ranking-audit"></a>
<a id="findings"></a>
<a id="limitations"></a>
<a id="next-experiment"></a>
<a id="objective"></a>
<a id="results---both-directions"></a>
<a id="setup"></a>
