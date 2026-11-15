# L7: ranking và độ tự tin của teacher

L7 kiểm tra structural có CE validation cao vì xếp hạng kém, quá tự tin hay cả hai. Dùng frozen pretrained MIN teacher trên toàn bộ FB15K-237 validation: 35.070 original và reciprocal query.

Các chỉ số gồm unfiltered CE/MRR/Hits@1, top-1 confidence, Expected Calibration Error (ECE, 10 bin) và CE ở temperature 1, 2, 4.

| Chiến lược | CE | MRR | Hits@1 | Confidence | ECE | CE T=2 | CE T=4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Structural | 6.3162 | 0.1681 | 0.1005 | 0.1959 | 0.1021 | 6.7524 | 8.0465 |
| Image | 6.4193 | 0.1586 | 0.1091 | 0.1088 | 0.0067 | 7.3782 | 8.3677 |
| Description | 4.9181 | 0.2081 | 0.1376 | 0.1414 | 0.0154 | 6.0726 | 7.6092 |
| Structural+Image | 5.7175 | 0.1628 | 0.1022 | 0.1509 | 0.0509 | 6.9337 | 8.1791 |
| Structural+Description | 5.2440 | 0.1765 | 0.1088 | 0.1749 | 0.0665 | 6.3178 | 7.8061 |
| Image+Description | 5.2477 | 0.1935 | 0.1278 | 0.1377 | 0.0156 | 6.5539 | 7.9476 |
| All three | 5.2626 | 0.1760 | 0.1111 | 0.1534 | 0.0466 | 6.5573 | 7.9675 |

Description vừa có CE thấp hơn vừa có unfiltered MRR/Hits@1 cao hơn structural. Structural tự tin hơn dù accuracy thấp hơn, và ECE cũng cao hơn. Vì ranking khác nhau, gap CE không chỉ đến từ confidence.

Tăng temperature lên 2 hoặc 4 làm CE của cả hai modality xấu hơn; lần thử này chưa khảo sát các temperature khác. ECE cũng phụ thuộc cách chia bin. Các số liệu đánh giá teacher, còn ảnh hưởng tới student cần kiểm tra riêng.

[L8](l8-filtered-ranking.md) dùng filtered ranking để đối chiếu với cách đánh giá Table 3.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="experiment-l7---teacher-calibration-and-ranking-audit"></a>
<a id="findings"></a>
<a id="interpretation"></a>
<a id="limitations"></a>
<a id="next-experiment"></a>
<a id="objective"></a>
<a id="results"></a>
<a id="setup"></a>
