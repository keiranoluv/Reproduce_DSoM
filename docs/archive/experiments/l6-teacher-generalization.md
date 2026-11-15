# L6: so chất lượng teacher giữa train, validation và test

L6 dùng cùng frozen pretrained MIN checkpoint trên FB15K-237. Chỉ số chính là unfiltered cross-entropy, tính đủ original và reciprocal query: 544.230 train, 35.070 validation và 40.932 test.

## CE trung bình

| Chiến lược | Train | Valid | Test |
|---|---:|---:|---:|
| Structural | 2.6913 | 6.3162 | 6.3383 |
| Image | 6.1023 | 6.4193 | 6.4497 |
| Description | 4.6177 | 4.9181 | 4.9370 |
| Structural+Image | 3.6250 | 5.7175 | 5.7435 |
| Structural+Description | 3.2194 | 5.2440 | 5.2609 |
| Image+Description | 4.9135 | 5.2477 | 5.2702 |
| All three | 3.7595 | 5.2626 | 5.2836 |

## Strategy có CE thấp nhất theo query

| Chiến lược | Train | Valid | Test |
|---|---:|---:|---:|
| Structural | 73.74 | 2.44 | 2.42 |
| Image | 0.89 | 5.32 | 5.37 |
| Description | 2.01 | 50.29 | 50.30 |
| Structural+Image | 9.35 | 4.01 | 3.92 |
| Structural+Description | 10.69 | 11.59 | 11.83 |
| Image+Description | 1.34 | 21.91 | 21.21 |
| All three | 1.99 | 4.44 | 4.95 |

## Mức tăng CE từ train sang validation

| Chiến lược | Gap |
|---|---:|
| Structural | +3.6249 |
| Image | +0.3170 |
| Description | +0.3004 |
| Structural+Image | +2.0925 |
| Structural+Description | +2.0247 |
| Image+Description | +0.3342 |
| All three | +1.5031 |

Structural có CE thấp hơn description trên 92.48% train query, nhưng chỉ 9.69% validation và 9.62% test query. Theo chiều ngược lại, description thắng ở 7.52% train, 90.31% validation và 90.38% test.

Chất lượng tương đối của hai teacher đảo chiều giữa train và held-out triples. Điều này có thể góp phần giải thích vì sao policy học chọn structural khi train nhưng có CE regret cao trên validation.

CE ở đây là unfiltered, khác filtered ranking trong Table 3. Kết quả chưa chỉ ra nguyên nhân overfitting hoặc chứng minh đổi strategy sẽ cải thiện student. [L7](l7-teacher-calibration.md) kiểm tra thêm ranking, confidence và temperature.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="ce-oracle-strategy-distribution-"></a>
<a id="experiment-l6---teacher-generalization-gap-audit"></a>
<a id="interpretation"></a>
<a id="limitations"></a>
<a id="mean-cross-entropy"></a>
<a id="next-experiment"></a>
<a id="objective"></a>
<a id="pairwise-comparison"></a>
<a id="setup"></a>
<a id="train-to-validation-ce-gap"></a>
