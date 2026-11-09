# Thí nghiệm L6 — Audit teacher generalization gap

## Mục tiêu

Đo chất lượng teacher strategies trên train/validation/test bằng cùng frozen Teacher MIN checkpoint.

## Thiết lập

Dataset: FB15K-237
Đánh giá: full dataset
Teacher: pretrained MIN, frozen
Số queries: original + reciprocal
Chỉ số: unfiltered cross-entropy (CE)

Số queries:
- Train: 544230
- Validation: 35070
- Test: 40932

## Cross-entropy trung bình

| Chiến lược | Train | Valid | Test |
|---|---:|---:|---:|
| Structural | 2.6913 | 6.3162 | 6.3383 |
| Image | 6.1023 | 6.4193 | 6.4497 |
| Description | 4.6177 | 4.9181 | 4.9370 |
| Structural+Image | 3.6250 | 5.7175 | 5.7435 |
| Structural+Description | 3.2194 | 5.2440 | 5.2609 |
| Image+Description | 4.9135 | 5.2477 | 5.2702 |
| All three | 3.7595 | 5.2626 | 5.2836 |

## Phân phối CE oracle strategies (%)

| Chiến lược | Train | Valid | Test |
|---|---:|---:|---:|
| Structural | 73.74 | 2.44 | 2.42 |
| Image | 0.89 | 5.32 | 5.37 |
| Description | 2.01 | 50.29 | 50.30 |
| Structural+Image | 9.35 | 4.01 | 3.92 |
| Structural+Description | 10.69 | 11.59 | 11.83 |
| Image+Description | 1.34 | 21.91 | 21.21 |
| All three | 1.99 | 4.44 | 4.95 |

## CE gap từ train sang validation

| Chiến lược | Gap |
|---|---:|
| Structural | +3.6249 |
| Image | +0.3170 |
| Description | +0.3004 |
| Structural+Image | +2.0925 |
| Structural+Description | +2.0247 |
| Image+Description | +0.3342 |
| All three | +1.5031 |

## So sánh theo cặp

Structural có CE thấp hơn Description:
- Train: 92.48%
- Validation: 9.69%
- Test: 9.62%

Description có CE thấp hơn Structural:
- Train: 7.52%
- Validation: 90.31%
- Test: 90.38%

## Diễn giải

Frozen teacher có train-to-validation CE gap lớn ở structural predictions.

Chất lượng tương đối Structural/Description đảo chiều giữa training và unseen triples.

Đây là bằng chứng teacher-quality mismatch có thể góp phần khiến RC ưu tiên
Structural khi training nhưng chọn teacher kém trên validation.

## Giới hạn

- CE là unfiltered, khác filtered ranking ở Table 3.
- Chưa chứng minh nguyên nhân overfitting.
- Chưa chứng minh RC strategy khác cải thiện student MRR/Hits@K.

## Thí nghiệm tiếp theo

L7: so teacher ranking quality, confidence và temperature sensitivity trên held-out data.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
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
