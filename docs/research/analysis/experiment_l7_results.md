# Thí nghiệm L7 — Audit calibration/ranking teacher

## Mục tiêu

Kiểm tra structural validation CE cao do ranking kém, quá tự tin, hay cả hai.

## Thiết lập

Dataset: FB15K-237
Teacher: frozen pretrained MIN
Split: full validation
Số queries: 35070 (original + reciprocal)

Metrics:
- Unfiltered cross-entropy
- Unfiltered MRR/Hits@1
- Top-1 confidence
- Kỳ vọng Calibration Error (ECE, 10 bins)
- CE ở temperatures 1, 2, 4

## Kết quả

| Chiến lược | CE | MRR | Hits@1 | Confidence | ECE | CE T=2 | CE T=4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Structural | 6.3162 | 0.1681 | 0.1005 | 0.1959 | 0.1021 | 6.7524 | 8.0465 |
| Image | 6.4193 | 0.1586 | 0.1091 | 0.1088 | 0.0067 | 7.3782 | 8.3677 |
| Description | 4.9181 | 0.2081 | 0.1376 | 0.1414 | 0.0154 | 6.0726 | 7.6092 |
| Structural+Image | 5.7175 | 0.1628 | 0.1022 | 0.1509 | 0.0509 | 6.9337 | 8.1791 |
| Structural+Description | 5.2440 | 0.1765 | 0.1088 | 0.1749 | 0.0665 | 6.3178 | 7.8061 |
| Image+Description | 5.2477 | 0.1935 | 0.1278 | 0.1377 | 0.0156 | 6.5539 | 7.9476 |
| All three | 5.2626 | 0.1760 | 0.1111 | 0.1534 | 0.0466 | 6.5573 | 7.9675 |

## Phát hiện

1. Description validation CE thấp hơn Structural.
2. Description cũng có unfiltered MRR/Hits@1 cao hơn.
3. Structural top-1 confidence cao hơn dù accuracy thấp hơn.
4. Structural ECE cao hơn Description.
5. Tăng temperature từ 1 lên 2 hoặc 4 làm CE tệ hơn cho cả hai modalities.

## Diễn giải

Không thể giải thích gap chỉ bằng confidence: ranking quality cũng khác.

Chỉ kiểm tra T=1,2,4; chưa loại trừ các calibration temperatures khác.

## Giới hạn

- Unfiltered ranking khác filtered Table 3.
- Metrics đánh giá teacher, không student.
- ECE nhạy với binning và top-1 accuracy.

## Thí nghiệm tiếp theo

L8: audit full filtered ranking cho bảy strategies bằng FB15K-237 to_skip filters.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="experiment-l7---teacher-calibration-and-ranking-audit"></a>
<a id="findings"></a>
<a id="interpretation"></a>
<a id="limitations"></a>
<a id="next-experiment"></a>
<a id="objective"></a>
<a id="results"></a>
<a id="setup"></a>
