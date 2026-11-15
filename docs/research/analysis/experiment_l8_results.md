# Thí nghiệm L8 — Audit filtered ranking teacher

## Mục tiêu

Đánh giá bảy frozen teacher strategies bằng filtered ranking trên FB15K-237.

## Thiết lập

- Dataset: FB15K-237
- Teacher: frozen pretrained MIN
- Split: full validation
- Triples gốc: 17535
- Queries evaluation: 35070 (RHS + LHS)
- Protocol: filtered ranking dùng dataset.to_skip

## Kết quả hai chiều

| Chiến lược | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|---:|
| Structural | 152.03 | 0.3643 | 0.2714 | 0.3980 | 0.5515 |
| Image | 960.54 | 0.2193 | 0.1641 | 0.2392 | 0.3284 |
| Description | 152.32 | 0.3394 | 0.2566 | 0.3681 | 0.5045 |
| Structural+Image | 197.61 | 0.3205 | 0.2357 | 0.3513 | 0.4889 |
| Structural+Description | 110.80 | 0.3790 | 0.2855 | 0.4159 | 0.5663 |
| Image+Description | 193.01 | 0.3098 | 0.2329 | 0.3366 | 0.4606 |
| All three | 130.62 | 0.3498 | 0.2618 | 0.3825 | 0.5246 |

## Phát hiện

1. Structural+Description tốt nhất về filtered MR/MRR/Hits@1/3/10.
2. Structural filtered MRR cao hơn Description.
3. Khác với unfiltered CE/MRR của L6/L7, nơi Description tốt hơn.
4. Chọn filtered/unfiltered ảnh hưởng đáng kể so sánh teacher.

## Giới hạn

- Đánh giá teacher strategies, không student.
- Teacher filtered MRR cao hơn không đảm bảo distill tốt hơn.
- Chỉ một pretrained teacher checkpoint.

## Thí nghiệm tiếp theo

L9: train student với fixed Structural+Description, so với K1 Description và K2 RC.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="experiment-l8---filtered-teacher-ranking-audit"></a>
<a id="findings"></a>
<a id="limitations"></a>
<a id="next-experiment"></a>
<a id="objective"></a>
<a id="results---both-directions"></a>
<a id="setup"></a>
