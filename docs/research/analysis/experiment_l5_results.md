# Thí nghiệm L5 — Audit teacher strategy phản thực

## Mục tiêu

Kiểm tra vì sao RC chọn Structural ở những steps đầu khi dùng continuous CE reward.

## Thiết lập

- Dataset: FB15K-237
- Baseline: L3a (continuous CE reward, raw policy input)
- Mẫu training: 2048
- Batch size: 16
- Số epochs: 1
- Tổng optimizer steps: 128
- Diagnostic batches: 20 đầu
- Training queries audit: 320
- Teacher: pretrained MIN đóng băng

## Advantage phản thực

Đánh giá cả bảy teacher strategies cho mỗi training query.

Advantage(strategy) =
    CE(average teacher) - CE(selected strategy)

Advantage dương nghĩa là strategy có CE thấp hơn average-teacher baseline.

## Kết quả

| Chiến lược | Mean CE | Mean Advantage | Oracle % | Sampled % |
|---|---:|---:|---:|---:|
| Structural | 2.7072 | +1.1887 | 79.38% | 80.94% |
| Image | 6.4623 | -2.5664 | 0.94% | 1.56% |
| Description | 4.7788 | -0.8829 | 0.94% | 4.69% |
| Structural+Image | 3.7623 | +0.1336 | 7.50% | 1.56% |
| Structural+Description | 3.2780 | +0.6179 | 7.50% | 3.75% |
| Image+Description | 5.1580 | -1.2622 | 0.31% | 5.00% |
| All three | 3.8959 | 0.0000 | 3.44% | 2.50% |

## Phát hiện

1. Structural có mean CE thấp nhất trên 320 training queries đầu.
2. Structural là CE oracle của 79.38% queries này.
3. Policy sample Structural trên 80.94% queries.
4. Structural nhận mean counterfactual advantage dương.
5. Ưu tiên Structural ban đầu phù hợp continuous CE reward trên training queries.

## So sánh với validation

Trong audit 1024 validation queries trước đó:
- Structural oracle: 2.64%
- Description oracle: 50.78%

Khác biệt gợi ý có thể có teacher-quality gap giữa train và validation.

## Giới hạn

- Chỉ audit 20 training batches đầu.
- Model state thay đổi giữa steps.
- Validation audit dùng checkpoint cố định.
- Chưa chứng minh nhân quả hoặc overfitting.
- Cần audit frozen teacher trên train/validation/test.

## Bước tiếp theo

L6: audit generalization gap bằng cùng frozen Teacher MIN checkpoint trên train, validation, test.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="comparison-with-validation"></a>
<a id="counterfactual-advantage"></a>
<a id="experiment-l5---counterfactual-teacher-strategy-audit"></a>
<a id="findings"></a>
<a id="limitations"></a>
<a id="next-step"></a>
<a id="objective"></a>
<a id="results"></a>
<a id="setup"></a>
