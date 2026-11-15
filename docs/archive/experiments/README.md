# Các lần thử policy trước đây

Những ghi chép này theo dõi quá trình khảo sát RC trên FB15K-237 và DB15K. Các cấu hình nhỏ giúp kiểm tra hành vi policy; kết quả chính theo bảng nằm trong [mục kết quả](../../research/README.md).

| Nhóm | Nội dung |
|---|---|
| [Optimizer policy](policy-optimizers.md) | So Adagrad và Adam ở các learning rate |
| [Thí nghiệm E-I](e-i-policy-diagnostics.md) | Theo dõi strategy, reward và hướng cập nhật |
| [J: fixed strategy](j-fixed-strategies.md) | So teacher cố định với learned RC trên tập nhỏ |
| [K: full-data pilot](k-full-data.md) | Description và learned RC, năm epoch |
| [L0/L1](l0-l1-policy-momentum.md) | Thử bỏ first-moment momentum Adam |
| [L2](l2-normalized-input.md) | Chuẩn hóa input policy |
| [L3](l3-continuous-reward.md), [L4](l4-policy-learning-rate.md) | Continuous reward và learning rate |
| [L5](l5-strategy-advantage.md), [L6](l6-teacher-generalization.md) | Đối chiếu CE giữa các strategy và split |
| [L7](l7-teacher-calibration.md), [L8](l8-filtered-ranking.md) | Calibration, raw và filtered ranking |
| [L9](l9-structural-description.md) | Student với teacher Structural+Description |
| [Saturation bước đầu](first-step-saturation.md) | Probe trước/sau cùng optimizer step |
| [Policy DB15K](db15k-policy.md) | A5/A9/A10, relation và hướng query |

Các đường dẫn log/checkpoint thuộc máy chạy cũ. Lệnh dùng cho clone mới nằm trong [hướng dẫn thực nghiệm](../../reproduction/running.md).
