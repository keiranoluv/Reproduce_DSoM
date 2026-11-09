# Table 4 — Student NEKD/NNKD theo cấu hình A5

Trạng thái: người dùng đã hoàn tất hai full runs tại
`runs/table4_components/full/20261023_075850_052725/`; kiểm tra CPU/files PASS.
NEKD test MRR .411020 (best epoch 75), NNKD .412105 (best epoch 42).
Xem [teacher rows còn lại](TABLE4_TEACHER_ENSEMBLES.md).
Trợ lý không chạy GPU khi chuẩn bị hoặc kiểm chứng.

## Phạm vi và protocol

Hai student rows cần bổ sung của [DSoM v1, Table 4](https://arxiv.org/html/2507.20738v1):

| Component | CLI coefficient | MRR paper | H@1 | H@3 | H@10 |
|---|---|---:|---:|---:|---:|
| RC + NEKD | alpha=1, beta=0 | 0.4100 | 0.3307 | 0.4493 | 0.5600 |
| RC + NNKD | alpha=0, beta=1 | 0.4233 | 0.3393 | 0.4639 | 0.5808 |

Launcher chép toàn bộ cấu hình từ A5 đã hoàn thành:
`runs/student/MMKB-DB15K/learned_rc_adam_full/20261021_173746_2988787/checkpoints/0/result.json`.
Kiểm tra learned RC có policy Adam riêng LR 1e-5, binary rewards +1/-10
và full MMKB-DB15K split gốc. Student giữ Adagrad LR 0.1, rank 2000,
batch 1000, N3 0.01, KD temperature 4, CE/KD weights 1/2, tối đa 200 epochs,
validation mỗi 3 epochs, seed 2022.

Giữ scheduler `min` để khớp A5 dù ngược chiều maximize validation Hits@10.
Chép cả input normalization, optimizer betas, feature paths, teacher checkpoint.
Full training chỉ đổi alpha/beta, định danh run và thư mục output.

Student/policy khởi tạo mới, teacher reuse từ Exp01 và luôn đóng băng; không
resume A5 student weights. RC vẫn học, không component nào dùng `source_frozen`.
Full NDKD vẫn qua `--kd_loss_type kdrl --rc_distill_loss ndkd --teacher_strategy rc`.
Hai hệ số tắt một distillation term, không tắt supervised CE/N3 hoặc RC.

## Audit công thức giữa released code và paper

`multi_dkd_loss` hiện tại trong `src/kd_optimizers.py` tính:

1. NEKD: KL giữa hai-bin distributions chứa tổng neighbor probability mass và tổng non-neighbor mass.
2. NNKD: KL giữa distributions tái chuẩn hóa trên non-neighbor entities, loại mọi training neighbors kể cả target hiện tại.
3. Tổng có trọng số `alpha * NEKD + beta * NNKD`, scale temperature bình phương và lấy trung bình batch cho cả hai terms.

NEKD **không phải** KL giữa từng neighbor probability riêng lẻ. Neighbors chỉ
lấy training triples và reciprocals, không từ validation/test filters.

**Khác biệt:** Section 4.5 mô tả neighbor bin là **trung bình** probabilities,
chia cho số neighbors. Implementation công bố dùng **tổng** qua `cat_mask`.
Hai định nghĩa trùng khi chỉ một neighbor, thường khác nếu có nhiều neighbors.

Các runs cố ý giữ SUM đã dùng trong A5, A9/A10 và reproduction hiện có, không
sửa source loss. Đây là component ablations theo released code, chưa xác minh
công thức MEAN của paper. Đổi sang MEAN cần full-NDKD baseline mới cùng cấu hình.

CPU tests tính độc lập grouped/conditional KL và gradients, kiểm tra hai
components ghép lại thành default NDKD, singleton tương đương DKD và thể hiện
khác biệt SUM/MEAN. Tests kiểm tra training-only reciprocal neighbors của
dataset loader thật và bốn full/smoke CLI configs, không import CUDA setup.

## Hướng dẫn thực thi

Lệnh chạy hiện tại, chọn từng ô/toàn bảng và prerequisites: [RUNNING.md](../reproduction/RUNNING.md).

## Kết quả và giới hạn

Launcher kiểm tra saved arguments, selected-checkpoint test metrics, checkpoint
tồn tại và ít nhất một RC policy-weight update khác không. `summary.json` và
`table4_student_results.md` đọc `final_result`, không W&B test summary cuối.
A4/A5/A9/A10 được đưa vào dưới dạng references. Smoke không có paper-MRR delta
và không phải benchmark scores.

Cách tái tạo summary được ghi trong RUNNING.md.

Khi đủ hai full runs, `component_pair_completed=true` hoàn tất sáu student
rows cùng A4/A5/A9/A10; riêng components chưa đủ toàn bảng. Kết quả Teacher Avg.,
ML-on-validation, RC-on-validation hoàn tất sau đó nằm ở [teacher ensembles](TABLE4_TEACHER_ENSEMBLES.md).
A4 là student distill từ average teacher, không phải direct Teacher Avg.
Một seed, reconstructed features và DB15K contamination vẫn là các giới hạn.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="mathematical-audit-released-code-versus-paper"></a>
<a id="results-and-limits"></a>
<a id="scope-and-protocol"></a>
<a id="table-4--a5-matched-nekd-only-and-nnkd-only-student-runs"></a>
