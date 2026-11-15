# Table 4: tách NEKD và NNKD

Hai cấu hình student đã hoàn tất trong `runs/table4_components/full/20261023_075850_052725/`: RC + NEKD có test MRR 0.411020 tại epoch 75; RC + NNKD có test MRR 0.412105 tại epoch 42. Chỉ số lấy từ checkpoint được chọn theo validation MRR.

## Hai thành phần được so sánh

Giá trị tham chiếu từ [DSoM v1, Table 4](https://arxiv.org/html/2507.20738v1):

| Component | CLI coefficient | MRR paper | H@1 | H@3 | H@10 |
|---|---|---:|---:|---:|---:|
| RC + NEKD | alpha=1, beta=0 | 0.4100 | 0.3307 | 0.4493 | 0.5600 |
| RC + NNKD | alpha=0, beta=1 | 0.4233 | 0.3393 | 0.4639 | 0.5808 |

Runner chép cấu hình A5 từ `runs/student/MMKB-DB15K/learned_rc_adam_full/20261021_173746_2988787/checkpoints/0/result.json`, rồi đổi alpha/beta. Teacher dùng lại từ Exp01 và đóng băng; student và policy được khởi tạo mới.

Student dùng rank 2000, batch 1000, Adagrad LR 0.1, N3 0.01, temperature 4, CE/KD weights 1/2, tối đa 200 epoch, validation mỗi 3 epoch, seed 2022. Policy dùng Adam riêng LR `1e-5`, reward +1/-10. Scheduler giữ `min` như A5. Feature paths, teacher checkpoint, optimizer betas và input normalization cũng được giữ nguyên.

Cả hai dùng `--kd_loss_type kdrl --rc_distill_loss ndkd --teacher_strategy rc`. Đặt alpha hoặc beta bằng zero chỉ tắt một term distillation; supervised CE, N3 và policy vẫn hoạt động.

## Đọc công thức trong code

`multi_dkd_loss` trong `src/kd_optimizers.py` gồm:

1. NEKD: KL giữa hai nhóm xác suất, neighbor và non-neighbor.
2. NNKD: KL trên phân phối đã chuẩn hóa lại của các non-neighbor entity, loại cả target và các training neighbor.
3. Tổng `alpha * NEKD + beta * NNKD`, nhân temperature bình phương và lấy trung bình batch.

Neighbor chỉ đến từ training triples và reciprocals. NEKD so khối xác suất của hai nhóm, không so từng neighbor riêng lẻ.

Code công bố cộng xác suất neighbor bằng SUM; §4.5 của paper mô tả MEAN. Hai cách trùng khi chỉ có một neighbor và thường khác khi có nhiều neighbor. Các cấu hình ở đây giữ SUM như A5/A9/A10. Nếu thử MEAN, cần một baseline full-NDKD cùng cấu hình để đối chiếu.

## Kết quả và cách chạy lại

Xem [bảng số liệu và teacher ensemble](table-4-teachers.md) hoặc [lệnh Table 4](../../reproduction/running.md#table-4). Runner cần đủ A4/A5/A9/A10; sau khi hai component hoàn tất trong cùng suite, `component_pair_completed=true` xác nhận đủ sáu student row.

`summary.json` và `table4_student_results.md` chứa `final_result`, reference và epoch. Kiểm tra file checkpoint, tham số đã lưu và policy weight update xác nhận hai lần chạy dùng learned RC. Các phép kiểm tra CPU còn đối chiếu grouped/conditional KL, gradient, phép ghép hai term và training-only neighbor.

A4 là student được distill từ average teacher. Ba dòng đánh giá trực tiếp teacher ensemble được ghi riêng. Một seed, DB15K overlap và đặc trưng dựng lại vẫn ảnh hưởng cách diễn giải bảng.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="mathematical-audit-released-code-versus-paper"></a>
<a id="results-and-limits"></a>
<a id="scope-and-protocol"></a>
<a id="table-4--a5-matched-nekd-only-and-nnkd-only-student-runs"></a>
