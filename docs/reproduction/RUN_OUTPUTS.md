# Kiểm tra output và xử lý lỗi

Hướng dẫn này dùng sau khi đã chạy `plan`, `smoke` hoặc `full`. Mỗi suite nằm trong một thư mục mới dưới `runs/`; giữ nguyên thư mục đó để lưu cấu hình, lệnh, log và checkpoint của lần chạy.

## Xác nhận workflow hoàn tất

- Với workflow baseline, kiểm tra `teacher/stage_result.json` và `student/stage_result.json`; stage hoàn tất có `status: PASS`.
- Với runner theo bảng, mở `summary.json`, kiểm tra `failures` rỗng và cờ hoàn tất phù hợp với toàn bảng hoặc phần dữ liệu đã chọn.
- `smoke` chỉ xác nhận một lần chạy nhỏ có thể thực thi; nó không thay thế một lần chạy `full`.
- Nếu cần lưu kết quả của student, dùng `checkpoints/0/result.json` của checkpoint được chọn theo validation; trường `final_result` lưu các chỉ số cho checkpoint đó. Không dùng test log cuối epoch thay cho checkpoint đã chọn.
- Workflow Table 7 ghi thêm `student/evaluation.json` khi hoàn thành bước đánh giá độc lập.

Các thư mục suite và tên file chi tiết được ghi ở từng mục trong `RUNNING.md`. Lưu thêm commit mã nguồn, cấu hình, seed, nguồn dữ liệu và thông tin môi trường cùng hồ sơ chạy.

## Vị trí output thường gặp

| Workflow | Thư mục output |
|---|---|
| Baseline | `runs/reproduction/DATASET/baseline/{plan,smoke,full}/SUITE/` |
| Table 2 | `runs/table2/{plan,smoke,full}/SUITE/` |
| Table 4 components | `runs/table4_components/{plan,smoke,full}/SUITE/` |
| Table 4 ensembles | `runs/table4_ensembles/{plan,smoke,full}/SUITE/` |
| Table 6 | `runs/table6/{plan,smoke,full}/SUITE/` |
| Table 7 | `runs/table7/{plan,smoke,full}/SUITE/` |

Giữ nguyên raw inputs, manifests, registry và checkpoint liên quan tới một suite. Các launcher có kiểm tra fingerprint/hash để phát hiện khi dữ liệu hoặc tham chiếu không còn khớp; không sửa manifest để bỏ qua kiểm tra đó. `runs/` và dữ liệu tải về được Git ignore.

## Lỗi thường gặp

| Thông báo hoặc tình huống | Việc cần kiểm tra |
|---|---|
| Thiếu `torch`, `transformers` hoặc package khác | Kích hoạt đúng virtual environment; xem `ENVIRONMENT.md`, chạy environment checker và `python -m pip check`. |
| `Commit tracked changes` hoặc source chưa được track | Review và commit thay đổi source/config cần thiết trước `full`; không commit dữ liệu hay thư mục `runs/`. |
| Thiếu baseline hoặc reference | Dùng registry cục bộ nhất quán và chạy trước prerequisite được ghi trong mục workflow tương ứng của `RUNNING.md`. |
| Registry có key trùng | Dùng registry riêng cho lượt chạy mới; không ghi đè entry của suite cũ. |
| Dataset, feature hoặc output đã tồn tại | Dùng lệnh verify nếu có hoặc chọn thư mục output/checkout riêng; không xóa artifacts cũ để vượt qua kiểm tra. |
| Mapping, số dòng hoặc coverage không khớp | Kiểm tra đúng dataset version, tên file và mapping; không đổi split để làm qua kiểm tra. |
| Không tải được asset hoặc encoder weights | Tải thủ công từ nguồn được liệt kê trong `DATA.md`; khi chạy offline, chuẩn bị cache encoder trước. |
| CUDA, driver hoặc VRAM lỗi | Chạy kiểm tra GPU trên máy, xem log và giảm batch size chỉ khi chấp nhận ghi nhận cấu hình chạy khác. |
| Lỗi W&B | Đăng nhập lại, đặt `WANDB_MODE=offline` hoặc bỏ `--wandb`; lưu output cục bộ không cần W&B. |
| Student yêu cầu full teacher | Chạy teacher ở chế độ `full`; workflow `full` không tiếp tục từ teacher smoke. |
| Checkpoint hoặc hash không khớp | Dừng lần chạy, kiểm tra plan/log và nguồn artifacts; không ghi đè hoặc báo cáo output chưa xác minh. |

Workflow full tạo lượt chạy mới; nó không tiếp tục optimizer từ smoke. Stage lỗi giữ log trong suite để điều tra. Khi thay đổi batch size, input hoặc cấu hình, ghi lại thay đổi cùng lần chạy.
