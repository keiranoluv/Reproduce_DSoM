# Đọc kết quả và xử lý lỗi

Mỗi lần chạy tạo một thư mục mới dưới `runs/`. Trong đó có cấu hình, lệnh, log, checkpoint và kết quả; giữ cả thư mục nếu muốn kiểm tra hoặc tổng hợp lại sau này.

## 1. Kiểm tra lần chạy đã hoàn tất

Với baseline, mở `teacher/stage_result.json` và `student/stage_result.json`. Stage hoàn tất có `status: PASS`. Với runner theo bảng, mở `summary.json`, kiểm tra `failures` rỗng và cờ hoàn tất tương ứng với phần đã chọn.

Một suite chỉ chạy một ô vẫn có thể PASS trong khi cờ toàn bảng là `false`. Tương tự, smoke thành công cho biết lần chạy nhỏ thực thi được; cần chạy `full` để lấy kết quả so với paper.

## 2. Lấy chỉ số từ checkpoint được chọn

Student lưu kết quả trong `checkpoints/0/result.json`. Dùng trường `final_result`, chứa chỉ số test của checkpoint có validation MRR tốt nhất. Log test ở epoch cuối hoặc W&B summary có thể thuộc checkpoint khác.

Table 7 ghi thêm `student/evaluation.json` sau khi nạp lại checkpoint và đánh giá độc lập. File này có cả kết quả raw/filtered, theo từng chiều và trung bình hai chiều.

## 3. Tìm thư mục đầu ra

| Nhóm thực nghiệm | Thư mục kết quả |
|---|---|
| Baseline | `runs/reproduction/DATASET/baseline/{plan,smoke,full}/SUITE/` |
| Table 2 | `runs/table2/{plan,smoke,full}/SUITE/` |
| Table 4 components | `runs/table4_components/{plan,smoke,full}/SUITE/` |
| Table 4 ensembles | `runs/table4_ensembles/{plan,smoke,full}/SUITE/` |
| Table 6 | `runs/table6/{plan,smoke,full}/SUITE/` |
| Table 7 | `runs/table7/{plan,smoke,full}/SUITE/` |

`DATASET`, `MODE`, `SUITE` là tên dataset, chế độ và thư mục của lần chạy thực tế. Từng mục trong [hướng dẫn chạy](running.md) ghi thêm tên file riêng của bảng.

Lưu cùng kết quả commit mã nguồn, seed, môi trường và nguồn dữ liệu. Launcher dùng hash để kiểm tra input, checkpoint và reference còn khớp; nếu có lỗi hash, đối chiếu lại các file đó với plan và log. Dữ liệu tải về, `runs/` và registry cục bộ được Git ignore.

## 4. Lỗi thường gặp

| Thông báo hoặc tình huống | Việc cần kiểm tra |
|---|---|
| Thiếu `torch`, `transformers` hoặc package khác | Kích hoạt đúng virtual environment, làm theo [cài môi trường](environment.md) rồi chạy `python -m pip check`. |
| `Commit tracked changes` hoặc source chưa được track | Kiểm tra và commit thay đổi source/config cần thiết trước `full`. Dữ liệu và `runs/` được giữ ngoài Git. |
| Thiếu baseline hoặc reference | Dùng cùng một registry và chạy trước các cấu hình phụ thuộc trong [hướng dẫn thực nghiệm](running.md). |
| Registry có key trùng | Dùng registry riêng cho lần chạy mới để giữ tham chiếu tới kết quả cũ. |
| Dataset, feature hoặc output đã tồn tại | Dùng lệnh verify để kiểm tra dữ liệu đã có; nếu tạo bộ mới, chọn thư mục đầu ra hoặc checkout riêng. |
| Mapping, số dòng hoặc độ phủ không khớp | Đối chiếu phiên bản dataset, tên file và mapping với nguồn trong hướng dẫn dữ liệu. |
| Không tải được dữ liệu hoặc trọng số encoder | Tải thủ công từ nguồn trong [hướng dẫn dữ liệu](data.md); khi chạy offline, chuẩn bị cache encoder trước. |
| CUDA, driver hoặc VRAM lỗi | Chạy kiểm tra GPU và xem log. Nếu giảm batch size để vừa VRAM, ghi giá trị mới cùng kết quả. |
| Lỗi W&B | Đăng nhập lại, đặt `WANDB_MODE=offline` hoặc bỏ `--wandb`; lưu output cục bộ không cần W&B. |
| Student yêu cầu full teacher | Hoàn tất teacher ở chế độ `full`, rồi truyền đúng thư mục stage cho student. |
| Checkpoint hoặc hash không khớp | Đối chiếu plan, log và file đầu vào để tìm phần đã thay đổi trước khi chạy tiếp. |

Stage lỗi vẫn giữ log trong suite. Khi thử cấu hình khác, chẳng hạn giảm batch size do thiếu VRAM, ghi lại thay đổi cùng kết quả mới để tiện đối chiếu.
