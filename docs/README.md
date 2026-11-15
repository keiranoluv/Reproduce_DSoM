# Tài liệu DSoM

Trang này là mục lục chính của tài liệu. Nếu muốn chạy lại từ một clone mới, hãy đi theo thứ tự **môi trường → dữ liệu → workflow chạy → kiểm tra artifacts**. Chỉ cần mở các báo cáo nghiên cứu riêng khi cần xem hồ sơ hoặc phân tích của một bảng.

## Chạy lại thực nghiệm

1. [Cài môi trường](reproduction/ENVIRONMENT.md) — Python, PyTorch/CUDA, GPU và dependencies.
2. [Chuẩn bị dữ liệu](reproduction/DATA.md) — nguồn tải, cấu trúc thư mục, preprocessing và feature extraction.
3. [Chọn và chạy workflow](reproduction/RUNNING.md) — lệnh `plan`, `smoke`, `full`, prerequisites và vị trí output.
4. [Kiểm tra output và xử lý lỗi](reproduction/RUN_OUTPUTS.md) — kiểm tra trạng thái suite, artifacts và lỗi thường gặp.

Đi theo thứ tự trên; các hướng dẫn này là nguồn lệnh hiện hành. Không cần chạy mọi bảng nếu chỉ tái lập một workflow.

## Hồ sơ nghiên cứu

[Mục lục báo cáo](research/README.md) chia nội dung thành báo cáo theo bảng, phân tích chẩn đoán, protocol và provenance dữ liệu/mã nguồn. Đây là tài liệu nghiên cứu và bằng chứng, không phải hướng dẫn cài đặt.

## Tài liệu lưu trữ

[Archive](archive/README.md) chứa handoff, hướng dẫn cũ và tài liệu đã được thay thế. Các lệnh và trạng thái trong đó có thể không còn áp dụng cho clone hiện tại.
