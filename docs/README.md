# Tài liệu DSoM

Nếu bắt đầu từ một bản clone mới, làm lần lượt bốn bước:

1. [Cài môi trường](reproduction/environment.md): tạo virtual environment, cài PyTorch và kiểm tra GPU.
2. [Chuẩn bị dữ liệu](reproduction/data.md): tải dataset, tạo mapping và trích xuất đặc trưng.
3. [Chạy thực nghiệm](reproduction/running.md): chọn bảng, chạy thử rồi train đầy đủ.
4. [Đọc kết quả](reproduction/outputs.md): tìm checkpoint được chọn, tổng hợp chỉ số và xử lý lỗi.

Muốn xem số liệu đã chạy, mở [kết quả và so sánh với paper](research/README.md). Các trang này ghi cấu hình, bảng số liệu và khác biệt cần lưu ý khi đối chiếu.

[Cấu hình thực nghiệm](research/methodology/protocol.md) giải thích `source_frozen`, cách chọn checkpoint và các khác biệt giữa code với paper. [Nguồn dữ liệu](research/provenance/README.md) lưu thông tin để kiểm tra lại dataset và mapping.

[Tài liệu cũ](archive/README.md) gồm các lần thử policy, ghi chú tiến độ và hướng dẫn đã có bản thay thế. Các trạng thái “đang chạy” hoặc “chưa hoàn tất” trong đó thuộc thời điểm viết.
