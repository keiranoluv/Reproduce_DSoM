# Nguồn dữ liệu và kiểm tra mapping

Các file ở đây giúp đối chiếu dữ liệu đầu vào với nguồn công khai. Để tải và chuẩn bị dataset, dùng [hướng dẫn dữ liệu](../../reproduction/data.md).

| File | Nội dung |
|---|---|
| [DB15K source trace](db15k_source_trace.json) | File ID, URL, hash và số lượng của raw DB15K |
| [WN9 source trace](wn9_source_trace.json) | Commit MoSE, hash nguồn và mapping đặc trưng |
| [Nguồn WN9](wn9-sources.md) | Kiểm tra triples, inverse facts và cách lấy đặc trưng WN18 |
| [Trích đoạn code Table 3](table3_code_audit.txt) | Nhánh/commit và dòng code được kiểm tra khi phân tích policy |

JSON và trích đoạn code giữ thông tin tại thời điểm kiểm tra. Các đường dẫn `runs/` trong đó trỏ tới file trên máy thực nghiệm, được giữ ngoài Git.
