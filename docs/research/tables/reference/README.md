# Số liệu và file tham chiếu

Thư mục này lưu giá trị paper và bản tổng hợp kết quả đã chạy. Có thể mở JSON để xem chỉ số khi chưa có checkpoint trên máy.

| File | Dùng để làm gì? |
|---|---|
| `table2_paper_targets.json`, `table6_paper_targets.json` | Giá trị paper để runner đối chiếu kết quả |
| `table4_ensemble_sources.json` | Commit/hash nguồn dùng khi dựng teacher ensemble |
| `table2_results_20261023.json` | Kết quả Table 2 và đường dẫn/hash từng result |
| `table6_results_20261023.json` | Kết quả Table 6, gồm các ô dùng lại |
| `table7_results_20261024.json` | Kết quả WN9 raw/filtered |

Runner đọc một số file trong thư mục này khi lập plan hoặc kiểm tra kết quả. Giữ bản tổng hợp gắn với suite gốc; lần chạy mới có summary riêng dưới `runs/`.
