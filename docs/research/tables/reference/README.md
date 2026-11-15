# Dữ liệu tham chiếu cho bảng

Các JSON trong thư mục này là targets, nguồn ensemble hoặc snapshot artifacts được dùng bởi report/runner tương ứng. Runner đọc một số file này để lập kế hoạch hoặc kiểm tra dữ liệu tham chiếu; không sửa file để thay đổi một lần chạy đã ghi nhận.

- `table2_paper_targets.json`, `table6_paper_targets.json`: giá trị tham chiếu được dùng ở các bảng tương ứng.
- `table4_ensemble_sources.json`: nguồn checkpoint cho các tổ hợp teacher ensemble.
- `table2_results_*.json`, `table6_results_*.json`, `table7_results_*.json`: snapshot có timestamp, gắn với báo cáo và hồ sơ lần chạy tương ứng.
