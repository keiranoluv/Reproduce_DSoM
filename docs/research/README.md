# Kết quả tái lập

Các trang dưới đây tổng hợp kết quả đã chạy và so sánh với paper. Mỗi trang ghi cách chọn checkpoint, cấu hình và đường dẫn file kết quả. Để chạy lại, dùng [hướng dẫn thực nghiệm](../reproduction/running.md).

| Bảng | Kết quả |
|---|---|
| Table 1 | [MKG-Y và MKG-W](tables/table-1-mkg-y-mkg-w.md) |
| Table 1, DB15K | [Baseline và thử nghiệm CleanV1](tables/db15k-clean-split.md) |
| Table 2 | [So sánh cách chọn teacher](tables/table-2-teacher-selection.md) |
| Table 3 | [FB15K-237 và WN18](tables/table-3-fb15k237-wn18.md) |
| Table 4 | [Student NEKD/NNKD](tables/table-4-students.md), [teacher ensemble](tables/table-4-teachers.md) |
| Table 6 | [Chọn từng modality của teacher](tables/table-6-teacher-modalities.md) |
| Table 7 | [WN9 với raw và filtered evaluation](tables/table-7-wn9.md) |

Table 5 là thống kê dataset, nằm trong [hướng dẫn dữ liệu](../reproduction/data.md). [Các file tham chiếu](tables/reference/README.md) lưu số liệu paper và bản tổng hợp JSON để đọc khi chưa có checkpoint trên máy.

Khi đối chiếu số liệu, xem thêm [cấu hình và khác biệt với paper](methodology/protocol.md). Baseline chính giữ policy `source_frozen`; các thí nghiệm learned RC có nhãn riêng. Mỗi cấu hình chỉ có một seed.

[Nguồn dữ liệu](provenance/README.md) giúp kiểm tra lại mapping và file đầu vào. Những lần thử policy và ghi chép tiến độ trước đây được lưu trong [archive](../archive/README.md).
