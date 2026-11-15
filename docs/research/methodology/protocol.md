# Cấu hình thực nghiệm và khác biệt với paper

Trang này giải thích các lựa chọn chung của repo để đọc và so sánh kết quả. Lệnh chạy từng bảng nằm trong [hướng dẫn thực nghiệm](../../reproduction/running.md).

## Model và cấu hình chung

Các runner dùng `ComplExMDR`: teacher có ba modality cấu trúc, ảnh và mô tả; student chỉ dùng cấu trúc. Những nhánh legacy như RSME, `soft_kl` và assistant distiller chưa được hoàn thiện trong workflow này. Các mô hình đối chứng như TMR/DySarl cần mã chạy riêng.

| Tham số | Giá trị baseline |
|---|---|
| Rank | 2000; vector ComplEx có `2 * rank` thành phần |
| Batch size | 1000 |
| Optimizer | Adagrad, LR 0.1 |
| Regularizer | N3, weight 0.01 |
| Init / scale | 0.001 / 20 |
| Số epoch tối đa | 200 |
| Validation / early stopping | Mỗi 3 epoch / 10 |
| Seed | 2022 |
| Student CE/KD weights | 1 / 2 |
| KD temperature | 4 |
| NDKD alpha/beta | 1 / 1 |

Teacher chọn checkpoint theo validation Hits@10, student theo validation MRR. Scheduler giữ `min` trên validation Hits@10 như baseline gốc, dù chỉ số này cần tăng. Nếu đổi scheduler, lưu thành thí nghiệm mới để so với cấu hình cũ.

Khi train student, teacher đóng băng. NDKD neighbor lấy từ training triples và reciprocals; evaluation filters lấy từ cả ba split theo benchmark.

## Policy source-frozen và learned RC

`source_frozen` vẫn khởi tạo policy và lấy mẫu strategy, nhưng loại policy parameters khỏi optimizer. Đây là mặc định của baseline chính, gồm Table 7.

Các cấu hình learned RC trên DB15K/Table 4 dùng Adam riêng LR `1e-5`, binary reward +1/-10. Table 7 cũng có tùy chọn này như một adaptation. Fixed S/V/D/average chọn teacher theo trọng số cố định và không train policy. Giữ tên cấu hình khi điền bảng để phân biệt các cách chạy.

Evaluator teacher gốc lấy minimum rank giữa modality theo ground truth. Chỉ số oracle này cần nhãn để chọn nhánh tốt nhất, nên được ghi riêng với student và teacher ensemble inference.

## Loss NDKD

Code công bố cộng xác suất neighbor bằng SUM; Eq.12 của paper mô tả MEAN. Repo giữ SUM cho các baseline và lần chạy đã lưu. NEKD là KL giữa khối xác suất neighbor/non-neighbor, còn NNKD là KL có điều kiện trên non-neighbor. Xem [cách tách hai term](../tables/table-4-students.md).

## Dữ liệu và đặc trưng

BERT/ViT features được dựng lại. Encoder ảnh dùng input 384 và classifier logits 1.000 chiều; paper mô tả ViT224. Ảnh tải được, quy tắc chọn ảnh và thứ tự RNG của vector thiếu có thể khác tác giả.

Validation DB15K công khai có duplicates và overlap train/test. Các bảng chính giữ dữ liệu gốc; [CleanV1](../tables/db15k-clean-split.md) thay bộ chia để thử độ nhạy và có số liệu riêng.

WN9 có nhiều inverse fact giữa các split và một self-loop không có trong WN18 union đang dùng. Đặc trưng frozen encoder được lấy theo synset ID, nhưng graph teacher/student WN9 phải train mới vì WN18 đã có nhiều held-out fact của WN9.

## Những bảng có adaptation riêng

- [Table 2](../tables/table-2-teacher-selection.md): thiếu cấu hình MulDE của tác giả, nên repo ghi rõ cách dựng `MulDE-adapted` và optimizer từng phần.
- [Table 4](../tables/table-4-teachers.md): Teacher Average lưu cả mean metrics và mean logits. ML-on-val giữ kiến trúc MoSE nhưng dùng final epoch thay vì chọn theo test. RC-on-val có reward và số epoch được chọn riêng cho lần tái lập.
- [Table 7](../tables/table-7-wn9.md): báo cả raw và filtered trên cùng checkpoint chọn theo filtered validation MRR. Source-frozen đạt filtered MRR 0.919634 và raw MRR 0.618763; cách đánh giá paper chưa được xác nhận.

## Khi tổng hợp kết quả

Ghi commit code, nguồn dataset/features, seed, policy optimizer, epoch được chọn và raw/filtered setting. Student lấy chỉ số từ `final_result`, vì log test cuối có thể thuộc epoch khác. Tách kết quả smoke, full và CleanV1; một seed chưa cho biết mức biến thiên giữa các lần chạy.

Cờ hoàn tất bảng cho biết các cấu hình yêu cầu đã chạy đủ. Những khác biệt trên vẫn cần được nêu khi so số liệu với paper.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="matched-configuration"></a>
<a id="method-and-data-differences"></a>
<a id="reporting"></a>
<a id="source-frozen-versus-learned-rc"></a>
<a id="supported-reproduction-protocol"></a>
<a id="supported-scope"></a>
