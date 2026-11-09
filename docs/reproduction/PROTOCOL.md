# Protocol reproduction được hỗ trợ

## Phạm vi hỗ trợ

Model training được hỗ trợ là `ComplExMDR`. Teacher tách structural, visual
và description relations; student chỉ dùng structure. Portable profiles,
launcher Tables 2/4/6 và workflow WN9 tạo các tổ hợp đã kiểm tra của model này.
Không có nghĩa mọi tùy chọn CLI kế thừa hoặc mọi model legacy đều hoạt động.

Cụ thể, cách khởi tạo RSME legacy còn tham chiếu global options chưa định nghĩa;
các distiller `soft_kl`/assistant legacy chưa hoàn chỉnh. Workflow không dùng
những nhánh này. Launcher DSoM không train TMR, DySarl hay competitor khác.
Script thí nghiệm lịch sử là hồ sơ nghiên cứu, không phải entrypoint cài đặt thay thế.

## Cấu hình đối chiếu

Baseline mới giữ rank 2000 (vector ComplEx thực tế có `2 * rank` thành phần),
batch 1000, Adagrad .1, N3 .01, init .001, scale 20, tối đa 200 epochs,
validation mỗi 3 epochs, early stopping 10 và seed 2022. Chọn teacher bằng
validation Hits@10, student bằng validation MRR. Scheduler `min` hiện có
đang theo dõi validation Hits@10 theo chiều ngược với mục tiêu maximize; giữ
nguyên để khớp source/baseline, không coi là best practice đã sửa. Muốn đổi
cần tạo một matched experiment mới.

Student có CE/KD weights 1/2, temperature 4, NDKD alpha/beta 1/1. Training
neighbors lấy từ training triples cùng reciprocal triples, không dùng
evaluation filters. Teacher được đóng băng khi train student. Evaluation
filters dùng toàn bộ splits gốc như benchmark.

## Phân biệt source-frozen và learned RC

`source_frozen` loại RC policy khỏi các optimizer parameter groups nhưng vẫn
khởi tạo và sample policy. Đây là protocol mặc định cho main baseline trong
reproduction này, gồm Table 7. Kết quả không chứng minh hiệu quả của learned RL.

Các learned variants DB15K/Table 4 dùng policy Adam riêng ở 1e-5, binary
rewards +1/-10. Tùy chọn Table 7 `--policy-optimizer Adam` là adaptation có
nhãn riêng. Không gộp nó và source-frozen vào một dòng không phân biệt.
Fixed S/V/D/average teacher selection không train policy. Teacher oracle
minimum ranks không thay được student results hoặc direct teacher ensembles.

## Khác biệt phương pháp và dữ liệu

- NDKD công bố nhóm **tổng** xác suất neighbors; Eq.12 của paper mô tả **trung bình**.
  Các default profiles và runs lịch sử giữ SUM. NEKD là KL giữa nhóm
  neighbor/non-neighbor mass, không phải KL giữa từng neighbor riêng lẻ.
- BERT/ViT features được tái dựng. Pipeline dùng image input 384 và classifier
  logits 1000 chiều, trong khi paper mô tả ViT224. Độ phủ ảnh, khả dụng URL cũ
  và thứ tự RNG của missing vectors có thể khác.
- Validation DB15K công khai overlap train/test và có duplicates. Main tables
  giữ nguyên nguồn. CleanV1 là sensitivity dataset riêng, khác kích thước
  training/filters/phân phối validation, không phải ước lượng nhân quả không nhiễu.
- WN9 có nhiều inverse facts giữa splits và giữ một self-loop từ nguồn không
  có trong WN18 union hiện tại. Có thể reuse generic encoder features theo
  synset ID; không reuse WN18 graph checkpoints cho WN9.
- MulDE/Table 2 là adaptation vì cấu hình adaptation DSoM của tác giả không
  được công bố. Chi tiết và optimizer ownership nằm trong báo cáo Table 2.
- Teacher Avg./Table 4 có cách hiểu chưa rõ; lưu cả mean metrics và mean logits.
  ML-on-val theo kiến trúc MoSE nhưng cố định final epoch thay vì chọn trên test.
  RC-on-val độc lập dùng lựa chọn reward/training được mô tả tường minh.
- Table 7 so sánh với các phương pháp báo raw metrics, còn evaluator DSoM dùng
  filtered. Cả hai được báo trên cùng checkpoint chọn bằng filtered validation.
  Chưa xác minh protocol chính xác của tác giả. Full source-frozen đạt filtered
  MRR .919634, raw .618763; gần paper .9213 ở filtered chưa chứng minh tương
  đương protocol. Xem [kết quả đã ghi nhận](RESULTS.md#recorded-table-7-full-result).

## Cách báo cáo

Báo source commit, nguồn dataset/features, seed, policy optimizer, epoch được
chọn và raw/filtered setting. Hoàn tất lưới bảng theo một profile đã ghi nhận
khác với tái lập chính xác con số hoặc protocol tác giả. Không coi cải thiện
một seed là bằng chứng thống kê. Tách smoke, full và sensitivity results.
Đọc student `final_result`, không dùng test log cuối. Tài liệu experiment/audit
giữ bằng chứng chi tiết, không âm thầm sửa kết quả benchmark cũ.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="matched-configuration"></a>
<a id="method-and-data-differences"></a>
<a id="reporting"></a>
<a id="source-frozen-versus-learned-rc"></a>
<a id="supported-reproduction-protocol"></a>
<a id="supported-scope"></a>
