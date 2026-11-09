# Tài liệu đã được thay thế

Thư mục này lưu các handoff, hướng dẫn setup và bản hướng dẫn trùng lặp đã được
thay thế. Nội dung mô tả trạng thái tại thời điểm viết; lệnh, đường dẫn máy và
nhãn “pending” bên trong **không phải hướng dẫn chạy hiện tại**.
Không xóa kết quả khoa học hoặc benchmark artifacts trong quá trình lưu trữ.

Nguồn hiện tại: [README](../../README.md), [môi trường](../reproduction/ENVIRONMENT.md),
[dữ liệu](../reproduction/DATA.md), [lệnh chạy](../reproduction/RUNNING.md).

## Handoff cũ

- [REPRODUCTION_PROGRESS.md](handoffs/REPRODUCTION_PROGRESS.md): handoff 16/10,
  blockers và trạng thái teacher/student đã được xử lý. Thay bằng RUNNING.md
  và [mục lục kết quả](../experiments/README.md).
- [reproduction_progress.md](handoffs/reproduction_progress.md): snapshot trước khi
  hoàn tất Experiment K. Kết quả nghiên cứu K vẫn giữ trong
  [experiment_k_results.md](../experiments/experiment_k_results.md).
- [db15k_reproduction_progress_20261022.md](handoffs/db15k_reproduction_progress_20261022.md):
  snapshot A9/A10 đang triển khai; thay bằng báo cáo Table 4 và
  [DB15K policy analysis](../experiments/db15k_rc_policy_analysis.md).

## Thiết lập đã có workflow thay thế

- [WN18_DATA_PREPARATION.md](setup/WN18_DATA_PREPARATION.md): giữ audit chuẩn bị
  WN18 lịch sử và coverage; các đường dẫn cục bộ, trạng thái chưa train đã cũ.
  Fresh clone dùng DATA.md; kết quả dùng
  [báo cáo Table 3](../experiments/TABLE3_FB15K237_WN18_SOURCE_FROZEN_RC.md).
- [TABLE1_MKGY_PREPROCESSING.md](setup/TABLE1_MKGY_PREPROCESSING.md): setup cục bộ,
  validation pending đã bị thay thế bởi portable preprocessing trong DATA.md.
- [TABLE1_MKGW_EXP01_SETUP.md](setup/TABLE1_MKGW_EXP01_SETUP.md): các bước sửa CLI và
  setup máy cũ đã hoàn thành; thay bằng DATA.md và RUNNING.md.

## Nội dung trùng lặp

- [TABLE1_MKGY_EXP01_RESULTS.md](superseded/TABLE1_MKGY_EXP01_RESULTS.md): bản riêng
  MKG-Y được bao phủ bởi [báo cáo MKG-Y/MKG-W](../experiments/TABLE1_MKGY_MKGW_EXP01_RESULTS.md).
  Đây là bản superseded, không có nghĩa kết quả bị bác bỏ.
- [TABLES.md](superseded/TABLES.md) và [CELLS.md](superseded/CELLS.md): hai bản
  hướng dẫn trước refactor; đã hợp nhất vào RUNNING.md. Đường dẫn gốc còn file
  redirect để tương thích với liên kết bên ngoài; các anchor cũ được giữ ở đó.

Phần diễn giải của các bản lưu đã được Việt hóa, giữ ý nghĩa và trạng thái
lịch sử; code blocks, logs, tên kỹ thuật và số liệu bảng được giữ nguyên.
Relative Markdown links được điều chỉnh để đọc được sau khi di chuyển;
anchors cũ được giữ để tương thích liên kết. Bản tiếng Anh nguyên gốc còn trong
Git history. Những nghiên cứu RC, ablations và source audits có giá trị khoa học
vẫn nằm trong `docs/experiments/`. Năm protocol Markdown được launcher snapshot
cũng giữ nguyên đường dẫn; không cần sửa code hoặc phá snapshots của runs cũ.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="setup-đã-có-workflow-thay-thế"></a>
