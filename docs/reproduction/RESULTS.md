# Kết quả, integrity và xử lý lỗi

## Xác nhận chạy thành công

Process exit 0 chưa đủ kiểm chứng khoa học. Với portable baseline launcher,
stage hoàn tất có `stage_result.json` với `status=PASS`, hashes của result và
checkpoint. Với table suite, xem `summary.json`, `failures` và cờ full-grid/row.
Smoke flags không chứng nhận full benchmark result.

| Workflow | Artifacts cuối |
|---|---|
| Portable baseline/Table 4 reference profile | `teacher/stage_result.json`, `student/stage_result.json`, selected training `result.json` |
| Table 2 | `summary.json`, `table2_results.md` |
| Table 4 component pair | `summary.json`, `table4_student_results.md` |
| Table 4 teacher ensembles | `summary.json`, combined `table4_results.md` |
| Table 6 | `summary.json`, `table6_results.md` |
| Table 7 | `summary.json`, `table7_results.md`, `student/evaluation.json` |

Chỉ workflow Table 7 tự reload và đánh giá độc lập selected student. Các
launcher bảng khác kiểm tra config đã lưu, file hashes, checkpoint tồn tại
và selected curve entries; không khẳng định inference checkpoint mới nếu
chưa thực hiện riêng.

## Chọn đúng metrics

Với student, dùng `result.json["final_result"]` của checkpoint có **validation
MRR lớn nhất**, không dùng `result.json["test"]` hoặc W&B `test/*` cuối cùng.
`best_epoch` chỉ checkpoint đó. `learn.py` chọn teacher theo validation Hits@10
lớn nhất dù một số thông báo cũ ghi "best Valid MRR". Khi modality split,
teacher lấy rank nhỏ nhất giữa modalities: đây là oracle upper-bound diagnostic,
không phải inference có thể triển khai.

Đọc kết quả baseline student đã đăng ký:

```bash
python - <<'PY'
import json
from pathlib import Path
registry = json.loads(Path('references.local.json').read_text())
for dataset, filename in registry['baselines'].items():
    result = json.loads(Path(filename).read_text())
    print(dataset, 'best epoch:', result['best_epoch'], result['final_result'])
PY
```

Metrics là tỷ lệ: MRR `.413` tương đương `41.3%`. Chênh `.01` là **một điểm
phần trăm**, không phải cải thiện tương đối một phần trăm. Paper Table 3 không
báo MRR; so sánh MR và Hits. Khác biệt một seed không chứng minh ý nghĩa
thống kê. WN9 raw/filtered dùng cùng saved student và phải ghi nhãn cả hai.

## Kết quả full Table 7 đã ghi nhận

Người dùng đã hoàn tất full dòng DSoM/WN9 và kiểm tra artifacts trên CPU PASS.
Metrics, giới hạn protocol và bằng chứng integrity được ghi trong
[báo cáo Table 7](../experiments/TABLE7_WN9_REPRODUCTION.md#recorded-full-run--2026-10-24)
và [JSON snapshot](../experiments/table7_results_20261024.json).
Lệnh thực thi nằm ở [RUNNING.md](RUNNING.md).

## Quản lý checkpoint và dữ liệu

Outputs training/extraction được ghi vào thư mục mới hoặc từ chối ghi đè final
files. Full bắt đầu mới, không từ smoke và không tự resume optimizer. Các
launcher có hỗ trợ sẽ hash và kiểm tra lại frozen teacher checkpoints, features,
data/maps cùng reference results đã chọn. Giữ source snapshots/manifests
cùng JSON results.

Load full-object torch checkpoints dùng `weights_only=False` để tương thích
model lưu local. Chỉ load checkpoint tự tạo hoặc tin cậy; không thay bằng
pickle tải từ nguồn bất kỳ. Teacher-only stage không được đăng ký thành
student baseline. Full student phải dùng full teacher với dataset/input hashes khớp.

Registry snapshot artifacts đã chọn; thêm baseline sau đó không sửa suite
plans cũ. Chuyển dataset/checkpoint sang filesystem path khác có thể cần cấu
hình lại vì manifest cũ thường ghi absolute paths. Không viết lại manifest
để vượt verify; tạo preparation/run mới có tài liệu hoặc giữ layout đường dẫn cũ.

## Các lỗi thường gặp

| Lỗi | Cách xử lý |
|---|---|
| `No module named torch/transformers/...` | Kích hoạt đúng venv; chạy `check_environment.py` với nhóm phù hợp và `python -m pip check` |
| Thiếu Python con trong `dsom/` | Dùng Python launcher hiện tại; Bash scripts lịch sử có thể còn đường dẫn đó |
| `Commit tracked changes` / workflow source chưa track | Review và commit source/config trước full; không commit data/runs đã ignore |
| Thiếu baseline/reference hoặc timestamp cũ | Truyền `--references references.local.json`; tạo/đăng ký full students cần thiết trước |
| Trùng registry entry | Chọn registry mới; cố ý không overwrite entry cũ |
| Dataset/features/run output đã tồn tại | Verify hoặc dùng checkout/output riêng; không tự xóa benchmark results |
| Mapping/count/coverage nguồn không khớp | Kiểm tra đúng dataset version, tên files, ent_links/SameAs; không đổi splits hoặc thay H5 để vượt kiểm tra |
| Không tải được Google Drive/Baidu | Tải thủ công từ nguồn gốc; giữ tên files, ghi khả dụng; không âm thầm đổi nguồn |
| Thiếu encoder weights khi offline | Tải đủ Hugging Face/torch caches qua mạng được phép trước extraction |
| Fingerprint feature checkpoint không khớp | Giữ checkpoint cũ; không resume khi raw inputs/mapping/backend đã đổi |
| Lỗi CUDA/driver/VRAM | Chạy GPU check tường minh trên máy; smoke đúng rank và xem logs; đổi batch/rank là đổi protocol |
| Lỗi đăng nhập W&B | Đăng nhập, dùng `WANDB_MODE=offline` hoặc bỏ `--wandb`; local results không cần tracking |
| Full student từ chối smoke teacher | Train full teacher mới trước; full không tiếp tục smoke |
| Selected checkpoint/metrics/hash không khớp | Dừng, kiểm tra plan/logs; không báo kết quả hay ghi đè artifacts nghi vấn |

Stage thất bại lưu logs; orchestration có hỗ trợ sẽ dừng các job phụ thuộc
đang chờ. Portable feature adapter kiểm tra đường dẫn trước khi import backend;
audit không decode mọi ảnh và không chứng minh feature equivalence với paper.

## Phạm vi đã được kiểm chứng

Đã kiểm tra môi trường cài đặt và package metadata công khai của direct pins.
Một venv Python 3.12 độc lập cài CPU torch và đủ requirements groups; `pip check`
và 160 tests PASS. Cả mười encoder modules lịch sử import được, BERT nhỏ với
random initialization chạy CPU forward không tải pretrained weights. Tests
có optimizer steps teacher/student thật ở kích thước nhỏ và mocked orchestration;
không chạy full GPU experiment.

Một local clone chuyển sang đường dẫn có dấu cách, không historical runs và
không `dsom/bin/python` trong repo, đã qua environment checks, feature verify
và baseline plan. Kiểm tra này liên kết MKG-Y features đã chuẩn bị, không tải
hay encode lại. Portable feature adapter và generic baseline entrypoint chưa
được chạy full extraction/training trên máy GPU mới. Historical core runs và
full WN9 workflow của người dùng là bằng chứng riêng; WN9 không phải full run
của portable baseline launcher mới. [Validation record](validation.json)
phân biệt rõ các phạm vi này.

Kiểm tra documentation refactor được lưu riêng trong
[documentation_validation.json](documentation_validation.json); không chứng nhận
GPU reproduction từ máy sạch.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="checkpoint-and-data-handling"></a>
<a id="common-failures"></a>
<a id="correct-metric-selection"></a>
<a id="recognize-success"></a>
<a id="recorded-table-7-full-result"></a>
<a id="results-integrity-and-troubleshooting"></a>
<a id="what-was-validated-for-this-guide"></a>
