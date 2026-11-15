# Hướng dẫn cũ: chạy các bảng

Nội dung này đã được gộp vào [hướng dẫn chạy thực nghiệm](../../reproduction/running.md). Các ví dụ giữ ở đây để tra lại cách dùng runner và registry trong phiên bản tài liệu trước.

## Baseline Table 1 và 3

Mỗi dataset cần graph/features đã kiểm tra, môi trường kích hoạt và source đã commit trước full. Bắt đầu với MKG-Y:

```bash
python scripts/run_reproduction.py --dataset MKG-Y --mode plan
python scripts/run_reproduction.py --dataset MKG-Y --mode smoke
```

```bash
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_reproduction.py --dataset MKG-Y --mode full --wandb \
  --registry references.local.json
```

Để chạy tuần tự năm dataset, sau khi smoke từng dataset và chọn registry chưa có key tương ứng:

```bash
for dataset in MMKB-DB15K MKG-W MKG-Y FB15K-237 WN18; do
  WANDB_PROJECT=DSoM-Reproduction \
  python scripts/run_reproduction.py --dataset "$dataset" --mode full --wandb \
    --registry references.local.json || break
done
```

Mỗi baseline train teacher rồi source-frozen student mới. Kết quả ở `runs/reproduction/DATASET/baseline/full/TIMESTAMP/student/checkpoints/0/result.json`, trường `final_result`.

## Chạy stage riêng

```bash
python scripts/run_reproduction.py --dataset MKG-Y --stage teacher --mode smoke
python scripts/run_reproduction.py --dataset MKG-Y --stage student --mode smoke \
  --teacher-run runs/reproduction/MKG-Y/baseline/smoke/TEACHER_SUITE/teacher
```

Thay placeholder bằng thư mục teacher stage thực tế. Full student cần full teacher, còn smoke có thể dùng smoke/full teacher. E2E full luôn train teacher mới.

## Registry

`--registry` ghi kết quả full student; `--references` đọc kết quả đã có. Mẫu JSON:

```json
{
  "schema_version": 1,
  "baselines": {
    "MKG-Y": "runs/reproduction/MKG-Y/baseline/full/YOUR_SUITE/student/checkpoints/0/result.json"
  },
  "table4": {},
  "table6": {}
}
```

Đường dẫn tương đối tính từ gốc repo. Registry cục bộ được Git ignore; mỗi suite lưu bản sao và hash các file được chọn. Clone mới cần truyền registry thay vì dựa vào cách tìm kết quả ở layout cũ.

## Table 2

Cần baseline cho các dataset được chọn; toàn bảng tạo 12 student và dùng lại 3 dòng DSoM:

```bash
python scripts/run_table2.py --mode plan --references references.local.json
python scripts/run_table2.py --mode smoke --workers 1 --references references.local.json
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table2.py --mode full --workers 1 --wandb \
  --references references.local.json
```

Chọn một phương pháp:

| Paper method | CLI `--methods` |
|---|---|
| ConfTeacher | `best_conf` |
| BestTeacher | `best_teacher` |
| BestStrategy | `best_strategy` |
| MulDE (adapted) | `mulde` |
| DSoM reference | Reuse từ `baselines` trong registry; tạo bằng `run_reproduction.py` |

```bash
# File checks only; inspect plan.json and confirm exactly one job.
CUDA_VISIBLE_DEVICES='' python scripts/run_table2.py --mode plan \
  --datasets MKG-Y --methods best_conf --references references.local.json

# GPU execution check: one new student, one epoch.
python scripts/run_table2.py --mode smoke --workers 1 \
  --datasets MKG-Y --methods best_conf --references references.local.json

# GPU: one fresh full student after review/commit and a passing smoke.
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table2.py --mode full --workers 1 --wandb \
  --datasets MKG-Y --methods best_conf --references references.local.json
```

Một ô thành công có `completed_new_cells=1` và `full_benchmark_grid_completed=false`. Nhiều giá trị dataset/method tạo mọi tổ hợp. Các selector khác:

| Workflow | Dataset / method selection |
|---|---|
| Tables 1/3 baseline | `run_reproduction.py --dataset MKG-Y` (or another supported dataset); `--stage teacher/student/e2e` |
| Table 4 reference students | `run_reproduction.py --dataset MMKB-DB15K --stage student --profile ...` |
| Table 4 component students | `run_table4_components.py --components nekd` hoặc `nnkd`; chỉ DB15K |
| Table 4 teacher ensembles | `run_table4_ensembles.py --methods avg`, `ml`, hoặc `rc`; chỉ DB15K |
| Table 6 | `run_table6.py --datasets MKG-Y --methods visual` (methods: `full`, `structural`, `visual`, `description`); có thể reuse, dùng `--rerun-existing` để train student mới |
| Table 7 | Chỉ WN9; `run_wn9_teacher.py` / `run_wn9_student.py` cho stages riêng |

## Table 4

Lấy full DB15K teacher từ registry:

```bash
DB_TEACHER=$(python - <<'PY'
import json
from pathlib import Path
registry = json.loads(Path('references.local.json').read_text())
result = json.loads(Path(registry['baselines']['MMKB-DB15K']).read_text())
print(Path(result['teacher_checkpoint']).resolve().parents[2])
PY
)
```

Train bốn reference student với cùng teacher:

```bash
for profile in table4-average table4-rc-ndkd table4-rc-kd table4-rc-dkd; do
  WANDB_PROJECT=DSoM-Reproduction \
  python scripts/run_reproduction.py --dataset MMKB-DB15K --stage student \
    --teacher-run "$DB_TEACHER" --profile "$profile" --mode full --wandb \
    --registry references.local.json || break
done
```

A4/A5/A9/A10 lần lượt là average-NDKD và learned-RC NDKD/KL/DKD. Policy learned dùng Adam `1e-5`; các reference này được giữ riêng với baseline source-frozen.

Khi đủ bốn reference, chạy hai component trong một suite:

```bash
python scripts/run_table4_components.py --mode plan --references references.local.json
python scripts/run_table4_components.py --mode smoke --references references.local.json
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table4_components.py --mode full --workers 1 --wandb \
  --references references.local.json
```

Truyền thư mục component suite đã hoàn tất cho ensemble:

```bash
T4_COMPONENT_SUITE=runs/table4_components/full/YOUR_COMPLETED_SUITE

python scripts/run_table4_ensembles.py --mode plan --references references.local.json \
  --student-suite "$T4_COMPONENT_SUITE"
python scripts/run_table4_ensembles.py --mode smoke --references references.local.json \
  --student-suite "$T4_COMPONENT_SUITE"
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table4_ensembles.py --mode full --wandb \
  --references references.local.json --student-suite "$T4_COMPONENT_SUITE"
```

Kết quả cuối gồm chín dòng và mean-logit diagnostic. [Trang teacher ensemble](../../research/tables/table-4-teachers.md) giải thích Teacher Average và các adaptation ML/RC trên validation.

## Table 6

Với registry mới, dùng lại ba full baseline và train chín S/V/D student. Cách tìm kết quả cũ có thể dùng thêm DB15K A1/A2, nên suite lịch sử có bảy job mới.

```bash
python scripts/run_table6.py --mode plan --references references.local.json
python scripts/run_table6.py --mode smoke --references references.local.json
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table6.py --mode full --wandb --references references.local.json
```

Thêm `--rerun-existing` vào cả plan/smoke/full để tạo 12 student mới. Teacher vẫn dùng lại. Kết quả nằm trong `runs/table6/full/SUITE/`.

## Table 7

WN9 dùng dữ liệu riêng, train teacher/student mới và đánh giá cả raw/filtered:

```bash
python scripts/run_table7.py --mode plan
python scripts/run_table7.py --mode smoke
WANDB_PROJECT=DSoM-Reproduction python scripts/run_table7.py --mode full --wandb
```

Các stage cũng có lệnh riêng:

```bash
python scripts/run_wn9_teacher.py --mode smoke
python scripts/run_wn9_student.py --mode smoke \
  --teacher-run runs/table7/teacher/smoke/YOUR_SUITE
```

Kết quả lịch sử tại [Table 7](../../research/tables/table-7-wn9.md#recorded-full-run--2026-10-24) đạt filtered MRR 0.919634 và raw 0.618763 trên cùng checkpoint chọn theo filtered validation. Tùy chọn policy Adam là adaptation riêng.

## Tổng hợp kết quả đã có

Chọn đường dẫn suite đúng bảng:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/run_table2.py --mode summarize --output runs/table2/full/YOUR_SUITE
CUDA_VISIBLE_DEVICES='' python scripts/run_table4_components.py --mode summarize --output runs/table4_components/full/YOUR_SUITE
CUDA_VISIBLE_DEVICES='' python scripts/run_table6.py --mode summarize --output runs/table6/full/YOUR_SUITE
CUDA_VISIBLE_DEVICES='' python scripts/run_table7.py --mode summarize --output runs/table7/full/YOUR_SUITE
```

Table 4 ensemble ghi summary khi hoàn tất, không có summarize mode. Baseline stage có `stage_result.json` và training `result.json`. Cách đọc hiện tại nằm trong [hướng dẫn đầu ra](../../reproduction/outputs.md).

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="explicit-reference-registry"></a>
<a id="fresh-baselines-tables-1-and-3"></a>
<a id="independent-teacherstudent-stages"></a>
<a id="reproduce-the-paper-tables"></a>
<a id="select-one-table-2-cell"></a>
<a id="summaries-without-gpu"></a>
<a id="table-2-selection-and-multi-teacher-distillation"></a>
<a id="table-4-six-students-plus-three-teacher-rows"></a>
<a id="table-6-modality-ablations"></a>
