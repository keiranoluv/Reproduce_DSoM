# Reproduce các bảng của paper

Bắt đầu từ [ENVIRONMENT.md](../../reproduction/ENVIRONMENT.md), [DATA.md](../../reproduction/DATA.md).
Lệnh giả định gốc repo, venv kích hoạt, features verified và revision có workflow.
**Smoke/full/extract dùng GPU**; plan/summarize kiểm tra CPU/files. Full cần
source commit và tracked tree sạch. Data/`.venv/`/registry được Git ignore.

Workflow reproduce DSoM rows/ablations, không mọi competitor. Giới hạn ở
[PROTOCOL.md](../../reproduction/PROTOCOL.md). Chọn từng ô theo
[hướng dẫn cũ](CELLS.md); lệnh hiện tại dùng RUNNING.md.

## Baselines mới cho Tables 1 và 3

Datasets hỗ trợ: `MMKB-DB15K`, `MKG-W`, `MKG-Y`, `FB15K-237`, `WN18`.
Mỗi E2E `run_reproduction.py` train teacher rồi student mới. Chuẩn bị dataset
trước khi gọi. WN9 dùng Table 7 riêng.

Chọn dataset đã chuẩn bị, ví dụ MKG-Y:

```bash
python scripts/run_reproduction.py --dataset MKG-Y --mode plan
python scripts/run_reproduction.py --dataset MKG-Y --mode smoke
```

Sau smoke PASS, tạo/đăng ký full baseline:

```bash
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_reproduction.py --dataset MKG-Y --mode full --wandb \
  --registry references.local.json
```

Lặp với DB15K/MKG-W cho Table 1, FB15K-237/WN18 cho Table 3. Sau data/smoke
checks, chỉ chạy vòng full grid nếu muốn chủ động train tuần tự:

```bash
for dataset in MMKB-DB15K MKG-W MKG-Y FB15K-237 WN18; do
  WANDB_PROJECT=DSoM-Reproduction \
  python scripts/run_reproduction.py --dataset "$dataset" --mode full --wandb \
    --registry references.local.json || break
done
```

Không chạy vòng nếu registry đã có baseline tương ứng, trừ khi chọn registry
mới: duplicate entry bị từ chối trước train. Không resume optimizer, seed
vẫn 2022. Reference result ở
`runs/reproduction/DATASET/baseline/full/TIMESTAMP/student/checkpoints/0/result.json`.
Dùng `final_result` của selected student. Paper Table 3 chỉ có MR/Hits, không
tự tạo MRR target.

## Teacher/student stages độc lập

Cùng launcher hỗ trợ stages riêng:

```bash
python scripts/run_reproduction.py --dataset MKG-Y --stage teacher --mode smoke
python scripts/run_reproduction.py --dataset MKG-Y --stage student --mode smoke \
  --teacher-run runs/reproduction/MKG-Y/baseline/smoke/TEACHER_SUITE/teacher
```

Dùng teacher stage directory được lệnh đầu in ra, không `.pth`. Independent
full dùng `--mode full` cả hai stages, full teacher và `--registry references.local.json`
trên student. Full/plan student từ chối smoke teacher. E2E full train teacher mới.

## Reference registry tường minh

Full student đầu tiên tạo `references.local.json`; có thể bắt đầu từ
`configs/references.example.json`. Paths tương đối từ gốc repo. Schema:

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

JSON riêng từng máy, Git ignore. `--registry` ghi completed full student;
`--references` đọc references cho tables. Registry không phải checkpoint,
không thay `--teacher-run`. Không sửa đồng thời khi plan. Mỗi suite snapshot
registry/hash và selected result/data hashes; thêm entry không liên quan
không đổi artifacts đã chọn của suite cũ.

Bỏ `--references` vẫn dùng historical lookup. Fresh clone **truyền registry**,
không phụ thuộc layout máy cũ hay tự chọn run mới nhất.

## Table 2: selection và distillation nhiều teachers

Cần full baselines/teachers đã đăng ký cho datasets được chọn, đủ ba cho toàn
bảng. Reuse DSoM rows, train 12 students: ConfTeacher, BestTeacher, BestStrategy,
MulDE-adapted × ba datasets.

```bash
python scripts/run_table2.py --mode plan --references references.local.json
python scripts/run_table2.py --mode smoke --workers 1 --references references.local.json
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table2.py --mode full --workers 1 --wandb \
  --references references.local.json
```

Outputs `runs/table2/full/SUITE/{summary.json,table2_results.md}`.
Thêm `--datasets MKG-Y` cho subset, không full grid. [Protocol](../../experiments/TABLE2_REPRODUCTION.md)
ghi MulDE adaptation choices chưa có từ tác giả. Reference là source-frozen RC.

### Chọn một ô Table 2

`--datasets` chọn cột dataset, `--methods` chọn dòng method; `--mode` chọn
plan/smoke/full/summarize, không method. Một pair train một student mới, báo
MR/MRR/Hits và reuse DSoM baseline row, không retrain full teacher.

| Paper method | CLI `--methods` |
|---|---|
| ConfTeacher | `best_conf` |
| BestTeacher | `best_teacher` |
| BestStrategy | `best_strategy` |
| MulDE (adapted) | `mulde` |
| DSoM reference | Reuse từ `baselines` trong registry; tạo bằng `run_reproduction.py` |

Ví dụ MKG-Y + ConfTeacher chỉ cần full MKG-Y baseline trong registry; tạo/đăng
ký bằng hướng dẫn teacher/student trước đó.

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

Nhiều giá trị tạo Cartesian product: `--datasets MKG-Y MKG-W --methods
best_conf mulde` là bốn students. Bỏ methods chọn cả bốn; bỏ datasets chọn
cả ba. Một ô full PASS có `status=PASS`, `planned_jobs=1`,
`completed_new_cells=1`, `full_benchmark_grid_completed=false`. False đúng
với subset. Suite riêng không tự gộp thành full-grid summary.

Selectors ở launcher khác vẫn áp dụng dependencies và reference checks:

| Workflow | Dataset / method selection |
|---|---|
| Tables 1/3 baseline | `run_reproduction.py --dataset MKG-Y` (or another supported dataset); `--stage teacher/student/e2e` |
| Table 4 reference students | `run_reproduction.py --dataset MMKB-DB15K --stage student --profile ...` |
| Table 4 component students | `run_table4_components.py --components nekd` hoặc `nnkd`; chỉ DB15K |
| Table 4 teacher ensembles | `run_table4_ensembles.py --methods avg`, `ml`, hoặc `rc`; chỉ DB15K |
| Table 6 | `run_table6.py --datasets MKG-Y --methods visual` (methods: `full`, `structural`, `visual`, `description`); có thể reuse, dùng `--rerun-existing` để train student mới |
| Table 7 | Chỉ WN9; `run_wn9_teacher.py` / `run_wn9_student.py` cho stages riêng |

## Table 4: sáu students và ba teacher rows

Dùng **cùng một full DB15K teacher** cho bốn profiles. Lấy stage directory
từ registered baseline, không chép timestamp cũ vào source:

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

Train bốn full reference students tuần tự trên GPU, reuse frozen teacher:

```bash
for profile in table4-average table4-rc-ndkd table4-rc-kd table4-rc-dkd; do
  WANDB_PROJECT=DSoM-Reproduction \
  python scripts/run_reproduction.py --dataset MMKB-DB15K --stage student \
    --teacher-run "$DB_TEACHER" --profile "$profile" --mode full --wandb \
    --registry references.local.json || break
done
```

Registry A4/A5/A9/A10 là average-NDKD, learned-RC NDKD/KL/DKD tương ứng.
Cùng matched hyperparameters; learned policy Adam LR 1e-5, khác Table 1
source-frozen row. Smoke dùng cùng student command, `--mode smoke`, bỏ
`--registry`; được dùng full teacher.

Khi đủ bốn **full** references, chạy component ablations:

```bash
python scripts/run_table4_components.py --mode plan --references references.local.json
python scripts/run_table4_components.py --mode smoke --references references.local.json
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table4_components.py --mode full --workers 1 --wandb \
  --references references.local.json
```

Chọn completed component suite thực tế được in ra:

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

`table4_results.md` cuối gộp chín rows và mean-logit diagnostic. Teacher Avg.
chưa rõ cách hiểu, lưu cả mean metrics/logits. ML/RC-on-val là adaptations,
không bit-exact author implementation. Xem [components](../../experiments/TABLE4_COMPONENT_ABLATIONS.md),
[ensembles](../../experiments/TABLE4_TEACHER_ENSEMBLES.md).

## Table 6: ablation modality

Cần full baselines DB15K/MKG-W/MKG-Y. Fresh registry reuse 3 full cells, train
**9 S/V/D students mới**. Default legacy không registry reuse thêm hai DB15K
ablations nên historical suite có 7 jobs mới. Fresh user không cần A1/A2.
Optional `table6` keys `MMKB-DB15K/structural`, `MMKB-DB15K/description` trỏ
kết quả đã review độc lập.

```bash
python scripts/run_table6.py --mode plan --references references.local.json
python scripts/run_table6.py --mode smoke --references references.local.json
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table6.py --mode full --wandb --references references.local.json
```

Thêm `--rerun-existing` nhất quán plan/smoke/full để train mới 12 cells; vẫn
reuse teachers. Full rows source-frozen. Output
`runs/table6/full/SUITE/table6_results.md`, `summary.json`. Một seed/source-frozen
không chứng minh learned-RC gains. [Kết quả](../../experiments/TABLE6_REPRODUCTION.md).

## Table 7: WN9 E2E

Full DSoM row local PASS CPU review ngày 2026-10-24. Filtered MRR .919634,
raw .618763 cùng checkpoint. [Kết quả/giới hạn](../../reproduction/RESULTS.md#recorded-table-7-full-result).
Lệnh dưới mô tả chạy mới, không bắt buộc rerun suite đã xong.

Chuẩn bị WN9 theo [DATA.md](../../reproduction/DATA.md#wn9-from-pinned-mose--verified-wn18-features).
Wrapper verify parent/source hashes, không dùng baseline registry.

```bash
python scripts/run_table7.py --mode plan
python scripts/run_table7.py --mode smoke
WANDB_PROJECT=DSoM-Reproduction python scripts/run_table7.py --mode full --wandb
```

Train teacher/student WN9 mới, reload student chấm raw/filtered; không reuse
WN18 graph checkpoint. Optional learned RC `--policy-optimizer Adam --policy-lr 0.00001`
là suite/smoke riêng. Default source-frozen. Có stages riêng:

```bash
python scripts/run_wn9_teacher.py --mode smoke
python scripts/run_wn9_student.py --mode smoke \
  --teacher-run runs/table7/teacher/smoke/YOUR_SUITE
```

Output `runs/table7/full/SUITE/{summary.json,table7_results.md}`. Full-row flag
cần evaluation checkpoint đầy đủ, không chỉ exit 0. Hai settings cùng selected
checkpoint filtered validation; exact author protocol còn mở.
[Chi tiết WN9](../../experiments/TABLE7_WN9_REPRODUCTION.md).

## Tổng hợp không GPU

Với suites đã hoàn thành:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/run_table2.py --mode summarize --output runs/table2/full/YOUR_SUITE
CUDA_VISIBLE_DEVICES='' python scripts/run_table4_components.py --mode summarize --output runs/table4_components/full/YOUR_SUITE
CUDA_VISIBLE_DEVICES='' python scripts/run_table6.py --mode summarize --output runs/table6/full/YOUR_SUITE
CUDA_VISIBLE_DEVICES='' python scripts/run_table7.py --mode summarize --output runs/table7/full/YOUR_SUITE
```

Table 4 teacher ensemble ghi summary khi xong, không có summarize mode.
Generic stages lưu `stage_result.json` và training `result.json`. Chọn metrics
theo [RESULTS.md](../../reproduction/RESULTS.md).

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="explicit-reference-registry"></a>
<a id="fresh-baselines-tables-1-and-3"></a>
<a id="independent-teacherstudent-stages"></a>
<a id="reproduce-the-paper-tables"></a>
<a id="select-one-table-2-cell"></a>
<a id="summaries-without-gpu"></a>
<a id="table-2-selection-and-multi-teacher-distillation"></a>
<a id="table-4-six-students-plus-three-teacher-rows"></a>
<a id="table-6-modality-ablations"></a>
