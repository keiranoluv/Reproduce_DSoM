# Chạy reproduction DSoM

Đây là nguồn lệnh training/evaluation chính thức cho Tables 1/2/3/4/6/7.
Trước khi chạy: hoàn thành [môi trường](ENVIRONMENT.md) và
[dữ liệu/features](DATA.md), kích hoạt venv và đứng ở thư mục gốc repo.
Chỉ model `ComplExMDR` và các cấu hình nêu dưới đây được hỗ trợ;
repo không retrain toàn bộ competitor trong paper.

## Quy ước chạy

`--mode plan` kiểm tra file, hash và ghi kế hoạch; không import Torch hay khởi tạo
GPU trong các launcher dưới đây. `smoke` là kiểm tra thực thi nhỏ trên GPU;
`full` bắt đầu training mới, không tiếp tục smoke và yêu cầu source đã commit,
working tree tracked sạch. Không sửa hyperparameters để biến smoke thành full.
Full có early stopping nên không nhất thiết chạy đủ 200 epochs.

Mỗi ô là một method × dataset, trả về tất cả MR/MRR/Hits được hỗ trợ. `--mode`
chọn cách thực thi; `--methods`, `--components`, `--profile` mới chọn phương pháp.
Một ô PASS không có nghĩa toàn bảng hoàn thành. Các suite chạy riêng không tự
động gộp thành một full-grid report.

Các lệnh full minh họa bật `--wandb`; cần cài `requirements/tracking.txt` và
đăng nhập theo ENVIRONMENT.md nếu dùng online. Có thể bỏ cờ này để chỉ lưu local hoặc đặt
`WANDB_MODE=offline`. Không có hàng đợi chạy ngầm. Các lệnh luôn tạo suite mới;
không xóa artifacts cũ để chạy lại.

<a id="registry"></a>

## Registry cho fresh clone

`references.local.json` là file JSON ở thư mục gốc repo, được Git ignore.
Nó chứa đường dẫn **result.json của full student đã PASS**, không phải checkpoint
hay dữ liệu được tải sẵn. Fresh clone chưa có baseline, teacher hoặc registry.

- `run_reproduction.py --registry references.local.json`: sau full student PASS,
  tự tạo file và đăng ký kết quả. Plan/smoke không đăng ký kết quả.
- `run_table2.py`, `run_table4_components.py`, `run_table4_ensembles.py`,
  `run_table6.py --references references.local.json`: đọc các kết quả đã đăng ký.
- `--teacher-run`: thư mục teacher stage đã hoàn thành từ workflow tương ứng;
  không nhận file `.pth`, registry hoặc thư mục teacher legacy thiếu metadata.

Có thể bắt đầu từ mẫu nếu cần đăng ký thủ công các kết quả cũ đã kiểm chứng:

```bash
# Chỉ tạo nếu chưa tồn tại; mẫu rỗng chưa đủ để chạy bảng phụ thuộc.
if [ ! -e references.local.json ]; then
  cp configs/references.example.json references.local.json
fi
```

Schema: `baselines` dùng key dataset (`MKG-Y`, `MMKB-DB15K`, ...); `table4` dùng
key A4/A5/A9/A10 ở phần Table 4; `table6` có thể chỉ tới kết quả ablation đã có.
Giá trị là đường dẫn `result.json` tương đối từ gốc repo hoặc tuyệt đối. Khi nhập
kết quả cũ, các đường dẫn teacher/features ghi bên trong kết quả cũng phải tồn tại;
không tự đổi manifest cũ để vượt qua kiểm tra integrity.

Khuyến nghị fresh clone: để launcher tự đăng ký baseline mới. Không đưa các
source-frozen baselines vào key learned-RC Table 4. Entry trùng bị từ chối trước
full training; nếu cố ý retrain, dùng registry mới và dùng nhất quán cho các bảng
phụ thuộc. Không sửa registry đồng thời khi launcher đang lập plan. Mỗi suite
snapshot registry và hash artifacts được chọn, không tự chọn checkpoint mới nhất.

<a id="table-1"></a>

## Table 1: baseline trên DB15K, MKG-W, MKG-Y

Dataset CLI: `MMKB-DB15K`, `MKG-W`, `MKG-Y`. Cần structural data, maps, filters
và hai feature arrays của dataset đã được kiểm tra. Không cần reference có sẵn.
Mỗi E2E tạo một full multimodal teacher rồi một structural student; baseline
là `source_frozen` RC, không phải bằng chứng policy đã học RL.

Chọn một dataset:

```bash
DATASET=MKG-Y
CUDA_VISIBLE_DEVICES='' python scripts/run_reproduction.py --dataset "$DATASET" --mode plan
python scripts/run_reproduction.py --dataset "$DATASET" --mode smoke
python scripts/run_reproduction.py --dataset "$DATASET" --mode full --wandb \
  --registry references.local.json
```

Toàn bộ dòng DSoM của Table 1: sau khi prepare và smoke từng dataset, chạy tuần tự
lệnh full dưới đây. Chỉ chạy khi registry chưa có các entry tương ứng:

```bash
for dataset in MMKB-DB15K MKG-W MKG-Y; do
  python scripts/run_reproduction.py --dataset "$dataset" --mode full --wandb \
    --registry references.local.json || break
done
```

Output: `runs/reproduction/DATASET/baseline/{plan,smoke,full}/SUITE/`.
E2E có `teacher/`, `student/`; stage thực thi thành công có `stage_result.json`.
Đọc student `checkpoints/0/result.json`, key `final_result`, được chọn bằng
validation MRR. Teacher chọn theo validation Hits@10; oracle modality ranks
của teacher không phải kết quả prediction cuối.

### Chạy teacher và student riêng

Có thể tách stages cho cả Table 1 và Table 3:

```bash
DATASET=MKG-Y
CUDA_VISIBLE_DEVICES='' python scripts/run_reproduction.py --dataset "$DATASET" --stage teacher --mode plan
python scripts/run_reproduction.py --dataset "$DATASET" --stage teacher --mode smoke
python scripts/run_reproduction.py --dataset "$DATASET" --stage teacher --mode full --wandb
```

Đặt `TEACHER_RUN` thành đường dẫn stage `teacher/` thực tế mà lệnh full in ra.
Cùng full teacher này có thể dùng cho plan, smoke và full student:

```bash
: "${TEACHER_RUN:?Đặt TEACHER_RUN thành thư mục full teacher đã PASS}"
CUDA_VISIBLE_DEVICES='' python scripts/run_reproduction.py --dataset "$DATASET" \
  --stage student --teacher-run "$TEACHER_RUN" --mode plan
python scripts/run_reproduction.py --dataset "$DATASET" --stage student \
  --teacher-run "$TEACHER_RUN" --mode smoke
python scripts/run_reproduction.py --dataset "$DATASET" --stage student \
  --teacher-run "$TEACHER_RUN" --mode full --wandb --registry references.local.json
```

Teacher-only không dùng `--registry`. Student plan/full từ chối smoke teacher;
student smoke được phép dùng smoke/full teacher. Không chạy cả E2E full và full
student đăng ký cùng key, trừ khi chọn registry khác cho thí nghiệm mới.

<a id="table-2"></a>

## Table 2: teacher selection

Datasets như Table 1. Cần baseline full đã đăng ký **cho mỗi dataset được chọn**,
bao gồm full teacher và feature files. Không cần đủ ba baseline nếu chỉ chạy một
dataset. Dòng DSoM được tái sử dụng; các method khác train student mới.

Phương pháp: `best_conf` = ConfTeacher, `best_teacher` = BestTeacher,
`best_strategy` = BestStrategy, `mulde` = MulDE adaptation.

```bash
DATASET=MKG-Y
METHOD=best_conf
CUDA_VISIBLE_DEVICES='' python scripts/run_table2.py --mode plan \
  --datasets "$DATASET" --methods "$METHOD" --references references.local.json
python scripts/run_table2.py --mode smoke --workers 1 \
  --datasets "$DATASET" --methods "$METHOD" --references references.local.json
python scripts/run_table2.py --mode full --workers 1 --wandb \
  --datasets "$DATASET" --methods "$METHOD" --references references.local.json
```

Toàn bảng: bỏ `--datasets "$DATASET" --methods "$METHOD"` ở cả ba lệnh trên.
Mặc định là 12 student mới (4 methods × 3 datasets) và 3 baseline tái sử dụng.
Nhiều giá trị tạo Cartesian product, ví dụ `--datasets MKG-Y MKG-W --methods
best_conf mulde` là 4 student mới. Muốn retrain dòng DSoM, dùng Table 1.

Output: `runs/table2/{plan,smoke,full}/SUITE/`. Sau training đọc `summary.json`
và `table2_results.md`. Kiểm tra `failures=[]`; một ô full PASS có
`completed_new_cells=1`, nhưng `full_benchmark_grid_completed=false` là đúng.
MulDE là adaptation; xem [protocol nghiên cứu](../experiments/TABLE2_REPRODUCTION.md).

<a id="table-3"></a>

## Table 3: baseline trên FB15K-237 và WN18

Cần structural data và hai feature arrays của dataset, không cần baseline cũ.
Chỉ reproduce dòng DSoM, cùng source-frozen profile của Table 1. Chọn một dataset:

```bash
DATASET=WN18
CUDA_VISIBLE_DEVICES='' python scripts/run_reproduction.py --dataset "$DATASET" --mode plan
python scripts/run_reproduction.py --dataset "$DATASET" --mode smoke
python scripts/run_reproduction.py --dataset "$DATASET" --mode full --wandb \
  --registry references.local.json
```

Thay `WN18` bằng `FB15K-237` cho ô còn lại. Sau prepare/smoke cả hai, toàn dòng:

```bash
for dataset in FB15K-237 WN18; do
  python scripts/run_reproduction.py --dataset "$dataset" --mode full --wandb \
    --registry references.local.json || break
done
```

Stages riêng, output và cách chọn `final_result` giống Table 1. Paper Table 3
không công bố MRR; so sánh MR/Hits trong [báo cáo Table 3](../experiments/TABLE3_FB15K237_WN18_SOURCE_FROZEN_RC.md).

<a id="table-4"></a>

## Table 4: chín dòng trên DB15K

Chỉ dataset `MMKB-DB15K`. Thứ tự bắt buộc: một full teacher từ workflow Table 1
→ bốn reference students → hai component students → ba teacher ensembles.
Các RC profiles ở bảng này dùng learned policy Adam 1e-5, khác baseline chính.

### Bốn reference students

Lấy cùng full teacher stage của baseline DB15K từ registry:

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

Teacher này phải được tạo bởi `run_reproduction.py`; checkpoint legacy đơn lẻ
không đáp ứng metadata stage. Chọn profile tương ứng:

| Dòng | Profile | Key `table4` trong registry |
|---|---|---|
| Average + NDKD student | `table4-average` | `A4_average_NDKD` |
| RC + NDKD | `table4-rc-ndkd` | `A5_RC_NDKD` |
| RC + KD | `table4-rc-kd` | `A9_RC_KD` |
| RC + DKD | `table4-rc-dkd` | `A10_RC_DKD` |

```bash
PROFILE=table4-rc-ndkd
CUDA_VISIBLE_DEVICES='' python scripts/run_reproduction.py --dataset MMKB-DB15K \
  --stage student --teacher-run "$DB_TEACHER" --profile "$PROFILE" --mode plan
python scripts/run_reproduction.py --dataset MMKB-DB15K --stage student \
  --teacher-run "$DB_TEACHER" --profile "$PROFILE" --mode smoke
python scripts/run_reproduction.py --dataset MMKB-DB15K --stage student \
  --teacher-run "$DB_TEACHER" --profile "$PROFILE" --mode full --wandb \
  --registry references.local.json
```

Mỗi profile train đúng một student mới. Chạy đủ bốn profile, cùng teacher và
registry, trước khi sang component/ensemble. Muốn tạo đủ references tuần tự,
sau khi smoke từng profile:

```bash
for profile in table4-average table4-rc-ndkd table4-rc-kd table4-rc-dkd; do
  python scripts/run_reproduction.py --dataset MMKB-DB15K --stage student \
    --teacher-run "$DB_TEACHER" --profile "$profile" --mode full --wandb \
    --registry references.local.json || break
done
```

Output: `runs/reproduction/MMKB-DB15K/PROFILE/MODE/SUITE/student/`;
đọc `stage_result.json` và selected `result.json["final_result"]`.

### Hai component students

Launcher hiện tại yêu cầu **cả bốn full references**, kể cả khi chỉ chạy một
component. `nekd` là RC + NEKD; `nnkd` là RC + NNKD.

```bash
COMPONENT=nekd
CUDA_VISIBLE_DEVICES='' python scripts/run_table4_components.py --mode plan \
  --components "$COMPONENT" --references references.local.json
python scripts/run_table4_components.py --mode smoke \
  --components "$COMPONENT" --references references.local.json
python scripts/run_table4_components.py --mode full --workers 1 --wandb \
  --components "$COMPONENT" --references references.local.json
```

Để tạo **một suite chứa cả hai components**, bỏ `--components "$COMPONENT"` ở
cả ba lệnh. Ensemble runner bắt buộc một suite full có cặp này; hai suite đơn
không được tự gộp. Dùng suite đã PASS nếu có, không cần retrain.
Output: `runs/table4_components/MODE/SUITE/{summary.json,table4_student_results.md}`.
Một component PASS có thể có `component_pair_completed=false`.

### Ba teacher ensembles

Cần bốn references và một full component-pair suite PASS. Đặt
`T4_COMPONENT_SUITE` thành thư mục suite thực tế vừa chạy, rồi chọn `avg`, `ml`
hoặc `rc`. `avg` chỉ evaluation; `ml`/`rc` fit fusion head/policy trên validation,
không train thêm graph teacher/student. Avg lưu cả mean metrics và mean logits.

```bash
: "${T4_COMPONENT_SUITE:?Đặt thư mục full component-pair suite đã PASS}"
METHOD=ml
CUDA_VISIBLE_DEVICES='' python scripts/run_table4_ensembles.py --mode plan \
  --methods "$METHOD" --references references.local.json --student-suite "$T4_COMPONENT_SUITE"
python scripts/run_table4_ensembles.py --mode smoke --methods "$METHOD" \
  --references references.local.json --student-suite "$T4_COMPONENT_SUITE"
python scripts/run_table4_ensembles.py --mode full --methods "$METHOD" --wandb \
  --references references.local.json --student-suite "$T4_COMPONENT_SUITE"
```

Toàn bộ ba teacher rows: bỏ `--methods "$METHOD"` ở cả ba lệnh.
Toàn Table 4 gồm đủ ba bước trên, không có một CLI duy nhất chạy cả chín rows.
Output: `runs/table4_ensembles/MODE/SUITE/{summary.json,table4_results.md}`;
`all_nine_rows_have_results=true` chỉ khi full và đủ ba methods/cặp components.
ML/RC-on-val là adaptations; exact author protocol chưa xác nhận.

<a id="table-5"></a>

## Table 5: thống kê dataset

Không có model hoặc `plan/smoke/full` để train Table 5. Sau bước chuẩn bị dữ liệu:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/prepare_dataset.py --dataset WN18 --verify-only
CUDA_VISIBLE_DEVICES='' python scripts/prepare_wn9_from_mose.py --mode verify
```

Lệnh đầu thay được bằng một trong năm dataset chuẩn; lệnh sau dành riêng WN9.
Đếm base relations trước reciprocal augmentation. Thống kê/sản phẩm preparation
ở `data/DATASET/`, xem [DATA.md](DATA.md#structural-preprocessing).
Giữ nguyên khác biệt thứ tự valid/test DB15K giữa raw source và paper.

<a id="table-6"></a>

## Table 6: teacher-modality ablation

Datasets: `MMKB-DB15K`, `MKG-W`, `MKG-Y`. Cần baseline full/teacher đã đăng ký cho
các dataset được chọn. Student luôn structural; method chọn modality của teacher:
`full`, `structural` (S), `visual` (V), `description` (D).

Chạy mới một ô, kể cả nếu registry đã có kết quả tương ứng:

```bash
DATASET=MKG-Y
METHOD=visual
CUDA_VISIBLE_DEVICES='' python scripts/run_table6.py --mode plan --rerun-existing \
  --datasets "$DATASET" --methods "$METHOD" --references references.local.json
python scripts/run_table6.py --mode smoke --rerun-existing \
  --datasets "$DATASET" --methods "$METHOD" --references references.local.json
python scripts/run_table6.py --mode full --rerun-existing --wandb \
  --datasets "$DATASET" --methods "$METHOD" --references references.local.json
```

Bỏ `--datasets "$DATASET" --methods "$METHOD"` để chọn toàn bộ 12 ô ở cả ba
lệnh. Giữ `--rerun-existing` nếu muốn 12 student mới; teacher vẫn tái sử dụng.
Bỏ cờ này để cho phép reuse: fresh registry mặc định reuse 3 baseline và train
9 S/V/D students. Legacy lookup không có `--references` có thể reuse thêm hai
DB15K cells; đó là lý do historical suite từng có 7 jobs mới + 5 reuse.
Optional registry keys `table6/MMKB-DB15K/structural` và
`table6/MMKB-DB15K/description` chỉ tới các kết quả đã kiểm chứng; launcher không
tự đăng ký cell mới. `--methods full` khác với execution `--mode full`.

Output: `runs/table6/MODE/SUITE/{summary.json,table6_results.md}`.
Kiểm tra failures và `full_table6_grid_completed`; subset PASS không phải full grid.

<a id="table-7"></a>

## Table 7: DSoM trên WN9

Cần WN9 đã chuẩn bị từ pinned MoSE và frozen-encoder WN18 features theo
[DATA.md](DATA.md#wn9-from-pinned-mose--verified-wn18-features). Không tái sử dụng
WN18 graph checkpoint; E2E train mới teacher rồi student. Không dùng registry.

```bash
CUDA_VISIBLE_DEVICES='' python scripts/run_table7.py --mode plan
python scripts/run_table7.py --mode smoke
python scripts/run_table7.py --mode full --wandb
```

Đây đã là một ô DSoM × WN9 và cũng là toàn dòng được hỗ trợ. Không có CLI để
train các competitor rows. Default `source_frozen`; optional
`--policy-optimizer Adam --policy-lr 0.00001` là thí nghiệm learned-RC riêng,
phải dùng cùng cấu hình cho smoke/full, không trộn với recorded baseline.

Teacher/student độc lập:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/run_wn9_teacher.py --mode plan
python scripts/run_wn9_teacher.py --mode smoke
python scripts/run_wn9_teacher.py --mode full --wandb
```

Đặt `WN9_TEACHER` thành thư mục full teacher stage thực tế mà lệnh trên in ra:

```bash
: "${WN9_TEACHER:?Đặt thư mục full WN9 teacher stage đã PASS}"
CUDA_VISIBLE_DEVICES='' python scripts/run_wn9_student.py --mode plan --teacher-run "$WN9_TEACHER"
python scripts/run_wn9_student.py --mode smoke --teacher-run "$WN9_TEACHER"
python scripts/run_wn9_student.py --mode full --wandb --teacher-run "$WN9_TEACHER"
```

Student plan/full cần full teacher đã hoàn thành. Với teacher từ E2E, dùng
subfolder `teacher/`; với standalone teacher, dùng stage directory được in ra.
Không thể plan student với một thư mục placeholder chưa tồn tại.

E2E output: `runs/table7/MODE/SUITE/`, có `teacher/`, `student/`, `summary.json`,
`table7_results.md`. Standalone: `runs/table7/{teacher,student}/MODE/SUITE/`.
Student tự reload selected checkpoint, đánh giá đủ RHS/LHS và lưu `evaluation.json`.
Raw và filtered dùng **cùng checkpoint chọn bằng filtered validation MRR**;
không có flag training riêng `--filtered`. Kết quả tại
`results.test.filtered.both` và `results.test.raw.both`.
`full_table7_dsom_row_completed=true` không đồng nghĩa
`exact_author_protocol_reproduced=true`; raw/filtered của paper còn chưa xác minh.

## Đọc kết quả và tổng hợp lại

Chỉ dùng student `result.json["final_result"]` của epoch chọn bằng validation MRR,
không dùng test epoch cuối hay W&B cuối run. Giữ source snapshot, source commit,
data/feature hashes và selected checkpoint. Xem [RESULTS.md](RESULTS.md) để
kiểm tra integrity và xử lý lỗi; [PROTOCOL.md](PROTOCOL.md) để diễn giải khoa học.

Đặt `SUITE` thành thư mục suite đã hoàn thành của đúng bảng rồi chọn **một** lệnh:

```bash
: "${SUITE:?Đặt thư mục suite đã hoàn thành của bảng cần summarize}"
# Chọn một lệnh tương ứng, không chạy cả bốn trên cùng SUITE.
CUDA_VISIBLE_DEVICES='' python scripts/run_table2.py --mode summarize --output "$SUITE"
CUDA_VISIBLE_DEVICES='' python scripts/run_table4_components.py --mode summarize --output "$SUITE"
CUDA_VISIBLE_DEVICES='' python scripts/run_table6.py --mode summarize --output "$SUITE"
CUDA_VISIBLE_DEVICES='' python scripts/run_table7.py --mode summarize --output "$SUITE"
```

Summarize chỉ kiểm tra file rồi ghi lại summary; không inference checkpoint mới.
Table 4 ensembles không có summarize mode; summary được ghi khi chạy xong.
Generic baseline stages có `stage_result.json` và training `result.json`.
