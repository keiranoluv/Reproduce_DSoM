# Chạy thực nghiệm

Cài [môi trường](environment.md), chuẩn bị [dữ liệu và đặc trưng](data.md), rồi chọn bảng trong mục lục dưới đây. Các lệnh chạy từ gốc repo sau khi kích hoạt virtual environment.

| Bảng | Runner chính | Dữ liệu hoặc prerequisite chính | Lệnh chi tiết |
|---|---|---|---|
| Table 1 | `scripts/run_reproduction.py` | DB15K, MKG-W hoặc MKG-Y cùng structural data và features | [Table 1](#table-1) |
| Table 2 | `scripts/run_table2.py` | Baseline full đã đăng ký cho dataset được chọn | [Table 2](#table-2) |
| Table 3 | `scripts/run_reproduction.py` | FB15K-237 hoặc WN18 cùng structural data và features | [Table 3](#table-3) |
| Table 4 | `scripts/run_reproduction.py`, `scripts/run_table4_components.py`, `scripts/run_table4_ensembles.py` | DB15K teacher và các reference students theo thứ tự | [Table 4](#table-4) |
| Table 5 | `scripts/prepare_dataset.py`, `scripts/prepare_wn9_from_mose.py` | Dữ liệu đã tải và chuẩn bị | [Table 5](#table-5) |
| Table 6 | `scripts/run_table6.py` | Baseline full/teacher đã đăng ký cho dataset được chọn | [Table 6](#table-6) |
| Table 7 | `scripts/run_table7.py` | WN9 đã chuẩn bị cùng WN18 frozen features | [Table 7](#table-7) |

## 1. Chạy thử trước khi train đầy đủ

Mỗi runner có ba chế độ:

- `plan`: kiểm tra đầu vào, lưu cấu hình và lệnh sẽ chạy.
- `smoke`: chạy thử với ít dữ liệu để kiểm tra teacher, student và đầu ra.
- `full`: bắt đầu một lần train mới với cấu hình đầy đủ.

`full` không tiếp tục từ optimizer hoặc checkpoint smoke. Trước khi chạy, commit các thay đổi source/config vì launcher lưu và kiểm tra phiên bản code. Mỗi lần gọi tạo một thư mục kết quả mới dưới `runs/`.

Các ví dụ `full` bên dưới có `--wandb`. Cài [tracking và đăng nhập](environment.md) nếu muốn đồng bộ online; dùng `WANDB_MODE=offline` để lưu W&B trên máy, hoặc bỏ cờ để chỉ lưu log thông thường.

<a id="registry"></a>

## 2. Lưu kết quả tham chiếu trong registry

`references.local.json` lưu đường dẫn `result.json` của các full student đã hoàn tất. Những bảng cần dùng lại teacher hoặc baseline sẽ đọc file này. Một bản clone mới chưa có registry hay checkpoint.

Sau một full student thành công, `run_reproduction.py --registry references.local.json` tự tạo file và đăng ký kết quả. Plan/smoke không thêm kết quả. Các runner Table 2, 4 và 6 nhận file qua `--references`.

Nên để launcher đăng ký baseline mới. Nếu cần nhập kết quả cũ đã kiểm tra, có thể bắt đầu từ mẫu:

```bash
# Chỉ tạo nếu chưa tồn tại; mẫu rỗng chưa đủ để chạy bảng phụ thuộc.
if [ ! -e references.local.json ]; then
  cp configs/references.example.json references.local.json
fi
```

Registry dùng key dataset trong `baselines`, key A4/A5/A9/A10 trong `table4`, và có thể thêm kết quả ablation trong `table6`. Giá trị là đường dẫn `result.json` tương đối từ gốc repo hoặc tuyệt đối. Các đường dẫn teacher/features bên trong kết quả cũng phải còn tồn tại.

Launcher từ chối key đã có trước khi train. Khi chạy lại một cấu hình, chọn registry mới và dùng cùng file đó cho các bảng phụ thuộc. Giữ riêng baseline `source_frozen` với các reference learned-RC của Table 4. Mỗi suite lưu bản sao registry và hash của kết quả được chọn, vì vậy tránh sửa registry khi đang lập plan.

`--teacher-run` nhận thư mục teacher stage đã hoàn tất và có metadata, ví dụ thư mục `teacher/` của một full baseline. Nó không nhận trực tiếp file `.pth`.

<a id="table-1"></a>

## 3. Table 1: DSoM trên DB15K, MKG-W, MKG-Y

Chọn `MMKB-DB15K`, `MKG-W` hoặc `MKG-Y`. Mỗi cấu hình train một multimodal teacher rồi distill một structural student. Cần dữ liệu graph và hai mảng đặc trưng của dataset; chưa cần kết quả tham chiếu.

```bash
DATASET=MKG-Y
CUDA_VISIBLE_DEVICES='' python scripts/run_reproduction.py --dataset "$DATASET" --mode plan
python scripts/run_reproduction.py --dataset "$DATASET" --mode smoke
python scripts/run_reproduction.py --dataset "$DATASET" --mode full --wandb \
  --registry references.local.json
```

Để chạy đủ ba dataset, chuẩn bị và smoke từng dataset trước, rồi chạy tuần tự:

```bash
for dataset in MMKB-DB15K MKG-W MKG-Y; do
  python scripts/run_reproduction.py --dataset "$dataset" --mode full --wandb \
    --registry references.local.json || break
done
```

Chỉ dùng vòng lặp khi registry chưa có các key tương ứng. Baseline dùng `source_frozen`: policy được khởi tạo và chọn tổ hợp teacher nhưng không cập nhật trọng số policy.

Kết quả nằm ở `runs/reproduction/DATASET/baseline/{plan,smoke,full}/SUITE/`. Một lần chạy cả hai stage có thư mục `teacher/` và `student/`. Xem `stage_result.json` để kiểm tra trạng thái; lấy chỉ số student từ `checkpoints/0/result.json`, trường `final_result`. Teacher chọn checkpoint theo validation Hits@10, student theo validation MRR.

### Chạy riêng teacher và student

Cách này dùng cho cả Table 1 và Table 3. Chạy teacher trước:

```bash
DATASET=MKG-Y
CUDA_VISIBLE_DEVICES='' python scripts/run_reproduction.py --dataset "$DATASET" --stage teacher --mode plan
python scripts/run_reproduction.py --dataset "$DATASET" --stage teacher --mode smoke
python scripts/run_reproduction.py --dataset "$DATASET" --stage teacher --mode full --wandb
```

Gán `TEACHER_RUN` bằng đường dẫn thư mục full teacher stage mà lệnh vừa in ra, rồi chạy student:

```bash
: "${TEACHER_RUN:?Đặt TEACHER_RUN thành thư mục full teacher đã PASS}"
CUDA_VISIBLE_DEVICES='' python scripts/run_reproduction.py --dataset "$DATASET" \
  --stage student --teacher-run "$TEACHER_RUN" --mode plan
python scripts/run_reproduction.py --dataset "$DATASET" --stage student \
  --teacher-run "$TEACHER_RUN" --mode smoke
python scripts/run_reproduction.py --dataset "$DATASET" --stage student \
  --teacher-run "$TEACHER_RUN" --mode full --wandb --registry references.local.json
```

Student plan/full cần full teacher; student smoke có thể dùng smoke hoặc full teacher. Teacher-only không nhận `--registry`. Nếu đã đăng ký kết quả từ lần chạy cả hai stage, dùng registry khác khi train lại student cùng key.

<a id="table-2"></a>

## 4. Table 2: so sánh cách chọn teacher

Cần baseline full đã đăng ký cho mỗi dataset được chọn. Runner dùng lại teacher và dòng DSoM, rồi train student mới cho các phương pháp còn lại:

- `best_conf`: ConfTeacher.
- `best_teacher`: BestTeacher.
- `best_strategy`: BestStrategy.
- `mulde`: MulDE adaptation.

Ví dụ chạy một phương pháp trên MKG-Y:

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

Bỏ `--datasets "$DATASET" --methods "$METHOD"` ở cả ba lệnh để chạy toàn bảng: 12 student mới và 3 baseline dùng lại. Truyền nhiều giá trị sẽ chạy mọi tổ hợp; chẳng hạn `--datasets MKG-Y MKG-W --methods best_conf mulde` tạo 4 student. Muốn train lại dòng DSoM, dùng lệnh Table 1.

Kết quả nằm ở `runs/table2/{plan,smoke,full}/SUITE/`, gồm `summary.json` và `table2_results.md`. Kiểm tra `failures=[]`. Khi chỉ chạy một ô, `completed_new_cells=1` và `full_benchmark_grid_completed=false` là kết quả bình thường.

<a id="table-3"></a>

## 5. Table 3: DSoM trên FB15K-237 và WN18

Cách chạy giống Table 1, với dữ liệu graph và hai mảng đặc trưng của dataset:

```bash
DATASET=WN18
CUDA_VISIBLE_DEVICES='' python scripts/run_reproduction.py --dataset "$DATASET" --mode plan
python scripts/run_reproduction.py --dataset "$DATASET" --mode smoke
python scripts/run_reproduction.py --dataset "$DATASET" --mode full --wandb \
  --registry references.local.json
```

Thay `WN18` bằng `FB15K-237` để chạy dataset còn lại. Sau khi chuẩn bị và smoke cả hai, có thể chạy đủ dòng DSoM bằng:

```bash
for dataset in FB15K-237 WN18; do
  python scripts/run_reproduction.py --dataset "$dataset" --mode full --wandb \
    --registry references.local.json || break
done
```

Baseline, cách chạy riêng stage và cấu trúc đầu ra giống Table 1.

<a id="table-4"></a>

## 6. Table 4: ablation trên DB15K

Bảng gồm chín dòng và cần chạy theo thứ tự: full teacher DB15K → bốn reference student → hai component student → ba teacher ensemble. Các cấu hình RC ở đây học policy bằng Adam `1e-5`, khác baseline `source_frozen`.

### Bước 1: bốn reference student

Lấy thư mục full teacher DB15K từ baseline đã đăng ký:

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

Teacher cần được tạo bằng `run_reproduction.py` để có đủ metadata stage. Chọn một profile:

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

Mỗi profile train một student mới. Để tạo đủ bốn reference, smoke từng profile rồi chạy:

```bash
for profile in table4-average table4-rc-ndkd table4-rc-kd table4-rc-dkd; do
  python scripts/run_reproduction.py --dataset MMKB-DB15K --stage student \
    --teacher-run "$DB_TEACHER" --profile "$profile" --mode full --wandb \
    --registry references.local.json || break
done
```

Dùng cùng teacher và registry cho cả bốn. Kết quả nằm ở `runs/reproduction/MMKB-DB15K/PROFILE/MODE/SUITE/student/`, với `stage_result.json` và `checkpoints/0/result.json`.

### Bước 2: hai component student

Runner yêu cầu đủ bốn reference ở bước 1, kể cả khi chỉ chọn một component. `nekd` tương ứng RC + NEKD, `nnkd` tương ứng RC + NNKD:

```bash
COMPONENT=nekd
CUDA_VISIBLE_DEVICES='' python scripts/run_table4_components.py --mode plan \
  --components "$COMPONENT" --references references.local.json
python scripts/run_table4_components.py --mode smoke \
  --components "$COMPONENT" --references references.local.json
python scripts/run_table4_components.py --mode full --workers 1 --wandb \
  --components "$COMPONENT" --references references.local.json
```

Để chuẩn bị cho bước teacher ensemble, **bỏ `--components "$COMPONENT"` ở cả ba lệnh** và chạy hai component trong cùng một suite. Runner ensemble cần một full suite có đủ cặp này; hai suite riêng không tự gộp lại.

Kết quả nằm ở `runs/table4_components/MODE/SUITE/`, gồm `summary.json` và `table4_student_results.md`. Nếu chỉ chạy một component thành công, `component_pair_completed` vẫn là `false`.

### Bước 3: ba teacher ensemble

Cần bốn reference và một full component-pair suite đã hoàn tất. Gán `T4_COMPONENT_SUITE` bằng đường dẫn suite ở bước 2, rồi chọn `avg`, `ml` hoặc `rc`:

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

Bỏ `--methods "$METHOD"` để chạy cả ba. `avg` chỉ đánh giá; `ml`/`rc` học phần kết hợp điểm trên validation, dùng lại graph teacher. Avg lưu cả trung bình chỉ số từng modality và kết quả xếp hạng từ trung bình logits.

Kết quả nằm ở `runs/table4_ensembles/MODE/SUITE/`, gồm `summary.json` và `table4_results.md`. Cờ `all_nine_rows_have_results=true` cần đủ các dòng từ ba bước trên. ML/RC trên validation có những lựa chọn adaptation được giải thích trong [kết quả teacher ensemble](../research/tables/table-4-teachers.md).

<a id="table-5"></a>

## 7. Table 5: thống kê dataset

Table 5 lấy số lượng từ dữ liệu đã chuẩn bị, không cần train model:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/prepare_dataset.py --dataset WN18 --verify-only
CUDA_VISIBLE_DEVICES='' python scripts/prepare_wn9_from_mose.py --mode verify
```

Lệnh đầu dùng được với năm dataset chuẩn; lệnh sau dành riêng WN9. Đếm base relations trước khi thêm reciprocal relations. Xem [thống kê và quy tắc xử lý dữ liệu](data.md#structural-preprocessing), gồm khác biệt valid/test DB15K với paper.

<a id="table-6"></a>

## 8. Table 6: chọn modality của teacher

Cần baseline full đã đăng ký cho dataset được chọn: `MMKB-DB15K`, `MKG-W` hoặc `MKG-Y`. Student luôn dùng cấu trúc; `--methods` chọn nguồn teacher: `full`, `structural`, `visual`, `description`.

Ví dụ train mới một ô, kể cả khi đã có kết quả tương ứng:

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

Bỏ `--datasets "$DATASET" --methods "$METHOD"` ở cả ba lệnh để chọn đủ 12 ô. Giữ `--rerun-existing` để train 12 student mới; teacher vẫn được dùng lại.

Nếu bỏ `--rerun-existing`, registry mới mặc định dùng lại 3 baseline và train 9 student S/V/D. Cách tìm kết quả cũ khi không truyền `--references` còn có thể dùng lại hai ô DB15K, nên suite lịch sử có 7 student mới và 5 kết quả dùng lại. Các key tùy chọn `table6/MMKB-DB15K/structural` và `table6/MMKB-DB15K/description` cho phép chỉ định kết quả cũ đã kiểm tra; runner không tự đăng ký ô mới.

`--methods full` là tên phương pháp, còn `--mode full` là chế độ chạy. Kết quả nằm ở `runs/table6/MODE/SUITE/`, gồm `summary.json` và `table6_results.md`. Cờ `full_table6_grid_completed` chỉ bật khi đủ bảng.

<a id="table-7"></a>

## 9. Table 7: DSoM trên WN9

Chuẩn bị WN9 theo [hướng dẫn dữ liệu](data.md#wn9-from-pinned-mose--verified-wn18-features). Runner train mới teacher rồi student WN9 và không cần registry:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/run_table7.py --mode plan
python scripts/run_table7.py --mode smoke
python scripts/run_table7.py --mode full --wandb
```

Đây là toàn bộ dòng DSoM được hỗ trợ trong Table 7. Mặc định dùng `source_frozen`. Muốn thử learned RC, thêm `--policy-optimizer Adam --policy-lr 0.00001` vào cả smoke và full, rồi lưu kết quả như một cấu hình riêng.

### Chạy từng stage

Chạy teacher:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/run_wn9_teacher.py --mode plan
python scripts/run_wn9_teacher.py --mode smoke
python scripts/run_wn9_teacher.py --mode full --wandb
```

Gán `WN9_TEACHER` bằng đường dẫn full teacher stage vừa hoàn tất, rồi chạy student:

```bash
: "${WN9_TEACHER:?Đặt thư mục full WN9 teacher stage đã PASS}"
CUDA_VISIBLE_DEVICES='' python scripts/run_wn9_student.py --mode plan --teacher-run "$WN9_TEACHER"
python scripts/run_wn9_student.py --mode smoke --teacher-run "$WN9_TEACHER"
python scripts/run_wn9_student.py --mode full --wandb --teacher-run "$WN9_TEACHER"
```

Với lần chạy cả hai stage, đường dẫn teacher là thư mục con `teacher/`. Với standalone teacher, dùng thư mục stage mà lệnh in ra. Student plan/full kiểm tra thư mục này và cần full teacher đã hoàn tất.

### Đọc kết quả WN9

Lần chạy cả hai stage lưu ở `runs/table7/MODE/SUITE/`, gồm `teacher/`, `student/`, `summary.json` và `table7_results.md`. Stage riêng lưu ở `runs/table7/{teacher,student}/MODE/SUITE/`.

Sau training, runner nạp lại selected student và đánh giá cả hai chiều RHS/LHS, lưu vào `student/evaluation.json`. Raw và filtered dùng **cùng checkpoint chọn theo filtered validation MRR**. Đọc `results.test.filtered.both` và `results.test.raw.both`; không cần flag training riêng `--filtered`.

Cờ `full_table7_dsom_row_completed` xác nhận dòng DSoM đã chạy đủ. Cách đánh giá Table 7 của tác giả còn chưa rõ; xem [so sánh raw/filtered](../research/tables/table-7-wn9.md).

## 10. Tổng hợp lại một suite đã có

Đặt `SUITE` bằng thư mục đã hoàn tất rồi chọn lệnh đúng với bảng:

```bash
: "${SUITE:?Đặt thư mục suite đã hoàn thành của bảng cần summarize}"
# Chọn một lệnh tương ứng, không chạy cả bốn trên cùng SUITE.
CUDA_VISIBLE_DEVICES='' python scripts/run_table2.py --mode summarize --output "$SUITE"
CUDA_VISIBLE_DEVICES='' python scripts/run_table4_components.py --mode summarize --output "$SUITE"
CUDA_VISIBLE_DEVICES='' python scripts/run_table6.py --mode summarize --output "$SUITE"
CUDA_VISIBLE_DEVICES='' python scripts/run_table7.py --mode summarize --output "$SUITE"
```

`summarize` kiểm tra file và tạo lại summary từ kết quả đã lưu. Nó không đánh giá checkpoint mới. Table 4 ensemble ghi summary khi chạy xong và không có chế độ này. Với baseline, đọc `stage_result.json` cùng training `result.json`.

Giữ commit mã nguồn, input, registry và thư mục kết quả của cùng một suite. Xem [đầu ra và xử lý lỗi](outputs.md) để kiểm tra tiếp.
