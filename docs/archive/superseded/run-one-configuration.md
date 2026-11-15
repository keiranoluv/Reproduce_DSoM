# Hướng dẫn cũ: chạy từng cấu hình

Trang này lưu cách trình bày lệnh theo từng ô dataset/phương pháp trước khi gộp tài liệu. Khi chạy mới, dùng [hướng dẫn thực nghiệm](../../reproduction/running.md). Một cấu hình trả đầy đủ chỉ số MR/MRR/Hits.

Các ví dụ dưới dùng full mode; trước đó cần plan, smoke và dữ liệu/features đã kiểm tra. Full train mới, yêu cầu source đã commit. `--registry` ghi full student thành công; `--references` đọc kết quả. Khi train lại key đã có, dùng registry khác.

## Table 1

Chọn `MMKB-DB15K`, `MKG-W` hoặc `MKG-Y`:

```bash
DATASET=MKG-Y
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_reproduction.py --dataset "$DATASET" --profile baseline \
  --stage e2e --mode full --wandb --registry references.local.json
```

Nếu đã có full teacher, truyền thư mục stage thực tế vào lệnh student:

```bash
TEACHER_RUN=runs/reproduction/MKG-Y/baseline/full/YOUR_SUITE/teacher
python scripts/run_reproduction.py --dataset MKG-Y --profile baseline \
  --stage student --teacher-run "$TEACHER_RUN" --mode full --wandb \
  --registry references.local.json
```

Teacher riêng dùng `--stage teacher` và bỏ `--registry`. Student plan/full cần full teacher; smoke cũng có thể dùng full teacher. Baseline là `source_frozen`.

## Table 2

Mỗi dataset cần baseline full đã đăng ký. Chọn phương pháp:

| Paper row | `--methods` |
|---|---|
| ConfTeacher | `best_conf` |
| BestTeacher | `best_teacher` |
| BestStrategy | `best_strategy` |
| MulDE adaptation | `mulde` |
| DSoM | Reuse Table 1 baseline; retrain bằng lệnh Table 1 |

```bash
DATASET=MKG-Y
METHOD=best_conf
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table2.py --datasets "$DATASET" --methods "$METHOD" \
  --mode full --workers 1 --wandb --references references.local.json
```

Mỗi tổ hợp train một student, dùng lại teacher và dòng DSoM. Nhiều dataset/method tạo mọi tổ hợp.

## Table 3

Chọn `FB15K-237` hoặc `WN18`:

```bash
DATASET=WN18
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_reproduction.py --dataset "$DATASET" --profile baseline \
  --stage e2e --mode full --wandb --registry references.local.json
```

Cách tách stage giống Table 1. Paper chỉ báo MR/Hits; MRR local được lưu để tham khảo. Repo chưa có lệnh train các competitor row.

## Table 4

Bốn reference student dùng chung một full DB15K teacher:

```bash
DB_TEACHER=runs/reproduction/MMKB-DB15K/baseline/full/YOUR_SUITE/teacher
```

| Row | `--profile` | Registry key |
|---|---|---|
| Average teacher + NDKD student | `table4-average` | `A4_average_NDKD` |
| RC + NDKD student | `table4-rc-ndkd` | `A5_RC_NDKD` |
| RC + vanilla KD student | `table4-rc-kd` | `A9_RC_KD` |
| RC + DKD student | `table4-rc-dkd` | `A10_RC_DKD` |

```bash
PROFILE=table4-rc-ndkd
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_reproduction.py --dataset MMKB-DB15K --stage student \
  --teacher-run "$DB_TEACHER" --profile "$PROFILE" --mode full --wandb \
  --registry references.local.json
```

Component/ensemble runner yêu cầu đủ bốn reference. Hai component là:

| Row | `--components` |
|---|---|
| RC + NEKD | `nekd` |
| RC + NNKD | `nnkd` |

```bash
COMPONENT=nekd
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table4_components.py --components "$COMPONENT" \
  --mode full --workers 1 --wandb --references references.local.json
```

Nếu chỉ chạy một component, `component_pair_completed=false` là bình thường. Ensemble cần một full suite có đủ hai component:

```bash
python scripts/run_table4_components.py --components nekd nnkd --mode full \
  --references references.local.json --wandb
```

Sau đó chọn teacher method:

| Row | `--methods` | Behavior |
|---|---|---|
| Teacher Avg. | `avg` | Chỉ evaluation; báo mean metrics và mean logits |
| Teacher + ML-on-val | `ml` | Trains one adapted validation fusion network |
| Teacher + RC-on-val | `rc` | Trains one adapted validation policy |

```bash
T4_COMPONENT_SUITE=runs/table4_components/full/YOUR_COMPLETED_PAIR_SUITE
METHOD=ml
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table4_ensembles.py --methods "$METHOD" --mode full --wandb \
  --references references.local.json --student-suite "$T4_COMPONENT_SUITE"
```

Các bước này dùng lại graph teacher; `avg` chỉ đánh giá, ML/RC học phần kết hợp điểm trên validation.

## Table 5

Sau khi chuẩn bị dữ liệu, kiểm tra số lượng của standard dataset:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/prepare_dataset.py \
  --dataset WN18 --verify-only
```

WN9 dùng lệnh riêng:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/prepare_wn9_from_mose.py --mode verify
```

Đếm base relation trước reciprocal, giữ valid/test DB15K theo file nguồn.

## Table 6

Dataset là `MMKB-DB15K`, `MKG-W` hoặc `MKG-Y`; cần baseline full tương ứng.

| Row | `--methods` |
|---|---|
| Full DSoM | `full` |
| S teacher | `structural` |
| V teacher | `visual` |
| D teacher | `description` |

```bash
DATASET=MKG-Y
METHOD=visual
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table6.py --datasets "$DATASET" --methods "$METHOD" \
  --rerun-existing --mode full --wandb --references references.local.json
```

`--rerun-existing` train student mới cho ô đã chọn, vẫn dùng lại teacher. Bỏ cờ cho phép dùng lại kết quả. Student luôn dùng cấu trúc; method chọn modality của teacher.

## Table 7

Chạy DSoM trên WN9, train teacher rồi student và đánh giá raw/filtered:

```bash
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table7.py --mode full --wandb --policy-optimizer source_frozen
```

Chạy stage riêng:

```bash
python scripts/run_wn9_teacher.py --mode full --wandb
# Set the actual completed full teacher stage printed above.
WN9_TEACHER=runs/table7/teacher/full/YOUR_SUITE
python scripts/run_wn9_student.py --mode full --wandb \
  --teacher-run "$WN9_TEACHER" --policy-optimizer source_frozen
```

Thay placeholder bằng thư mục full teacher stage đã hoàn tất. WN18 graph checkpoint không dùng được cho WN9. Learned RC `--policy-optimizer Adam --policy-lr 0.00001` được lưu như biến thể riêng.

## Kiểm tra đầu ra

Mỗi lệnh tạo suite riêng. Một ô PASS vẫn có thể chưa đủ bảng; các suite không tự gộp. Dùng chỉ số checkpoint chọn theo validation. Xem [đọc kết quả](../../reproduction/outputs.md) và [bản hướng dẫn toàn bảng cũ](run-all-tables.md) để tra lại bố trí trước đây.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="four-reference-student-rows"></a>
<a id="interpreting-subset-completion"></a>
<a id="run-one-datasetmethod-cell-in-tables-17"></a>
<a id="table-1-dsom--one-multimodal-dataset"></a>
<a id="table-2-one-teacher-selection-method--one-dataset"></a>
<a id="table-3-dsom--one-structural-benchmark"></a>
<a id="table-4-one-ablation-row--db15k"></a>
<a id="table-5-dataset-statistics"></a>
<a id="table-6-one-teacher-modality-ablation--one-dataset"></a>
<a id="three-teacher-ensemble-rows"></a>
<a id="two-component-student-rows"></a>
