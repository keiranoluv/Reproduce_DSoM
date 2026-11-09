# Chạy từng ô dataset/method trong Tables 1–7

Dùng sau [setup](../../reproduction/ENVIRONMENT.md) và [dữ liệu/features](../../reproduction/DATA.md).
Đứng gốc repo, kích hoạt venv. Lệnh cho người dùng thực thi; đọc/validate
tài liệu không cần training.

Một ô = **method × dataset**, trả mọi MR/MRR/Hits. Không train riêng Hits@1.
Competitor rows Tables 1/3/7 không được hỗ trợ; Table 5 chỉ thống kê.

Với mỗi lệnh dưới, đổi `--mode full` thành `--mode plan` để kiểm tra CPU,
rồi `--mode smoke` kiểm tra nhỏ. Smoke/full dùng GPU, full train mới và cần
committed source/tracked tree sạch. W&B tùy chọn, plan không mở W&B. Không
resume smoke checkpoints.

`--registry` đăng ký full student hoàn thành; `--references` đọc. Chỉ full
mới ghi registry, không overwrite entry: muốn retrain dùng file khác và dùng
nhất quán cho dependencies. [Chi tiết](TABLES.md#explicit-reference-registry).

## Table 1: DSoM × một multimodal dataset

Chọn một trong `MMKB-DB15K`, `MKG-W`, `MKG-Y`:

```bash
DATASET=MKG-Y
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_reproduction.py --dataset "$DATASET" --profile baseline \
  --stage e2e --mode full --wandb --registry references.local.json
```

Tạo teacher rồi structural student mới, đăng ký `baselines/DATASET`. Chỉ dòng
source-frozen DSoM. Nếu đã có full teacher, train student mới bằng **stage
directory** thực tế, không `.pth`:

```bash
TEACHER_RUN=runs/reproduction/MKG-Y/baseline/full/YOUR_SUITE/teacher
python scripts/run_reproduction.py --dataset MKG-Y --profile baseline \
  --stage student --teacher-run "$TEACHER_RUN" --mode full --wandb \
  --registry references.local.json
```

Teacher riêng dùng `--stage teacher`, bỏ `--registry`. Full/plan student cần
full teacher; smoke student được dùng full teacher.

## Table 2: một teacher-selection method × một dataset

Datasets `MMKB-DB15K`, `MKG-W`, `MKG-Y`. Chỉ cần registered full Table 1
baseline/teacher/features của dataset được chọn.

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

Một pair train **một student mới** với teacher cũ, summary thêm reused DSoM
row. Nhiều datasets/methods tạo Cartesian product. `--mode` không chọn method.

## Table 3: DSoM × một structural benchmark

Chọn một trong `FB15K-237`, `WN18`:

```bash
DATASET=WN18
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_reproduction.py --dataset "$DATASET" --profile baseline \
  --stage e2e --mode full --wandb --registry references.local.json
```

Tạo teacher/source-frozen student mới; tách stages như Table 1. Table 3 báo
MR/Hits, local MRR không phải paper MRR target. Không retrain competitors.

## Table 4: một ablation row × DB15K

Table 4 chỉ `MMKB-DB15K`, chín rows dùng ba entrypoints. Learned-RC profiles
khác main source-frozen baseline.

### Bốn reference student rows

Dùng chung full DB15K teacher từ portable workflow; sau Table 1 E2E, đặt
đường dẫn stage thực tế:

```bash
DB_TEACHER=runs/reproduction/MMKB-DB15K/baseline/full/YOUR_SUITE/teacher
```

| Row | `--profile` | Registry key |
|---|---|---|
| Average teacher + NDKD student | `table4-average` | `A4_average_NDKD` |
| RC + NDKD student | `table4-rc-ndkd` | `A5_RC_NDKD` |
| RC + vanilla KD student | `table4-rc-kd` | `A9_RC_KD` |
| RC + DKD student | `table4-rc-dkd` | `A10_RC_DKD` |

Chọn **một** profile rồi chạy:

```bash
PROFILE=table4-rc-ndkd
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_reproduction.py --dataset MMKB-DB15K --stage student \
  --teacher-run "$DB_TEACHER" --profile "$PROFILE" --mode full --wandb \
  --registry references.local.json
```

Train một student mới, reuse full teacher. Chỉ lặp profile khác khi cần row
đó. Component/ensemble runner yêu cầu đủ bốn registered full profiles, kể cả
chọn một method, vì load cả bốn references.

### Hai component student rows

Cần bốn references đã đăng ký, cùng DB15K teacher.

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

Train một student mới, summary gồm bốn reference students;
`component_pair_completed=false` đúng với suite một component.

### Ba teacher ensemble rows

Ensemble cần **một full component-pair suite** đủ NEKD/NNKD. Hai suite đơn
không tự gộp. Tạo prerequisite bằng:

```bash
python scripts/run_table4_components.py --components nekd nnkd --mode full \
  --references references.local.json --wandb
```

Nếu đã có suite đủ cặp, dùng lại không retrain. Chọn một teacher method:

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

Chọn một teacher method, không train graph teacher/student mới. Vẫn báo
teacher diagnostics/student rows. [Chi tiết Table 4](TABLES.md#table-4-six-students-plus-three-teacher-rows).
`avg` không training nhưng evaluation GPU vẫn gọi bằng smoke/full.

## Table 5: thống kê dataset

Table 5 không model/train. Prepare theo [DATA.md](../../reproduction/DATA.md),
sau đó verify standard dataset:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/prepare_dataset.py \
  --dataset WN18 --verify-only
```

WN9 dùng source/feature verification riêng:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/prepare_wn9_from_mose.py --mode verify
```

Counts entities/relations/splits ở data guide và preparation reports. Đếm
base relations trước reciprocals. Giữ khác biệt DB15K valid/test nguồn-paper,
không đổi splits.

## Table 6: một teacher-modality ablation × một dataset

Datasets `MMKB-DB15K`, `MKG-W`, `MKG-Y`; cần registered full Table 1
baseline/teacher của dataset được chọn.

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

`--rerun-existing` đảm bảo **student mới** cho pair, gồm `--methods full`,
vẫn reuse teacher. Không cờ có thể reuse cell, nhất là full baseline. Dùng cờ
nhất quán plan/smoke/full nếu muốn train mới. Student luôn structural, method
chọn teacher modality. Launcher không tự đăng ký cells mới.

## Table 7: DSoM × WN9

WN9 là dataset duy nhất, DSoM là row duy nhất hỗ trợ. E2E đã là một ô:
teacher mới → student mới → raw **và** filtered eval cùng selected checkpoint.

```bash
WANDB_PROJECT=DSoM-Reproduction \
python scripts/run_table7.py --mode full --wandb --policy-optimizer source_frozen
```

Chạy hai stages riêng:

```bash
python scripts/run_wn9_teacher.py --mode full --wandb
# Set the actual completed full teacher stage printed above.
WN9_TEACHER=runs/table7/teacher/full/YOUR_SUITE
python scripts/run_wn9_student.py --mode full --wandb \
  --teacher-run "$WN9_TEACHER" --policy-optimizer source_frozen
```

Standalone student plan/full cần full teacher đã hoàn thành, không thư mục
placeholder. WN18 graph checkpoints không hợp lệ.
`--policy-optimizer Adam --policy-lr 0.00001` là learned-RC experiment riêng.
Raw/filtered là evaluation settings, không training commands khác.
[Kết quả và giới hạn](../../reproduction/RESULTS.md#recorded-table-7-full-result).

## Diễn giải trạng thái hoàn thành subset

Mỗi invocation tạo suite riêng. Một ô PASS không đủ full table; pair/grid
flags có thể đúng khi false. Không tự gộp suites. Giữ plan/results/source
snapshot/checkpoint; dùng validation-selected metrics, không W&B test cuối.
Outputs/summarize ở [TABLES.md](TABLES.md), [RESULTS.md](../../reproduction/RESULTS.md).

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
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
