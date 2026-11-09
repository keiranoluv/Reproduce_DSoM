# DSoM — Tái lập kết quả paper

DSoM (*Dark Side of Modalities: Reinforced Multimodal Distillation for Multimodal
Knowledge Graph Reasoning*, ACM MM 2025) dùng multimodal teacher để distill một
structural student cho bài toán knowledge graph reasoning.
[Paper](https://arxiv.org/abs/2507.20738v1) · [Code tác giả](https://github.com/OreOZhao/DSoM)

Repository này ghi nhận reproduction, nguồn dữ liệu và khác biệt với paper;
cung cấp workflow chuẩn bị dữ liệu, features, teacher → student và evaluation.
Model được hỗ trợ là `ComplExMDR`. Các competitor trong paper không được retrain
bởi các workflow này.

## Trạng thái reproduction

Đã có full results cho dòng DSoM của **Tables 1 và 3**, teacher-selection methods
của **Table 2**, chín dòng của **Table 4**, 12 ô của **Table 6** và dòng DSoM/WN9
của **Table 7**. Table 5 là thống kê dataset, không có training.

Đây là kết quả dưới protocol đã ghi nhận, **chưa khẳng định exact reproduction**.
Baseline chính dùng `source_frozen` RC; learned-RC của Table 4 là cấu hình riêng.
Features BERT/ViT được tái dựng, DB15K có validation contamination, WordNet có
inverse facts, và protocol raw/filtered chính xác của Table 7 chưa được xác minh.

## Kết quả chính

MRR dưới đây là test score của student tại checkpoint chọn bằng validation MRR,
không phải teacher oracle hay test score ở epoch cuối.

| Dataset | Paper MRR | Reproduced MRR |
|---|---:|---:|
| DB15K | 0.426800 | 0.413046 |
| MKG-W | 0.409000 | 0.405840 |
| MKG-Y | 0.400200 | 0.406919 |
| WN9, reproduced filtered | 0.921300 | 0.919634 |

WN9 **raw MRR = 0.618763** trên cùng checkpoint; kết quả filtered gần paper không
chứng minh hai bên dùng cùng protocol. Với Table 3, paper không báo MRR:
FB15K-237 có reproduced MR **116.5569** so với paper **116**;
WN18 có MR **8.9588** so với **7**. Chỉ một seed, không kết luận ý nghĩa thống kê.
Báo cáo và provenance nằm trong [mục lục nghiên cứu](docs/experiments/README.md).

## Cài đặt cơ bản

Chạy ở gốc repository trên Linux với Python 3.12. Ví dụ dùng profile PyTorch
2.7.0/CUDA 12.8; máy ghi nhận trước đây dùng RTX 5090 32 GiB, chưa xác định VRAM
tối thiểu. Cần driver phù hợp trên máy của bạn.

```bash
git clone https://github.com/keiranoluv/Reproduce_DSoM.git DSoM
cd DSoM
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c requirements/constraints-tested.txt \
  -r requirements/train.txt -r requirements/data-features.txt -r requirements/dev.txt
python scripts/check_environment.py --features --dev
python -m pip check
```

[Cài đặt chi tiết](docs/reproduction/ENVIRONMENT.md) có CPU-only setup, W&B và
kiểm tra GPU. W&B là tùy chọn; bỏ `--wandb` nếu không dùng. Môi trường CPU sạch
đã qua tests; chưa kiểm chứng toàn bộ pipeline GPU trên máy mới từ đầu.

## Quick Start: một dataset từ đầu đến kết quả

Ví dụ MKG-Y. Trước tiên tải **triples, descriptions và raw image ZIP** theo
[hướng dẫn dữ liệu](docs/reproduction/DATA.md), đặt đúng layout `raw/mmrns/`.
Clone repo không tự tải các assets này; features/checkpoints không được đóng gói.

```bash
# CPU: chuẩn bị structural data mới. Nếu đã tồn tại, dùng --verify-only.
python scripts/prepare_dataset.py --dataset MKG-Y --source-dir raw/mmrns/MKG-Y

# GPU: tạo hai feature arrays; cần raw assets và pretrained encoder weights.
python scripts/build_features.py --dataset MKG-Y --modality text --mode extract \
  --input raw/mmrns/MKG-W-Y-raw_data/MKG-Y_ent2descreption.txt --device cuda --batch-size 16
python scripts/build_features.py --dataset MKG-Y --modality image --mode extract \
  --input raw/mmrns/MKG-W-Y-raw_data/MKG-Y_img.zip --batch-size 8

# Kiểm tra features và plan trước khi training.
python scripts/build_features.py --dataset MKG-Y --modality text --mode verify
python scripts/build_features.py --dataset MKG-Y --modality image --mode verify
CUDA_VISIBLE_DEVICES='' python scripts/run_reproduction.py --dataset MKG-Y --mode plan
python scripts/run_reproduction.py --dataset MKG-Y --mode smoke
```

Sau smoke PASS, review/commit mọi thay đổi source trước khi chạy full:

```bash
python scripts/run_reproduction.py --dataset MKG-Y --mode full \
  --registry references.local.json
```

Full train mới teacher → student và thực hiện validation/test evaluation.
Mở thư mục suite được in ra dưới `runs/reproduction/MKG-Y/baseline/full/`:
kiểm tra `teacher/stage_result.json`, `student/stage_result.json` đều PASS;
đọc `student/checkpoints/0/result.json`, key `final_result`.
Workflow này đánh giá trong training; chỉ workflow Table 7 tự reload checkpoint
để evaluation độc lập sau training.

`references.local.json` được tạo khi full student PASS, lưu đường dẫn kết quả
để các bảng khác sử dụng; file này không commit lên Git. `--registry` ghi đăng ký,
`--references` đọc đăng ký. [Giải thích và prerequisite](docs/reproduction/RUNNING.md#registry).

## Chạy từng Table

Nguồn lệnh chính thức là [RUNNING.md](docs/reproduction/RUNNING.md): mỗi phần có
lệnh **plan → smoke → full**, chọn một ô hoặc toàn bảng, output và cách đọc score.

- [Table 1](docs/reproduction/RUNNING.md#table-1): `run_reproduction.py --dataset MKG-Y`;
  thay dataset bằng `MMKB-DB15K` hoặc `MKG-W` cho hai ô còn lại.
- [Table 2](docs/reproduction/RUNNING.md#table-2): `run_table2.py --datasets MKG-Y --methods best_conf`;
  cần baseline full của dataset được chọn.
- [Table 3](docs/reproduction/RUNNING.md#table-3): `run_reproduction.py --dataset WN18`
  hoặc `--dataset FB15K-237`.
- [Table 4](docs/reproduction/RUNNING.md#table-4): reference students → components
  → ensembles; dùng chung DB15K teacher và đủ references trước bước phụ thuộc.
- [Table 6](docs/reproduction/RUNNING.md#table-6): `run_table6.py --datasets MKG-Y --methods visual`;
  chọn `--rerun-existing` nếu muốn train lại ô đã có.
- [Table 7](docs/reproduction/RUNNING.md#table-7): `run_table7.py`; WN9 teacher/student
  mới, tự báo cả raw và filtered.

Các tên script ở trên nằm trong `scripts/`; đây là chỉ dẫn chọn workflow,
không phải chuỗi lệnh cần chạy tất cả. Lệnh đầy đủ, registry và thứ tự prerequisites
chỉ được duy trì tại RUNNING.md.

## Tài liệu chuyên sâu và citation

[Đọc kết quả và xử lý lỗi](docs/reproduction/RESULTS.md) ·
[Protocol và giới hạn](docs/reproduction/PROTOCOL.md) ·
[Bằng chứng nghiên cứu](docs/experiments/README.md).
Hướng dẫn setup/handoff đã bị thay thế được lưu ở
[docs/deprecated](docs/deprecated/README.md), không dùng cho fresh clone.

Nếu sử dụng paper hoặc code tác giả, vui lòng trích dẫn:

```bibtex
@inproceedings{zhao2025DSoM,
author = {Zhao, Yu and Zhang, Ying and Sui, Xuhui and Zhou, Baohang and Zhu, Haoze and Pan, Jeff Z. and Yuan, Xiaojie},
title = {Dark Side of Modalities: Reinforced Multimodal Distillation for Multimodal Knowledge Graph Reasoning},
year = {2025},
isbn = {9798400720352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746027.3754576},
doi = {10.1145/3746027.3754576},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {2506–2515},
numpages = {10},
keywords = {knowledge distillation, multimodal knowledge graphs, multimodal knowledge reasoning, multimodal learning},
location = {Dublin, Ireland},
series = {MM '25}
}
```

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="dsom--reproduce-paper"></a>
