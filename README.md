# DSoM - Tái lập thực nghiệm

## Web demo

**[Mở DSoM Explorer](https://keiranoluv.github.io/Reproduce_DSoM/)**

Khám phá 150 thực thể WN9 về động vật và đồ vật, thử trả lời 5 câu hỏi và xem 5 case study. Mỗi câu có đồ thị quan hệ và liên kết tới triple trong dữ liệu gốc.

Chạy trên máy:

```bash
python -m http.server 4173 --directory demo
```

Mở http://localhost:4173. Demo dùng dữ liệu đã xuất sẵn, không cần GPU hoặc checkpoint. Phần xếp hạng được soạn để minh họa cách đọc kết quả. Xem [hướng dẫn demo và nguồn dữ liệu](demo/README.md).

## Tái lập thực nghiệm

Repository phục vụ tái lập bài báo **Dark Side of Modalities: Reinforced Multimodal Distillation for Multimodal Knowledge Graph Reasoning (DSoM)**, ACM Multimedia 2025.

Mã nguồn được kế thừa từ [DSoM của tác giả](https://github.com/OreOZhao/DSoM), bổ sung script chuẩn bị dữ liệu, chạy thực nghiệm, tổng hợp kết quả và web demo.

- [Paper](https://arxiv.org/abs/2507.20738v1) · [Mã nguồn tác giả](https://github.com/OreOZhao/DSoM)
- [Kết quả và so sánh với paper](docs/research/README.md) · [Mục lục tài liệu](docs/README.md)

## 1. Phạm vi

Các script chạy những cấu hình DSoM và ablation dưới đây. Một lần chạy sinh nhiều chỉ số, không cần chạy riêng cho MRR hay từng Hits@K.

| Bảng | Nội dung | Dataset |
|---|---|---|
| Table 1 | Teacher và student DSoM | DB15K, MKG-W, MKG-Y |
| Table 2 | So sánh cách chọn teacher | DB15K, MKG-W, MKG-Y |
| Table 3 | Teacher và student DSoM | FB15K-237, WN18 |
| Table 4 | Thành phần loss và teacher ensemble | DB15K |
| Table 5 | Thống kê dữ liệu sau tiền xử lý | Sáu dataset |
| Table 6 | Chọn từng modality của teacher | DB15K, MKG-W, MKG-Y |
| Table 7 | Teacher và student DSoM | WN9 |

Tên `DB15K` trong paper tương ứng `MMKB-DB15K` trong lệnh chạy. Table 5 chỉ cần chuẩn bị và đếm dữ liệu. Repo chưa có script chạy lại các mô hình đối chứng khác trong Table 1, 3 và 7.

Baseline chính dùng `source_frozen`: policy vẫn chọn tổ hợp teacher nhưng trọng số policy được giữ nguyên khi train student. Table 4 có các cấu hình policy học bằng Adam. Khi so kết quả, xem [cấu hình và khác biệt với paper](docs/research/methodology/protocol.md) của từng bảng.

## 2. Cài đặt

Môi trường đã dùng: Linux x86_64, Python 3.12.3, NVIDIA RTX 5090 32 GiB, PyTorch `2.7.0+cu128`, torchvision `0.22.0+cu128`. Cần GPU NVIDIA và driver tương thích để train và trích xuất đặc trưng ảnh. Repo chưa đo mức VRAM tối thiểu.

```bash
git clone https://github.com/keiranoluv/Reproduce_DSoM.git
cd Reproduce_DSoM

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c requirements/constraints-tested.txt \
  -r requirements/train.txt -r requirements/data-features.txt -r requirements/dev.txt

python scripts/check_environment.py --features --gpu
python -m pip check
```

Các lệnh bên dưới chạy từ thư mục gốc repo, sau khi kích hoạt môi trường. Nếu chỉ chuẩn bị dữ liệu hoặc kiểm tra cấu hình trên CPU, xem [cài môi trường CPU](docs/reproduction/environment.md). Lần trích xuất đặc trưng đầu tiên cần tải trọng số BERT/ViT.

## 3. Chuẩn bị dữ liệu

Tải dữ liệu gốc, chạy tiền xử lý rồi trích xuất đặc trưng văn bản và ảnh cho dataset cần dùng. Dữ liệu, đặc trưng và checkpoint được giữ ngoài Git.

[Hướng dẫn dữ liệu](docs/reproduction/data.md) liệt kê nguồn tải, cấu trúc thư mục và lệnh cho cả sáu dataset. Ví dụ với MKG-Y, sau khi tải dữ liệu MMRNS vào `raw/mmrns/`:

```bash
python scripts/prepare_dataset.py --dataset MKG-Y --source-dir raw/mmrns/MKG-Y

python scripts/build_features.py --dataset MKG-Y --modality text --mode extract \
  --input raw/mmrns/MKG-W-Y-raw_data/MKG-Y_ent2descreption.txt \
  --device cuda --batch-size 16
python scripts/build_features.py --dataset MKG-Y --modality image --mode extract \
  --input raw/mmrns/MKG-W-Y-raw_data/MKG-Y_img.zip --batch-size 8

python scripts/build_features.py --dataset MKG-Y --modality text --mode verify
python scripts/build_features.py --dataset MKG-Y --modality image --mode verify
```

`prepare_dataset.py` tạo mapping và dữ liệu graph; `build_features.py` tạo vector đầu vào cho teacher. Nếu đã có dữ liệu hoặc đặc trưng, dùng chế độ `verify` tương ứng để kiểm tra trước khi train.

## 4. Logging W&B

Mỗi lần chạy đều lưu log và kết quả trên máy. Để theo dõi thêm trên Weights & Biases:

```bash
python -m pip install -c requirements/constraints-tested.txt -r requirements/tracking.txt
wandb login
```

Thêm `--wandb` vào lệnh `full`. Có thể đặt `WANDB_MODE=offline` để lưu W&B trên máy rồi đồng bộ sau. Xem thêm [cấu hình môi trường](docs/reproduction/environment.md).

## 5. Chạy một cấu hình

Ví dụ chạy dòng DSoM của Table 1 trên MKG-Y:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/run_reproduction.py \
  --dataset MKG-Y --mode plan
python scripts/run_reproduction.py --dataset MKG-Y --mode smoke
python scripts/run_reproduction.py --dataset MKG-Y --mode full \
  --registry references.local.json
```

`plan` kiểm tra đầu vào và ghi lệnh sẽ chạy. `smoke` chạy thử với ít dữ liệu. `full` train teacher rồi student từ đầu; nó không tiếp tục từ checkpoint smoke. Trước khi chạy `full`, commit các thay đổi mã nguồn và cấu hình vì launcher kiểm tra phiên bản code.

Sau khi student hoàn tất, launcher tạo hoặc cập nhật `references.local.json` với đường dẫn kết quả. Các bảng cần dùng lại baseline hoặc teacher sẽ đọc file này. Registry được Git ignore; giữ nó cùng thư mục kết quả.

## 6. Chạy từng bảng

[Hướng dẫn chạy thực nghiệm](docs/reproduction/running.md) có lệnh cho một cấu hình, toàn bảng và thứ tự chuẩn bị kết quả phụ thuộc.

| Bảng | Script chính |
|---|---|
| Table 1, 3 | `scripts/run_reproduction.py` |
| Table 2 | `scripts/run_table2.py` |
| Table 4 | `scripts/run_reproduction.py`, `scripts/run_table4_components.py`, `scripts/run_table4_ensembles.py` |
| Table 5 | `scripts/prepare_dataset.py`, `scripts/prepare_wn9_from_mose.py` |
| Table 6 | `scripts/run_table6.py` |
| Table 7 | `scripts/run_table7.py` |

Table 2 và 6 dùng lại teacher từ baseline của dataset đã chọn. Table 4 cần chạy các nhóm student trước teacher ensemble. WN9 cần chuẩn bị dữ liệu riêng và train teacher mới. Chọn bảng cần tái lập rồi làm theo mục tương ứng trong hướng dẫn.

## 7. Đọc kết quả

Mỗi lệnh tạo thư mục mới dưới `runs/`, chứa cấu hình, lệnh, log và kết quả. Với student, lấy chỉ số ở `final_result` trong `checkpoints/0/result.json`: đây là kết quả test của checkpoint được chọn theo validation MRR. Log test ở epoch cuối có thể khác checkpoint này.

Runner theo bảng ghi thêm `summary.json` và bảng Markdown. Xem [vị trí file và cách xử lý lỗi](docs/reproduction/outputs.md), hoặc đọc [các bảng kết quả đã lưu](docs/research/README.md) để đối chiếu với paper mà chưa cần train lại.

## 8. Trích dẫn

```bibtex
@inproceedings{zhao2025DSoM,
  author = {Zhao, Yu and Zhang, Ying and Sui, Xuhui and Zhou, Baohang and Zhu, Haoze and Pan, Jeff Z. and Yuan, Xiaojie},
  title = {Dark Side of Modalities: Reinforced Multimodal Distillation for Multimodal Knowledge Graph Reasoning},
  year = {2025},
  publisher = {Association for Computing Machinery},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  pages = {2506--2515},
  doi = {10.1145/3746027.3754576}
}
```
