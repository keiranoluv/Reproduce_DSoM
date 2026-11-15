# DSoM — Hướng dẫn chạy lại thực nghiệm

Repository này chứa mã nguồn và hướng dẫn chuẩn bị môi trường, dữ liệu, đặc trưng và chạy các workflow thực nghiệm được hỗ trợ cho bài báo *Dark Side of Modalities: Reinforced Multimodal Distillation for Multimodal Knowledge Graph Reasoning* (ACM Multimedia 2025).

- Repository: [github.com/keiranoluv/Reproduce_DSoM](https://github.com/keiranoluv/Reproduce_DSoM)
- Bài báo: [arXiv:2507.20738](https://arxiv.org/abs/2507.20738v1)
- Mã nguồn của tác giả: [github.com/OreOZhao/DSoM](https://github.com/OreOZhao/DSoM)

Mục lục tài liệu và thứ tự tái lập từ đầu nằm tại [docs/README.md](docs/README.md).

## Demo tương tác

[**Mở DSoM Explorer**](https://keiranoluv.github.io/Reproduce_DSoM/) để khám phá
150 thực thể WN9 về động vật và đồ vật, trả lời 5 câu hỏi từ validation/test và
xem 5 case study. Mỗi câu có đồ thị bằng chứng, nguồn triple và bảng xếp hạng minh họa.
Demo chạy trong trình duyệt, không cần checkpoint hoặc inference model.
[Mã nguồn, dữ liệu và cách chạy tại máy](demo/README.md).

## Môi trường

Các lệnh trong README dành cho Linux x86_64 và Python 3.12. Môi trường đã được ghi nhận khi chạy là Python 3.12.3, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128 và NVIDIA RTX 5090 với 32 GiB VRAM. CUDA training và trích xuất đặc trưng ảnh cần GPU NVIDIA cùng driver tương thích. Repo chưa xác định VRAM tối thiểu; cấu hình khác cần được kiểm tra trên máy sử dụng. CPU dùng được cho tiền xử lý dữ liệu và các bước kiểm tra/plan; khả năng chạy full training trên CPU chưa được xác nhận.

### Cài đặt

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

`--gpu` khởi tạo CUDA và thực hiện một phép tính nhỏ để kiểm tra GPU. Cấu hình cài CPU, tracking W&B tùy chọn, và ghi chú tương thích driver nằm trong [hướng dẫn môi trường](docs/reproduction/ENVIRONMENT.md). Cần tải các trọng số encoder khi trích xuất đặc trưng lần đầu; chúng được lưu trong cache của thư viện tương ứng.

## Dữ liệu

Raw datasets, đặc trưng đã trích xuất và model checkpoints không được lưu trong Git. Tải đúng nguồn dữ liệu, sắp xếp file theo cấu trúc hướng dẫn, rồi chạy lệnh preprocessing và feature extraction tương ứng. [Hướng dẫn dữ liệu](docs/reproduction/DATA.md) liệt kê nguồn tải, tên file, cấu trúc thư mục và lệnh cho từng dataset. Ví dụ bên dưới dùng MKG-Y.

## Chạy nhanh một workflow

Chạy các lệnh từ thư mục gốc repo sau khi kích hoạt virtual environment và chuẩn bị assets theo [hướng dẫn dữ liệu](docs/reproduction/DATA.md).

```bash
# Chuẩn bị dữ liệu cấu trúc
python scripts/prepare_dataset.py --dataset MKG-Y --source-dir raw/mmrns/MKG-Y

# Trích xuất đặc trưng văn bản và hình ảnh
python scripts/build_features.py --dataset MKG-Y --modality text --mode extract \
  --input raw/mmrns/MKG-W-Y-raw_data/MKG-Y_ent2descreption.txt \
  --device cuda --batch-size 16
python scripts/build_features.py --dataset MKG-Y --modality image --mode extract \
  --input raw/mmrns/MKG-W-Y-raw_data/MKG-Y_img.zip --batch-size 8

# Kiểm tra đầu vào và thực hiện lần chạy thử
python scripts/build_features.py --dataset MKG-Y --modality text --mode verify
python scripts/build_features.py --dataset MKG-Y --modality image --mode verify
CUDA_VISIBLE_DEVICES='' python scripts/run_reproduction.py \
  --dataset MKG-Y --mode plan
python scripts/run_reproduction.py --dataset MKG-Y --mode smoke
```

Sau khi smoke hoàn tất, chạy full:

```bash
python scripts/run_reproduction.py --dataset MKG-Y --mode full \
  --registry references.local.json
```

`plan` kiểm tra và lập kế hoạch chạy; `smoke` kiểm tra workflow với lần chạy nhỏ; `full` bắt đầu một lượt training mới. Trước khi full, bảo đảm các thay đổi source cần thiết đã được commit. `references.local.json` là registry cục bộ được tạo/cập nhật bởi lệnh full thành công và được Git ignore; nó dùng để nối các workflow phụ thuộc kết quả đã chạy trước đó. Muốn ghi log lên W&B, cài tracking, đăng nhập rồi thêm `--wandb` vào lệnh full. Cấu hình tracking offline hoặc local được mô tả trong [hướng dẫn môi trường](docs/reproduction/ENVIRONMENT.md).

Lệnh cài tracking tùy chọn:

```bash
python -m pip install -c requirements/constraints-tested.txt -r requirements/tracking.txt
wandb login
```

Mỗi suite plan/smoke/full ghi cấu hình, lệnh, log và artifacts vào thư mục mới dưới `runs/`. Giữ thư mục này cùng registry và ghi lại commit mã nguồn, nguồn dữ liệu, cấu hình môi trường và seed khi lưu hồ sơ chạy. Thư mục `runs/`, dữ liệu tải về và registry cục bộ không được commit theo `.gitignore`.

## Các workflow và entrypoint

Các lệnh đầy đủ, thứ tự prerequisite, chế độ plan/smoke/full và cách chạy một phần hoặc toàn bộ bảng được duy trì tại [hướng dẫn chạy](docs/reproduction/RUNNING.md).

| Workflow trong bài báo | Entry point chính |
|---|---|
| Table 1 và Table 3 | `scripts/run_reproduction.py` |
| Table 2 | `scripts/run_table2.py` |
| Table 4 | `scripts/run_table4_components.py`, `scripts/run_table4_ensembles.py` |
| Table 5 | `scripts/prepare_dataset.py`, `scripts/prepare_wn9_from_mose.py` |
| Table 6 | `scripts/run_table6.py` |
| Table 7 | `scripts/run_table7.py` |

Để xem thứ tự hướng dẫn, tài liệu output và archive, bắt đầu từ [mục lục docs](docs/README.md). Chỉ chạy workflow cần thiết; prerequisites và đầy đủ lệnh cho từng bảng nằm trong [hướng dẫn chạy](docs/reproduction/RUNNING.md).

## Trích dẫn

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
