# Cài đặt môi trường

Thực hiện từ thư mục gốc repository trên Linux x86_64. Profile được ghi nhận khi chạy: Python 3.12.3, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, NVIDIA RTX 5090 32 GiB. Chưa xác định VRAM tối thiểu; GPU khác và driver khác cần được kiểm tra trên máy sử dụng.

CUDA được dùng cho training và trích xuất đặc trưng ảnh. CPU có thể dùng để chuẩn bị dữ liệu, kiểm tra, lập plan và trích xuất đặc trưng văn bản; full training trên CPU chưa được xác nhận.

## 1. Cài công cụ hệ thống và tạo virtual environment

Cần Git, Python 3.12 có module `venv`, và `unzip`. Trên Ubuntu 24.04 có thể cài các package hệ thống `git`, `python3.12`, `python3.12-venv`, `unzip` bằng trình quản lý package của hệ điều hành.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2. Cài PyTorch

Profile GPU đã ghi nhận dùng CUDA 12.8 wheels:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Cần NVIDIA driver tương thích với CUDA runtime trong PyTorch. Gói CUDA của PyTorch và phiên bản CUDA hiển thị bởi `nvidia-smi` là hai thông tin khác nhau; cài package Python không tự cập nhật driver.

Nếu chỉ chuẩn bị môi trường CPU:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/lock-cpu-py312.txt
```

`lock-cpu-py312.txt` là lock CPU đầy đủ, gồm cả dependencies tùy chọn. Không dùng file này cho môi trường CUDA vì nó pin CPU wheels.

## 3. Cài dependencies của repository

Trong môi trường CUDA, cài các nhóm phục vụ training, xử lý dữ liệu/features và phát triển:

```bash
python -m pip install -c requirements/constraints-tested.txt \
  -r requirements/train.txt \
  -r requirements/data-features.txt \
  -r requirements/dev.txt
```

| File | Nội dung |
|---|---|
| `requirements/train.txt` | Dependencies khi training; cài PyTorch riêng ở bước trước. |
| `requirements/data-features.txt` | Encoder văn bản/hình ảnh, xử lý ảnh, tải dữ liệu và HDF5. |
| `requirements/dev.txt` | Công cụ phát triển. |
| `requirements/tracking.txt` | W&B tùy chọn. |
| `requirements/constraints-tested.txt` | Phiên bản pin trực tiếp/quan trọng; không phải lock đầy đủ mọi dependency bắc cầu. |

## 4. Kiểm tra cài đặt

```bash
python scripts/check_environment.py --features --dev
python -m pip check
python scripts/check_environment.py --features --gpu
```

Hai lệnh đầu kiểm tra Python và metadata package; lệnh cuối khởi tạo CUDA và chạy một phép tính tensor nhỏ. Nếu không có GPU, bỏ `--gpu`. Kiểm tra này không đo VRAM cần cho full run; hãy đánh giá trên máy đích trước khi chạy workflow đầy đủ.

## W&B tùy chọn

Để ghi tracking online, cài dependency rồi đăng nhập:

```bash
python -m pip install -c requirements/constraints-tested.txt \
  -r requirements/tracking.txt
python scripts/check_environment.py --wandb
wandb login
```

Thêm `--wandb` vào lệnh `full` muốn ghi tracking. Không cần W&B để lưu artifacts cục bộ. Có thể đặt `WANDB_MODE=offline`; cấu hình project/entity có thể truyền bằng biến môi trường tương ứng.

## Trọng số encoder

Lần trích xuất đầu tiên có thể tải pretrained weights BERT/ViT về cache của Hugging Face hoặc Torch Hub. Giữ cache đó để chạy lại; khi làm việc offline, tải weights trước khi bắt đầu extraction. Các weights encoder này tách biệt với checkpoint được tạo trong training.

Sau khi cài xong, chuyển sang [hướng dẫn dữ liệu](DATA.md).

<!-- Stable anchors retained for older links. -->
<a id="create-and-activate"></a>
<a id="environment-setup"></a>
<a id="install-project-dependencies"></a>
<a id="install-the-torch-build-first"></a>
<a id="tracking-and-model-downloads"></a>
<a id="verify-before-gpu-work"></a>
