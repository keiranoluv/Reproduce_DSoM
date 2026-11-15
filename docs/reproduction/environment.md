# Cài môi trường

Hướng dẫn này dùng Linux x86_64 và Python 3.12. Môi trường đã chạy thực nghiệm là Python 3.12.3, PyTorch `2.7.0+cu128`, torchvision `0.22.0+cu128`, NVIDIA RTX 5090 32 GiB.

Training và trích xuất đặc trưng ảnh cần GPU NVIDIA. CPU dùng được cho tiền xử lý, kiểm tra dữ liệu, lập kế hoạch chạy và trích xuất đặc trưng văn bản. Repo chưa kiểm tra full training trên CPU hoặc đo VRAM tối thiểu.

## 1. Tạo virtual environment

Cần Git, Python 3.12 có module `venv` và `unzip`. Trên Ubuntu 24.04, các package tương ứng là `git`, `python3.12`, `python3.12-venv`, `unzip`.

Chạy từ thư mục gốc repo:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Nhớ kích hoạt lại môi trường bằng `source .venv/bin/activate` khi mở terminal mới.

## 2. Cài PyTorch

Bản CUDA dùng trong các lần chạy đã lưu:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Driver NVIDIA trên máy phải hỗ trợ CUDA runtime của bản PyTorch này. Phiên bản CUDA hiển thị trong `nvidia-smi` là thông tin của driver; cài PyTorch không cập nhật driver.

Nếu chỉ làm việc trên CPU, dùng bộ cài sau thay cho bộ cài CUDA và bước dependencies bên dưới:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/lock-cpu-py312.txt
```

`lock-cpu-py312.txt` chứa đầy đủ dependencies, kể cả các nhóm tùy chọn, và ghim PyTorch CPU. Giữ riêng môi trường này với môi trường CUDA.

## 3. Cài thư viện của repo

Sau khi cài PyTorch CUDA, cài các nhóm thư viện cần dùng:

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

File constraints ghim những phiên bản chính đã dùng khi chạy. Các dependency gián tiếp vẫn có thể được pip giải quyết theo môi trường cài đặt.

## 4. Kiểm tra cài đặt

```bash
python scripts/check_environment.py --features --dev
python -m pip check
python scripts/check_environment.py --features --gpu
```

Hai lệnh đầu kiểm tra Python và các package. Lệnh có `--gpu` khởi tạo CUDA và chạy một phép tính tensor nhỏ. Bỏ lệnh đó nếu đang dùng CPU. Kết quả kiểm tra GPU chưa cho biết một cấu hình full có vừa VRAM hay không; dùng bước `smoke` của bảng cần chạy để kiểm tra tiếp.

## 5. Bật W&B nếu cần

Các runner luôn lưu kết quả trên máy. Để ghi thêm lên W&B:

```bash
python -m pip install -c requirements/constraints-tested.txt \
  -r requirements/tracking.txt
python scripts/check_environment.py --wandb
wandb login
```

Thêm `--wandb` vào lệnh `full`. Có thể đặt `WANDB_MODE=offline` để lưu tracking trên máy, hoặc dùng các biến môi trường W&B để chọn project và entity.

## 6. Chuẩn bị trọng số encoder

Lần trích xuất đầu tiên sẽ tải trọng số BERT/ViT về cache của Hugging Face hoặc Torch Hub. Nếu máy chạy offline, tải sẵn trọng số và giữ cache trước khi trích xuất. Đây là trọng số encoder dùng để tạo đặc trưng; checkpoint teacher/student sẽ được tạo khi train.

Tiếp theo: [chuẩn bị dữ liệu](data.md).

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="create-and-activate"></a>
<a id="environment-setup"></a>
<a id="install-project-dependencies"></a>
<a id="install-the-torch-build-first"></a>
<a id="tracking-and-model-downloads"></a>
<a id="verify-before-gpu-work"></a>
