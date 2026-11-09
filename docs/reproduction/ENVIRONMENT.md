# Cài đặt môi trường

Chạy lệnh từ thư mục gốc repo reproduction. Profile được ghi nhận dùng Linux x86_64 và Python
3.12. Môi trường thực nghiệm là Python 3.12.3, torch 2.7.0+cu128,
torchvision 0.22.0+cu128 và RTX 5090 32 GiB. Chưa xác nhận VRAM tối thiểu
hoặc các dòng GPU khác. Training/trích xuất ViT dùng CUDA; CPU hỗ trợ chuẩn bị
dữ liệu, kiểm tra integrity, plan, tests và trích xuất BERT nếu được chọn.

Cài Git, Python 3.12 kèm module `venv`, và `unzip` bằng trình quản lý package
của hệ điều hành. Ví dụ Ubuntu 24.04 cung cấp `git`, `python3.12`,
`python3.12-venv`, `unzip`; cài package hệ thống có thể cần quyền quản trị.

## Tạo và kích hoạt môi trường

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Không bắt buộc dùng Conda environment tên `dsom`. Các Python launcher được hỗ trợ
gọi `sys.executable` của môi trường hiện tại, kể cả khi repo/venv nằm ở vị trí khác.
Sau khi kích hoạt, gọi `python scripts/NAME.py`. Các Bash script lịch sử có thể
vẫn yêu cầu `./dsom/bin/python`; chúng không phải luồng fresh clone trong hướng dẫn này.

## Cài bản torch trước

Với profile CUDA 12.8 đã dùng trong thực nghiệm:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Lệnh theo [hướng dẫn chính thức PyTorch 2.7.0](https://pytorch.org/get-started/previous-versions/#v270).
Cần NVIDIA driver tương thích; cài Python package không sửa được driver không phù hợp.
Bản CUDA wheel khác với thông tin driver do `nvidia-smi` báo. Workflow không dùng Torchaudio.

Nếu **chỉ kiểm tra CPU**, thay bằng CPU wheels:

```bash
python -m pip install torch==2.7.0 torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

Cài CPU wheels không làm các training entrypoint legacy chạy được trên CPU.
Nên tạo CPU environment riêng để không thay thế môi trường CUDA đang hoạt động.

## Cài dependencies

```bash
python -m pip install -c requirements/constraints-tested.txt \
  -r requirements/train.txt -r requirements/data-features.txt \
  -r requirements/dev.txt
```

Các nhóm dependencies:

| File | Công dụng |
|---|---|
| `train.txt` | NumPy và hiển thị tiến độ; cài torch riêng |
| `data-features.txt` | BERT, ViT, Pillow, tải dữ liệu và HDF5 legacy audits |
| `dev.txt` | pytest và Ruff |
| `tracking.txt` | W&B tùy chọn |
| `constraints-tested.txt` | Pins trực tiếp/quan trọng đã ghi nhận; chưa là full transitive lock |
| `environment-observed.json` | Bằng chứng máy/môi trường, không phải install script |
| `lock-cpu-py312.txt` | Full resolved CPU environment độc lập gồm optional groups; chỉ CPU |

NumPy được pin yêu cầu Python >=3.12. Không thay profile này bằng requirements
MMRNS/Python/PyTorch cũ từ repo upstream khác. Các source repo chỉ cung cấp dữ liệu,
không được cài thành training stack thay thế. Các pin trực tiếp đã được đối chiếu
với package metadata công khai. Một CPU venv Python 3.12 độc lập đã cài thành công
torch/torchvision từ CPU index chính thức và đủ bốn nhóm requirements:
`pip check` cùng 160 tests đều PASS. Dependencies CPU đã resolve được lưu trong
`lock-cpu-py312.txt`. Sau khi cài CPU torch/torchvision ở trên, tái tạo môi trường bằng:

```bash
python -m pip install -r requirements/lock-cpu-py312.txt
```

Không áp dụng CPU lock cho CUDA environment vì file pin các wheels `+cpu`.
Đợt cập nhật tài liệu này chưa cài CUDA sạch/train GPU từ môi trường trống.
Xem [bằng chứng kiểm chứng](validation.json).

## Kiểm tra trước khi chạy GPU

```bash
python scripts/check_environment.py --features --dev
python -m pip check
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests
```

Environment checker chỉ đọc metadata của distribution đã cài, không import
Torch; nó báo lỗi nếu thiếu package hoặc phiên bản khác profile. CPU tests có
import Torch, chặn khởi tạo CUDA trong tests nhạy với GPU và dùng dữ liệu giả lập
nhỏ khi cần thực hiện optimizer steps.

Chỉ chạy lệnh sau khi bạn muốn khởi tạo GPU:

```bash
python scripts/check_environment.py --features --gpu
```

Lệnh này tạo một CUDA tensor nhỏ. Sau đó vẫn cần smoke training/trích xuất
trên phần cứng của bạn. Cùng phiên bản package chưa đủ chứng minh driver tương thích,
VRAM đủ hoặc kết quả số tương đương trên GPU khác.

## Tracking và tải model

```bash
python -m pip install -c requirements/constraints-tested.txt -r requirements/tracking.txt
python scripts/check_environment.py --wandb
wandb login
```

Thêm `--wandb` để bật tracking. Nếu không có cờ này, các launcher được hỗ trợ
đặt `DSOM_WANDB=0`, kể cả khi shell đã bật trước đó. Để tracking offline:

```bash
export WANDB_MODE=offline
export WANDB_PROJECT=DSoM-Reproduction
```

Sau đó dùng lệnh training trong [RUNNING.md](RUNNING.md), thêm `--wandb`
nếu muốn tracking. Lưu kết quả local không cần W&B. Teacher-ensemble runner gọi
W&B API trực tiếp khi được yêu cầu; các launcher khác bật training hooks hiện có.

Thư viện tải pretrained weights BERT/ViT ở lần trích xuất đầu tiên. Giữ cache
Hugging Face và torch hub khi chạy lại; chạy offline cần cache đã có weights.
Các weights này khác với graph teacher/student checkpoints. Bạn tự chọn GPU,
ví dụ `CUDA_VISIBLE_DEVICES=0`; launcher không tự chọn GPU vật lý hoặc xếp job chạy ngầm.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="create-and-activate"></a>
<a id="environment-setup"></a>
<a id="install-project-dependencies"></a>
<a id="install-the-torch-build-first"></a>
<a id="tracking-and-model-downloads"></a>
<a id="verify-before-gpu-work"></a>
