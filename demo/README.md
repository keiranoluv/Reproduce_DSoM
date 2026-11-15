# DSoM Explorer

Web tương tác dùng dữ liệu MKG-Y thật: **150 thực thể, 182 cạnh train, 10 câu hỏi
(2 validation + 8 test), 5 case study**. Chạy hoàn toàn trong trình duyệt, không
cần model, GPU, backend, Node.js hoặc API key.

## Chạy tại máy

Từ gốc repository:

```bash
python -m http.server 4173 --directory demo
```

Mở http://localhost:4173. Cần HTTP server để tải JSON; không mở `index.html`
bằng `file://`. Có thể deploy nguyên thư mục `demo/` trên static hosting.
Các URL tài nguyên tương đối nên hoạt động cả ở GitHub Pages project subpath.

## Tương tác

- **Khám phá:** tìm thực thể, bấm node hoặc liên kết trong panel, lọc quan hệ,
  chuyển tổng quan, kéo nền, zoom bằng cuộn hoặc các nút `− / ↺ / +`.
- **Thử thách:** chọn một trong bốn đáp án, mở gợi ý cấu trúc/ảnh/văn bản và kiểm tra.
  Câu trả lời lưu tại localStorage; nút “Làm lại” xóa tiến độ trong trình duyệt.
- **Góc nhìn:** năm case study từ dữ liệu, có liên kết sang thực thể và câu hỏi.
- **Phương pháp:** giới thiệu teacher đa phương thức → distillation → student cấu trúc,
  cùng nguồn, provenance và ghi chú Source-Frozen RC.

## Dữ liệu và giới hạn

`data/demo.json` là snapshot dùng trực tiếp trên web. Đồ thị chỉ chứa train edges.
Mọi câu hỏi đều vắng mặt ở train; tám câu test cũng vắng mặt ở validation.
Mỗi câu có đúng một known-positive trong bốn lựa chọn, sau khi đối chiếu toàn bộ
train/valid/test. Quan hệ có thể có thêm đuôi đúng ngoài các lựa chọn; việc một
triple vắng trong benchmark không chứng minh nó sai ngoài đời.

Đây là lát cắt được tuyển chọn để giải thích bài toán, không phải mẫu ngẫu nhiên
hoặc công cụ đánh giá model. Gợi ý là cho người xem và có thể chứa đáp án. Không có
inference hay scores model; kết quả quiz là kết quả người chơi. Case study minh họa
trực giác về modalities, không tuyên bố giải thích cơ chế hoặc lỗi của checkpoint.
Biến thể reproduction được dẫn là **Source-Frozen RC**, không phải learned-policy RL.

Có 119 mô tả nguồn, 149 thực thể có ảnh nguồn và 48 thumbnail được đóng gói (~775 KB).
`imageAvailable` thể hiện ảnh có trong archive; `image` thể hiện thumbnail trong
bundle. Không có thumbnail không đồng nghĩa thiếu modality nguồn. Nhóm thực thể
được suy ra từ vai trò quan hệ để hiển thị, không phải ground-truth entity types.

Ảnh giữ đúng source member được chọn trong manifest training. Quyền ảnh thuộc
chủ sở hữu gốc; repo không gán giấy phép mở mới cho các ảnh. Mô tả là trích đoạn
tiếng Anh từ file mô tả MKG-Y, có liên kết tham khảo định danh DBpedia. Nguồn
dataset: [MMRNS](https://github.com/quqxui/MMRNS#datasets). Hash các file đầu vào,
CRC32 source members và SHA256 thumbnail nằm trong `data/provenance.json`.

## Tạo lại snapshot

Chỉ cần khi thay đổi tuyển chọn dữ liệu. Giữ nguyên raw benchmark và mappings.
Chạy trong môi trường reproduction đã cài Pillow:

```bash
python scripts/export_demo_data.py \
  --source-dir src_data/MKG-Y \
  --mapping data/MKG-Y/ent_id \
  --image-manifest data/MKG-Y/img_features_manifest.json \
  --images /path/to/MKG-Y_img.zip \
  --descriptions /path/to/MKG-Y_ent2descreption.txt
```

Exporter không train/encode; chỉ chọn triples và xuất thumbnail theo manifest.
Lựa chọn dùng seed 2022; cùng đầu vào và phiên bản Pillow tạo bundle lặp lại.
Raw splits, ZIP ảnh, features và checkpoints giữ ngoài Git như trước.

## Kiểm tra và triển khai

```bash
python -m unittest discover -s tests -p 'test_demo_data.py' -v
```

Các kiểm tra không cần dependencies training: integrity snapshot, câu hỏi,
case links, SHA256 thumbnail và dung lượng bundle < 2 MiB. Nếu có raw MKG-Y,
kiểm tra thêm split hashes, source fidelity và known-positive distractors.
Trong CI không có raw assets, kiểm tra đối chiếu raw được báo skip tường minh.

Workflow `.github/workflows/demo-pages.yml` kiểm tra và deploy nguyên `demo/`
khi push thay đổi demo lên `main`; PR chỉ chạy kiểm tra. Lần đầu, bật Settings →
Pages → Source: **GitHub Actions**. Có thể chạy lại bằng workflow_dispatch.
URL dự kiến: https://keiranoluv.github.io/Reproduce_DSoM/.

Browser QA đã thực hiện với Chromium desktop và viewport điện thoại: tìm kiếm,
lọc đồ thị, gợi ý, trả lời đúng/sai, lưu/reset tiến độ và chuyển đủ năm case.
