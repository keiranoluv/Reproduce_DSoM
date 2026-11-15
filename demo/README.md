# DSoM Explorer — Animals & Everyday Objects

Web tương tác dùng **150 thực thể WN9, 5 câu hỏi (1 validation + 4 test),
5 case study**. Tên thực thể, synset IDs, quan hệ và mô tả nguồn giữ tiếng Anh;
câu dẫn, hướng dẫn và giải thích dùng tiếng Việt. Chạy hoàn toàn trong trình
duyệt, không cần checkpoint, GPU, backend, Node.js hoặc API key.

## Chạy tại máy

Từ gốc repository:

```bash
python -m http.server 4173 --directory demo
```

Mở http://localhost:4173. Cần HTTP server để tải JSON, không mở `index.html`
bằng `file://`. Deploy nguyên thư mục `demo/` trên static hosting. Tài nguyên
sử dụng URL tương đối, hỗ trợ GitHub Pages project subpath.

## Tương tác

- **Khám phá:** tìm theo tên tiếng Anh, alias hoặc synset ID; bấm node, lọc quan
  hệ, chuyển tổng quan, kéo nền và zoom. Nhãn đồ thị giữ cỡ chữ đọc được trên mobile.
- **Thử thách:** năm câu về corgi, parrot, pedal, car và building. Mở gợi ý,
  chọn một trong bốn đáp án rồi kiểm tra. Tiến độ lưu tại localStorage với key WN9
  riêng, không dùng lại tiến độ MKG-Y. “Làm lại” xóa tiến độ trong trình duyệt.
- **Khám phá bằng chứng:** sau khi trả lời, mở đồ thị nhỏ. Cạnh train màu xanh;
  đáp án validation/test màu cam nét đứt. Bấm cạnh hoặc bản ghi để xem triple,
  split, dòng nguồn và link tới đúng dòng ở MoSE revision đã pin. Bấm node để
  xem synset và mô tả tiếng Anh. Phần giải thích chỉ rõ có/không có named-inverse
  fact trong train.
- **Xếp hạng minh họa:** bốn lựa chọn có thứ tự và điểm được soạn sẵn trong
  `illustrativeRanking()` ở `app.js`, gồm cả ví dụ đúng và sai. Không dùng
  checkpoint, không chạy inference, không phải xác suất hay xếp hạng trên toàn
  bộ 6.555 thực thể. Các điểm không phải kết quả thực nghiệm DSoM.
- **Góc nhìn:** năm case study có liên kết sang thực thể và câu hỏi.

## Dữ liệu và ý nghĩa phần giải thích

`data/demo.json` là snapshot dùng trực tiếp trên web. Đồ thị chính chỉ chứa
train edges, kèm số dòng nguồn. Triple câu hỏi vắng mặt ở train; bốn câu test
cũng vắng mặt ở validation. Mỗi câu có đúng một known-positive trong bốn lựa
chọn sau khi đối chiếu toàn bộ train/valid/test; vắng mặt trong benchmark không
chứng minh một liên kết sai ngoài đời.

**WN9 có nhiều inverse facts giữa các split.** Ví dụ `corgi _hypernym dog` nằm
ở test, còn `dog _hyponym corgi` nằm ở train. Demo giữ nguyên nguồn và chỉ rõ
thông tin này trong bằng chứng. Câu parrot không có named-inverse trong train;
ngữ cảnh train của câu đó được trình bày là dữ kiện liên quan, không phải một
chuỗi chứng minh đáp án. Không dùng checkpoint graph WN18 cho WN9.

Vùng đồ thị và lời giải thích giúp người xem tra nguồn, đọc quan hệ và hiểu
ngữ cảnh. Chúng **không phải attribution nhân quả** cho một model đã train.
Ẩn/hiện cạnh là thao tác khám phá, không thay đổi điểm minh họa. Ảnh/văn bản
là gợi ý cho người xem; student của workflow reproduction dùng cấu trúc khi
chấm điểm. Gợi ý có thể tiết lộ đáp án, nên quiz không phải protocol đánh giá
model. Source-Frozen RC là biến thể reproduction lịch sử được dẫn, không có
nghĩa policy RL của demo được train.

## Nguồn, ảnh và license

Nguồn WN9 là [MoSE4MKGC](https://github.com/OreOZhao/MoSE4MKGC/tree/192d9ff2882b5be6ef8239a37cbd90c5dd102b5a/src_data/WN9),
pin revision `192d9ff2882b5be6ef8239a37cbd90c5dd102b5a`. Nhãn và gloss tiếng Anh
lấy từ `ent_name.txt` và `ent_dscp.txt`. Copyright WordNet thuộc Princeton
University; xem [WordNet license](WORDNET-LICENSE.txt).

Ảnh chọn đúng member mà manifest WN18 đã dùng, căn chỉnh WN9 `n########` sang
WN18 `########` theo synset ID, không ghép row index giữa datasets. Chỉ tái sử
dụng ảnh/đặc trưng encoder chung, không tái sử dụng checkpoint graph WN18.
Quyền ảnh thuộc các chủ sở hữu gốc; repo không gán giấy phép mở mới cho ảnh.
`imageAvailable` thể hiện có ảnh nguồn; `image` là thumbnail đóng gói. Không có
thumbnail không đồng nghĩa thiếu modality nguồn. Nhóm hiển thị là metadata
biên tập cho UI, không phải nhãn phân loại benchmark.

`data/provenance.json` ghi hash file đầu vào, số dòng nguồn, CRC32 source members
và SHA256 thumbnail. Raw benchmark, ZIP ảnh, features và checkpoints giữ ngoài Git.

## Tạo lại snapshot

Chạy trong môi trường reproduction đã cài Pillow:

```bash
python scripts/export_demo_data.py --help
```

Exporter chỉ tuyển chọn dữ liệu, tạo bản ghi bằng chứng và thumbnail; không
train hoặc inference. Truyền các đường dẫn được liệt kê trong `--help` nếu
raw assets nằm ngoài layout workspace. Không sửa các split gốc.

## Kiểm tra và triển khai

```bash
python -m unittest discover -s tests -p 'test_demo_data.py' -v
```

Kiểm tra không cần dependencies training: scope, references, câu hỏi, bằng
chứng/inverse facts, nguồn hàng dữ liệu, SHA256 thumbnail và bundle dưới 2 MiB.
Nếu có raw WN9, đối chiếu thêm split hashes, triple gốc và known-positive
choices. Trong CI không có raw assets, kiểm tra raw được báo skip tường minh.

Workflow `.github/workflows/demo-pages.yml` kiểm tra và deploy nguyên `demo/`
khi push thay đổi demo lên `main`; PR chỉ chạy kiểm tra. Có thể chạy lại bằng
workflow_dispatch. URL: https://keiranoluv.github.io/Reproduce_DSoM/.
