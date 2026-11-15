# DSoM Explorer

**[Mở web demo](https://keiranoluv.github.io/Reproduce_DSoM/)**

Demo giới thiệu 150 thực thể WN9 về động vật và đồ vật qua đồ thị, 5 câu hỏi và 5 case study. Tên thực thể, synset ID, quan hệ và mô tả nguồn giữ tiếng Anh; phần hướng dẫn dùng tiếng Việt.

## Chạy trên máy

Từ gốc repo:

```bash
python -m http.server 4173 --directory demo
```

Mở http://localhost:4173. Demo chạy bằng JSON và ảnh đã xuất sẵn, không cần checkpoint, GPU, backend, Node.js hoặc API key. Dùng HTTP server để trình duyệt tải được JSON.

Có thể deploy nguyên thư mục `demo/` lên static hosting. Các đường dẫn tương đối hỗ trợ cả GitHub Pages project subpath.

## Cách dùng

- **Khám phá:** tìm theo tên tiếng Anh, alias hoặc synset ID. Bấm node để xem thông tin, lọc quan hệ, kéo nền và zoom đồ thị.
- **Thử thách:** trả lời năm câu về corgi, parrot, pedal, car và building. Mỗi câu có gợi ý, bốn lựa chọn và nút kiểm tra. Tiến độ lưu trong localStorage bằng key riêng cho WN9; “Làm lại” xóa tiến độ trên trình duyệt.
- **Xem bằng chứng:** sau khi trả lời, mở đồ thị nhỏ. Cạnh train màu xanh; đáp án validation/test màu cam nét đứt. Bấm cạnh hoặc bản ghi để xem triple, split, dòng nguồn và liên kết tới đúng dòng MoSE.
- **Đọc xếp hạng:** bốn lựa chọn có điểm và thứ tự soạn sẵn trong `illustrativeRanking()` ở `app.js`, gồm ví dụ đúng và sai. Phần này minh họa cách đọc bảng xếp hạng; điểm không phải xác suất hoặc kết quả inference DSoM trên 6.555 entity.
- **Góc nhìn:** năm case study dẫn sang các thực thể và câu hỏi liên quan.

## Đọc đồ thị và lời giải thích

`data/demo.json` là dữ liệu dùng trực tiếp trên web. Đồ thị chính chỉ có train edges. Năm câu hỏi gồm một validation và bốn test triple, đều vắng ở train; bốn câu test cũng vắng ở validation. Mỗi câu có đúng một known-positive trong bốn lựa chọn sau khi đối chiếu cả ba split. Một lựa chọn vắng trong benchmark vẫn có thể đúng ngoài thực tế.

WN9 có nhiều inverse facts giữa các split. Chẳng hạn `corgi _hypernym dog` ở test có chiều đảo `dog _hyponym corgi` trong train; demo chỉ rõ điều này. Câu parrot không có named-inverse trong train, nên đồ thị của nó cung cấp ngữ cảnh liên quan thay vì một chuỗi suy ra đáp án.

Đồ thị giúp tra nguồn và đọc quan hệ. Ẩn/hiện cạnh không đổi điểm minh họa, và các cạnh không giải thích quan hệ nhân quả trong quyết định của model. Ảnh/văn bản là gợi ý cho người xem, có thể tiết lộ đáp án; student của phần tái lập dùng cấu trúc khi chấm điểm. Quiz được thiết kế để tìm hiểu dữ liệu. Nhãn Source-Frozen RC dẫn tới biến thể tái lập đã lưu; demo không train policy.

## Nguồn dữ liệu và ảnh

WN9 lấy từ [MoSE4MKGC](https://github.com/OreOZhao/MoSE4MKGC/tree/192d9ff2882b5be6ef8239a37cbd90c5dd102b5a/src_data/WN9), commit `192d9ff2882b5be6ef8239a37cbd90c5dd102b5a`. Nhãn và gloss tiếng Anh lấy từ `ent_name.txt`, `ent_dscp.txt`. Copyright WordNet thuộc Princeton University; xem [WordNet license](WORDNET-LICENSE.txt).

Ảnh được chọn từ đúng member trong manifest WN18, ánh xạ synset WN9 `n########` sang WN18 `########`. Cách này dùng chung ảnh/đặc trưng encoder; graph checkpoint WN18 được giữ riêng. Quyền ảnh thuộc chủ sở hữu gốc.

`imageAvailable` ghi entity có ảnh nguồn hay không; `image` trỏ thumbnail đóng gói. Một entity chưa có thumbnail vẫn có thể có ảnh ở nguồn. Nhóm hiển thị được biên tập cho giao diện. `data/provenance.json` lưu input hash, dòng nguồn, CRC32 archive member và SHA256 thumbnail. Raw benchmark, ZIP ảnh, features và checkpoint được giữ ngoài Git.

## Tạo lại dữ liệu demo

Trong môi trường tái lập có Pillow:

```bash
python scripts/export_demo_data.py --help
```

Exporter tuyển chọn thực thể, tạo bản ghi bằng chứng và thumbnail. Dùng các đối số trong `--help` nếu raw assets ở đường dẫn khác. Các split gốc được giữ nguyên khi xuất.

## Kiểm tra và triển khai

```bash
python -m unittest discover -s tests -p 'test_demo_data.py' -v
```

Kiểm tra bao gồm phạm vi dữ liệu, tham chiếu, câu hỏi, inverse facts, dòng nguồn, hash thumbnail và bundle dưới 2 MiB. Khi có raw WN9, kiểm tra thêm split hash, triple gốc và known-positive choices; CI báo skip phần này nếu thiếu raw assets.

[Workflow deploy](../.github/workflows/demo-pages.yml) kiểm tra và đưa `demo/` lên GitHub Pages khi push thay đổi demo vào `main`. Pull request chỉ chạy kiểm tra. Có thể chạy workflow thủ công bằng `workflow_dispatch`.
