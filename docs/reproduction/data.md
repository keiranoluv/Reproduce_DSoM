# Chuẩn bị dữ liệu

Mỗi dataset cần hai phần: dữ liệu graph đã tiền xử lý và vector đặc trưng văn bản/ảnh. Làm theo thứ tự tải dữ liệu → tiền xử lý graph → trích xuất đặc trưng → kiểm tra. Chỉ cần chuẩn bị dataset của bảng định chạy.

Các lệnh chạy từ gốc repo sau khi kích hoạt môi trường. Ví dụ dùng thư mục `raw/`, đã được Git ignore. Có thể để dữ liệu ở nơi khác và truyền đường dẫn thực tế vào lệnh.

## 1. Tải dữ liệu gốc

| Asset | Nguồn | Files cần lấy |
|---|---|---|
| DB15K, MKG-W, MKG-Y triples/text/links | [MMRNS](https://github.com/quqxui/MMRNS#datasets), [Drive](https://drive.google.com/drive/folders/1sFC-P9RKnikqNXjmLcj0IX7x5zvRs-Yj) | Files gốc bên dưới; HDF5 embeddings không phải raw images |
| FB15K-237 và WN18 triples/text | [KG-BERT](https://github.com/yao8839836/kg-bert/tree/master/data) | Dataset subfolders, gồm descriptions |
| Archive ảnh FB/WN | [MKGformer](https://github.com/zjunlp/MKGformer#data-download), [Drive archive](https://drive.google.com/file/d/197c4fCLVC6F7sCBqZIDwm5tjWeGJnZ5-/view) | `fb15k_wn18_images.zip` |
| Nguồn ảnh FB/WN thay thế | [MoSE](https://github.com/OreOZhao/MoSE4MKGC#dataset-download) | Baidu link trong README, mã `4xmv`; verify layout/nội dung, không giả định byte-identical |
| WN9 triples/text | [MoSE WN9](https://github.com/OreOZhao/MoSE4MKGC/tree/main/src_data/WN9), nguồn gốc [IKRL](https://github.com/thunlp/IKRL) | `train`, `valid`, `test`, `ent_dscp.txt`, `ent_name.txt` |
| URLs raw images DB15K | [MMKB visual data](https://github.com/mniepert/mmkb#visual-data-for-fb15k-yago15k-and-dbpedia15k) | Image URL archive công bố và DB15K SameAs links |

Với KG-BERT và MoSE, tải repo bằng Git:

```bash
mkdir -p raw
git clone https://github.com/yao8839836/kg-bert.git raw/kg-bert
git clone https://github.com/OreOZhao/MoSE4MKGC.git raw/mose
git -C raw/kg-bert rev-parse HEAD
git -C raw/mose rev-parse HEAD
```

Ghi lại commit nguồn cùng kết quả thực nghiệm. Riêng WN9, script kiểm tra đúng bản MoSE dưới đây:

```bash
git -C raw/mose fetch origin 192d9ff2882b5be6ef8239a37cbd90c5dd102b5a
git -C raw/mose checkout --detach 192d9ff2882b5be6ef8239a37cbd90c5dd102b5a
```

Dữ liệu MMRNS và archive ảnh cần tải riêng từ các liên kết trong bảng. Clone repo MMRNS chưa bao gồm dữ liệu trên Drive. Drive/Baidu có thể yêu cầu tài khoản hoặc bị giới hạn lượt tải; nếu thiếu file, hoàn tất việc tải trước khi xử lý dataset đó.

Sắp xếp dữ liệu như sau:

```text
raw/
  kg-bert/data/{FB15k-237,WN18}/...
  mose/src_data/WN9/...
  mmrns/
    MMKB-DB15K/{train.txt,valid.txt,test.txt,DB15K_description.txt,DB15K_SameAsLink.txt}
    MKG-W/{train.txt,valid.txt,test.txt,ent_links}
    MKG-Y/{train.txt,valid.txt,test.txt}
    MKG-W-Y-raw_data/
      MKG-W_ent2descreption.txt
      MKG-Y_ent2descreption.txt
      MKG-W_img.zip
      MKG-Y_img.zip
  fb15k_wn18_images.zip
  fb-wn-images/FB15k-images/m.ENTITY/IMAGE.JPEG
  mmkb/image-graph_urls.tar.gz
  mmkb/image-graph_images/m.ENTITY/IMAGE.jpg
```

`descreption` là tên file trong nguồn MMRNS. Nếu file tải về có tên khác, truyền tên thực tế vào lệnh. Với MKG-W, đặt file `ent_links` được công bố trong `raw/mmrns/MKG-W/`; file này có thể nằm trong thư mục raw-data riêng của gói tải về.

Archive ảnh FB/WN khoảng 2.9 GiB nén. Ảnh giải nén, trọng số encoder, đặc trưng và checkpoint cần thêm dung lượng; chạy nhiều bảng có thể dùng hàng chục GiB. Kiểm tra dung lượng bằng `df -h .` trước khi tải.

<a id="structural-preprocessing"></a>

## 2. Tiền xử lý graph

`prepare_dataset.py` đưa file nguồn vào `src_data/` rồi gọi `src/process_datasets.py` để tạo mapping, triples và bộ lọc đánh giá:

```text
src_data/DATASET/{train,valid,test,...}
data/DATASET/{ent_id,rel_id,train.pickle,valid.pickle,test.pickle,to_skip.pickle,probas.pickle,...}
```

Chọn lệnh cho dataset cần dùng:

```bash
python scripts/prepare_dataset.py --dataset MKG-Y --source-dir raw/mmrns/MKG-Y
python scripts/prepare_dataset.py --dataset MKG-W --source-dir raw/mmrns/MKG-W
python scripts/prepare_dataset.py --dataset MMKB-DB15K --source-dir raw/mmrns/MMKB-DB15K

python scripts/prepare_dataset.py --dataset FB15K-237 \
  --source-dir raw/kg-bert/data/FB15k-237 \
  --train-file train.tsv --valid-file dev.tsv --test-file test.tsv

python scripts/prepare_dataset.py --dataset WN18 \
  --source-dir raw/kg-bert/data/WN18 \
  --train-file train.tsv --valid-file dev.tsv --test-file test.tsv
```

Script giữ nguyên thứ tự triples. Với FB15K-237/WN18, repo đã có một số raw splits; script chỉ dùng lại khi nội dung và thứ tự khớp nguồn tải về, cho phép khác kiểu xuống dòng. Hash của cả nguồn và bản trong repo được lưu để đối chiếu.

Nếu dữ liệu đã được xử lý, dùng `--verify-only`:

```bash
python scripts/prepare_dataset.py --dataset MKG-Y --verify-only
```

Thay `MKG-Y` bằng dataset tương ứng. Bước này dựng lại mapped triples và reciprocal filters để so với dữ liệu đang có. Mapping được sắp xếp từ cả ba split; neighbor dùng cho NDKD chỉ lấy từ training.

Số lượng trong các bộ dữ liệu đã dùng:

| Dataset | Entities | Base relations | Train | Valid | Test |
|---|---:|---:|---:|---:|---:|
| MMKB-DB15K | 12842 | 279 | 79222 | 9904 | 9902 |
| MKG-W | 15000 | 169 | 34196 | 4276 | 4274 |
| MKG-Y | 15000 | 28 | 21310 | 2665 | 2663 |
| FB15K-237 | 14541 | 237 | 272115 | 17535 | 20466 |
| WN18 | 40943 | 18 | 141442 | 5000 | 5000 |
| WN9 | 6555 | 9 | 11741 | 1337 | 1319 |

DB15K có thứ tự số lượng valid/test khác Table 5 của paper. Giữ nguyên tên và nội dung file nguồn. [CleanV1](../research/tables/db15k-clean-split.md) là một bộ chia riêng dùng để kiểm tra độ nhạy, có kết quả được ghi riêng.

## 3. Trích xuất đặc trưng văn bản

`build_features.py` nhận đường dẫn file mô tả và gọi encoder của dataset. Nếu chưa thêm `--mode extract`, lệnh chỉ kiểm tra đầu vào. Chạy lần lượt những dataset cần dùng:

```bash
# First preflight one input:
python scripts/build_features.py --dataset MKG-Y --modality text \
  --input raw/mmrns/MKG-W-Y-raw_data/MKG-Y_ent2descreption.txt

# Explicit extraction (one dataset at a time):
python scripts/build_features.py --dataset MKG-Y --modality text --mode extract \
  --input raw/mmrns/MKG-W-Y-raw_data/MKG-Y_ent2descreption.txt --device cuda --batch-size 16
python scripts/build_features.py --dataset MKG-W --modality text --mode extract \
  --input raw/mmrns/MKG-W-Y-raw_data/MKG-W_ent2descreption.txt \
  --links src_data/MKG-W/ent_links --device cuda --batch-size 16
python scripts/build_features.py --dataset MMKB-DB15K --modality text --mode extract \
  --input raw/mmrns/MMKB-DB15K/DB15K_description.txt --device cuda --batch-size 16
python scripts/build_features.py --dataset FB15K-237 --modality text --mode extract \
  --input raw/kg-bert/data/FB15k-237/entity2textlong.txt --device cuda --batch-size 32
python scripts/build_features.py --dataset WN18 --modality text --mode extract \
  --input raw/kg-bert/data/WN18/entity2text.txt --device cuda --batch-size 32
```

Các dataset dùng `bert-base-uncased`, hidden state áp chót (`hidden_states[-2]`), mean pooling có attention mask và độ dài tối đa 512. Mô tả thiếu được thay bằng vector ngẫu nhiên với seed 2022. FB15K-237 dùng file mô tả dài `entity2textlong.txt`.

MKG-W cần `ent_links` để nối URI Wikidata của graph với khóa media DBpedia. Xem log và manifest để biết độ phủ mô tả và kết quả ánh xạ.

## 4. Trích xuất đặc trưng ảnh

### MKG-Y và MKG-W

Dùng archive ảnh gốc của từng dataset:

```bash
python scripts/build_features.py --dataset MKG-Y --modality image --mode extract \
  --input raw/mmrns/MKG-W-Y-raw_data/MKG-Y_img.zip --batch-size 8
python scripts/build_features.py --dataset MKG-W --modality image --mode extract \
  --input raw/mmrns/MKG-W-Y-raw_data/MKG-W_img.zip \
  --links src_data/MKG-W/ent_links --batch-size 8
```

### FB15K-237 và WN18

FB15K-237 đọc thư mục `FB15k-images/` sau khi giải nén:

```bash
mkdir -p raw/fb-wn-images
unzip -n raw/fb15k_wn18_images.zip 'FB15k-images/*' -d raw/fb-wn-images
python scripts/build_features.py --dataset FB15K-237 --modality image --mode extract \
  --input raw/fb-wn-images/FB15k-images --batch-size 8
```

WN18 đọc trực tiếp phần ảnh còn lại trong archive chung:

```bash
python scripts/build_features.py --dataset WN18 --modality image --mode extract \
  --input raw/fb15k_wn18_images.zip --batch-size 8
```

### DB15K

Ảnh MMKB dùng khóa Freebase, được nối với DB15K qua SameAs links. Nếu đã có ảnh, truyền thư mục đó vào lệnh trích xuất. Nếu chưa có, tải URL archive từ MMKB và chạy downloader với đầy đủ đường dẫn:

```bash
python scripts/download_db15k_mmkb_images.py \
  --ent-file data/MMKB-DB15K/ent_id \
  --sameas raw/mmrns/MMKB-DB15K/DB15K_SameAsLink.txt \
  --url-archive raw/mmkb/image-graph_urls.tar.gz \
  --images-dir raw/mmkb/image-graph_images \
  --journal data/MMKB-DB15K/mmkb_raw_image_downloads.jsonl \
  --summary data/MMKB-DB15K/mmkb_raw_image_download_summary.json \
  --limit 0 --workers 8

python scripts/build_features.py --dataset MMKB-DB15K --modality image --mode extract \
  --input raw/mmkb/image-graph_images \
  --links raw/mmrns/MMKB-DB15K/DB15K_SameAsLink.txt --batch-size 8
```

Một số URL cũ có thể hỏng; số ảnh tải và encode thành công được ghi trong log/manifest. HDF5 embeddings BEiT của MMRNS hoặc VGG của MMKB không thay được ảnh gốc trong pipeline BERT/ViT này. Tập ảnh tải lại chưa được xác nhận trùng với tập ảnh tác giả dùng.

### Encoder và đầu ra

Encoder ảnh cần CUDA, dùng `pytorch_pretrained_vit/B_16_imagenet1k`, chuyển ảnh sang RGB, resize 384 và normalize với mean/std 0.5. Mỗi entity lấy ảnh hợp lệ đầu tiên theo thứ tự tên file và xuất classifier logits 1.000 chiều. Paper mô tả input 224, nên cần ghi rõ khác biệt này khi báo cáo kết quả.

Wrapper lưu thông tin nhận diện input, mapping và encoder cùng checkpoint trích xuất. Với thư mục ảnh, thông tin này dựa trên đường dẫn, kích thước và mtime. Nó từ chối ghi đè đầu ra hoàn chỉnh hoặc tiếp tục checkpoint cũ thiếu thông tin cần thiết. Nếu đổi input, giữ kết quả cũ và dùng checkout riêng hoặc builder gốc để tạo bộ đặc trưng mới.

## 5. Kiểm tra đặc trưng

Chạy cho từng dataset đã chuẩn bị:

```bash
python scripts/build_features.py --dataset MKG-Y --modality text --mode verify
python scripts/build_features.py --dataset MKG-Y --modality image --mode verify
```

Text có shape `(entities, 768)`, ảnh có shape `(entities, 1000)`; cả hai là float32, không có NaN/Inf và theo thứ tự `ent_id`. Ngoài shape, cần giữ manifest và kiểm tra mapping nguồn để bảo đảm mỗi hàng thuộc đúng entity.

<a id="wn9-from-pinned-mose--verified-wn18-features"></a>

## 6. Chuẩn bị WN9

WN9 dùng graph từ bản MoSE đã ghim ở bước 1 và lấy đặc trưng encoder từ WN18 theo synset ID. Vì vậy, hoàn tất tiền xử lý WN18 và cả hai loại đặc trưng trước:

```bash
python scripts/prepare_wn9_from_mose.py --mode audit \
  --source-repo raw/mose --parent-features data/WN18 \
  --parent-text raw/kg-bert/data/WN18/entity2text.txt

python scripts/prepare_wn9_from_mose.py --mode prepare \
  --source-repo raw/mose --parent-features data/WN18 \
  --parent-text raw/kg-bert/data/WN18/entity2text.txt

python scripts/prepare_wn9_from_mose.py --mode verify
```

Script kiểm tra file MoSE, synset, mô tả và shape của đặc trưng cha. Nó tra hàng theo ID, thay vì dùng cùng chỉ số hàng giữa hai dataset. Bộ đã chuẩn bị có 6.547 hàng ảnh thật và 8 hàng ảnh được điền từ WN18.

Teacher và student WN9 phải train mới. Graph WN18 chứa nhiều facts thuộc phần giữ lại để đánh giá WN9, nên checkpoint graph WN18 không phù hợp để dùng ở đây. Xem [kiểm tra nguồn WN9](../research/provenance/wn9-sources.md) nếu cần đối chiếu chi tiết.

Giữ file tải về và manifest cùng kết quả. Khi dữ liệu và đặc trưng đã qua kiểm tra, chuyển sang [chạy thực nghiệm](running.md).

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="bert-text-features"></a>
<a id="download-prepare-and-verify-data"></a>
<a id="local-layout-and-sources"></a>
<a id="vit-image-features"></a>
