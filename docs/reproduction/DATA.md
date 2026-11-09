# Tải, chuẩn bị và kiểm tra dữ liệu

Chạy mọi lệnh từ gốc repo sau khi kích hoạt môi trường. Chỉ chuẩn bị dataset
cần cho bảng muốn reproduce. Preprocessing chạy CPU; encoding là bước riêng
phải gọi tường minh. Git không chứa pretrained graph checkpoints hay feature arrays.

## Cấu trúc thư mục và nguồn

Có thể tải vào thư mục bất kỳ; ví dụ dùng `raw/` đã được Git ignore. Truyền
đường dẫn tường minh, không cần dựng lại thư mục `/workspace/...` của máy khác.

```bash
mkdir -p raw
git clone https://github.com/yao8839836/kg-bert.git raw/kg-bert
git clone https://github.com/OreOZhao/MoSE4MKGC.git raw/mose
git -C raw/kg-bert rev-parse HEAD
git -C raw/mose rev-parse HEAD
```

Ghi source revision cùng thí nghiệm. Bước chuẩn bị WN9 đã audit yêu cầu MoSE
revision được pin; checkout đúng phiên bản trước khi chuẩn bị WN9:

```bash
git -C raw/mose fetch origin 192d9ff2882b5be6ef8239a37cbd90c5dd102b5a
git -C raw/mose checkout --detach 192d9ff2882b5be6ef8239a37cbd90c5dd102b5a
```

Nguồn dữ liệu chính:

| Asset | Nguồn | Files cần lấy |
|---|---|---|
| DB15K, MKG-W, MKG-Y triples/text/links | [MMRNS](https://github.com/quqxui/MMRNS#datasets), [Drive](https://drive.google.com/drive/folders/1sFC-P9RKnikqNXjmLcj0IX7x5zvRs-Yj) | Files gốc bên dưới; HDF5 embeddings không phải raw images |
| FB15K-237 và WN18 triples/text | [KG-BERT](https://github.com/yao8839836/kg-bert/tree/master/data) | Dataset subfolders, gồm descriptions |
| Archive ảnh FB/WN | [MKGformer](https://github.com/zjunlp/MKGformer#data-download), [Drive archive](https://drive.google.com/file/d/197c4fCLVC6F7sCBqZIDwm5tjWeGJnZ5-/view) | `fb15k_wn18_images.zip` |
| Nguồn ảnh FB/WN thay thế | [MoSE](https://github.com/OreOZhao/MoSE4MKGC#dataset-download) | Baidu link trong README, mã `4xmv`; verify layout/nội dung, không giả định byte-identical |
| WN9 triples/text | [MoSE WN9](https://github.com/OreOZhao/MoSE4MKGC/tree/main/src_data/WN9), nguồn gốc [IKRL](https://github.com/thunlp/IKRL) | `train`, `valid`, `test`, `ent_dscp.txt`, `ent_name.txt` |
| URLs raw images DB15K | [MMKB visual data](https://github.com/mniepert/mmkb#visual-data-for-fb15k-yago15k-and-dbpedia15k) | Image URL archive công bố và DB15K SameAs links |

Google Drive/Baidu có thể yêu cầu trình duyệt, tài khoản hoặc chờ quota. Tải đúng
assets gốc và sắp xếp như dưới đây; clone MMRNS không tự tải dữ liệu trên Drive.
Nếu nguồn không truy cập được, dừng ở dataset đó và ghi nhận asset còn thiếu.
Không tự thay bằng dataset cùng tên hoặc H5 features. Đợt viết tài liệu không tải
lại các archive lớn.

Ví dụ cấu trúc dữ liệu tải về trước preprocessing:

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

Tên `descreption` là cách viết từ upstream. Nếu tên file tải về khác, truyền
đường dẫn thực tế thay vì đổi giả định parser. File `ent_links` của MKG-W có thể
nằm trong thư mục raw-data riêng của MMRNS: sao chép file đã công bố đó vào
`raw/mmrns/MKG-W/` trước khi chạy lệnh structural bên dưới.

FB/WN ZIP dùng chung được ghi nhận khoảng 2.9 GiB nén. Giải nén, encoder weights,
features và teacher/distiller checkpoints cần thêm dung lượng; toàn bộ lưới
thí nghiệm có thể tốn hàng chục GiB. Kiểm tra `df -h .` và chọn bảng phù hợp.
Repo không commit hoặc phân phối lại raw images.

## Preprocessing structural data

Portable preparer staging các source files rồi gọi `src/process_datasets.py`
hiện có. Script từ chối nếu processed outputs đã tồn tại. Repo đang track raw
splits FB15K-237/WN18: chỉ tái sử dụng staging khi thứ tự triples khớp nguồn tải
về, cho phép khác newline, và không ghi lại file. Hash nguồn và staging đều
được ghi nhận. Khi thành công:

```text
src_data/DATASET/{train,valid,test,...}
data/DATASET/{ent_id,rel_id,train.pickle,valid.pickle,test.pickle,to_skip.pickle,probas.pickle,...}
```

Chỉ chạy lệnh cho dataset cần dùng:

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

Nếu outputs đã có, kiểm tra thay vì chạy lại preparation:

```bash
python scripts/prepare_dataset.py --dataset MKG-Y --verify-only
```

Lặp lại verify với tên dataset khác. Script độc lập dựng lại mapped triples
và reciprocal filters để đối chiếu chính xác. Mapping được sắp xếp trên mọi
splits, còn training neighbors của NDKD chỉ lấy từ training.

Số lượng kỳ vọng theo các benchmark files thực tế đã dùng:

| Dataset | Entities | Base relations | Train | Valid | Test |
|---|---:|---:|---:|---:|---:|
| MMKB-DB15K | 12842 | 279 | 79222 | 9904 | 9902 |
| MKG-W | 15000 | 169 | 34196 | 4276 | 4274 |
| MKG-Y | 15000 | 28 | 21310 | 2665 | 2663 |
| FB15K-237 | 14541 | 237 | 272115 | 17535 | 20466 |
| WN18 | 40943 | 18 | 141442 | 5000 | 5000 |
| WN9 | 6555 | 9 | 11741 | 1337 | 1319 |

Số valid/test DB15K tải về khác thứ tự được ghi trong Table 5 của paper.
Giữ tên file nguồn: validation gốc có duplicates và overlaps; không deduplicate,
đổi chỗ hoặc chia lại để khớp bảng. Xem [truy nguồn](../experiments/DB15K_CLEAN_SPLIT_SENSITIVITY.md).
CleanV1 là sensitivity experiment riêng, không thay benchmark chính.

## Text features BERT

`scripts/build_features.py` chuyển đường dẫn đầu vào tường minh cho các encoder
dataset hiện có, không sửa encoder. Mặc định `--mode audit` chỉ kiểm tra đường dẫn
và entity mapping, không Torch hay download; đây là **kiểm tra đầu vào**, không
phải audit toàn bộ độ phủ ảnh/văn bản. `--mode extract` mới encoding. Ví dụ dùng
CUDA; text cũng hỗ trợ `--device cpu` nhưng chậm hơn nhiều.

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

Tất cả dùng `bert-base-uncased`, hidden state áp chót (`hidden_states[-2]`),
attention-mask mean pooling, max length 512 và missing vectors với seed-2022.
FB15K-237 cố ý dùng file mô tả dài. Với MKG-W, phải ánh xạ structural URIs
Wikidata sang DBpedia media keys bằng `ent_links`, không ghép theo row index.
Backend kiểm tra cú pháp/coverage nguồn; xem logs/manifests trước khi training.

## Image features ViT

Với MKG-Y/MKG-W, dùng raw ZIP gốc tương ứng:

```bash
python scripts/build_features.py --dataset MKG-Y --modality image --mode extract \
  --input raw/mmrns/MKG-W-Y-raw_data/MKG-Y_img.zip --batch-size 8
python scripts/build_features.py --dataset MKG-W --modality image --mode extract \
  --input raw/mmrns/MKG-W-Y-raw_data/MKG-W_img.zip \
  --links src_data/MKG-W/ent_links --batch-size 8
```

Với FB15K-237, giải nén subtree `FB15k-images/` từ ZIP dùng chung đã tải:

```bash
mkdir -p raw/fb-wn-images
unzip -n raw/fb15k_wn18_images.zip 'FB15k-images/*' -d raw/fb-wn-images
python scripts/build_features.py --dataset FB15K-237 --modality image --mode extract \
  --input raw/fb-wn-images/FB15k-images --batch-size 8
```

WN18 đọc subtree còn lại trực tiếp trong ZIP, không cần giải nén lần nữa:

```bash
python scripts/build_features.py --dataset WN18 --modality image --mode extract \
  --input raw/fb15k_wn18_images.zip --batch-size 8
```

DB15K dùng MMKB images với Freebase keys, được căn chỉnh qua SameAs links
đã công bố. Nếu đã có raw images, truyền thư mục đó. Nếu chưa, lấy URL archive
công bố từ MMKB rồi chạy downloader lịch sử với **đầy đủ đường dẫn tường minh**
vì các giá trị mặc định legacy phụ thuộc máy cũ:

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

URL cũ có thể hỏng; ghi độ phủ thực tế. Chưa xác nhận ảnh tải lại là đúng tập
ảnh gốc của paper. Không tự thay bằng MMRNS BEiT hoặc MMKB VGG H5 vectors.
Image backend cần CUDA, dùng `pytorch_pretrained_vit/B_16_imagenet1k`,
RGB/resize 384/Normalize(.5,.5), chọn ảnh hợp lệ đầu tiên theo thứ tự tên file
và lấy classifier logits 1000 chiều. Khác input 224 được paper mô tả; chưa xác
minh features tương đương chính xác với tác giả.

Wrapper từ chối ghi đè final outputs và gắn feature checkpoint mới với
fingerprints của input/mapping/backend. Với thư mục ảnh, fingerprint dùng
path/size/mtime metadata, không hash toàn bộ nội dung ảnh. Không tự resume
checkpoint legacy thiếu fingerprint này. Giữ kết quả cũ; dùng builder gốc
hoặc checkout sạch riêng nếu tái dựng từ inputs đã thay đổi.

Kiểm tra cả hai outputs cho mỗi dataset đã chuẩn bị:

```bash
python scripts/build_features.py --dataset MKG-Y --modality text --mode verify
python scripts/build_features.py --dataset MKG-Y --modality image --mode verify
```

Arrays kỳ vọng là text `(entities,768)` và image `(entities,1000)`, float32,
finite, theo thứ tự `ent_id`. Shape/finiteness chưa chứng minh alignment đúng
về ngữ nghĩa; giữ source links, alignment checks và feature manifests.

## WN9 từ MoSE đã pin và WN18 features đã kiểm tra

Trước hết hoàn tất preparation WN18 và cả hai features ở trên, sau đó:

```bash
python scripts/prepare_wn9_from_mose.py --mode audit \
  --source-repo raw/mose --parent-features data/WN18 \
  --parent-text raw/kg-bert/data/WN18/entity2text.txt

python scripts/prepare_wn9_from_mose.py --mode prepare \
  --source-repo raw/mose --parent-features data/WN18 \
  --parent-text raw/kg-bert/data/WN18/entity2text.txt

python scripts/prepare_wn9_from_mose.py --mode verify
```

Preparer yêu cầu đúng MoSE files đã pin, synsets/descriptions khớp và WN18
features đúng shape; lấy rows theo synset ID thay vì vị trí hàng. Nguồn đã ghi
nhận giữ 6,547 rows ảnh thật và 8 rows imputed. Không cần encode WN9 chỉ để lặp
lại generic frozen features. Phải train mới WN9 graph teacher/student: WN18
graph training chứa nhiều held-out facts WN9. Xem [audit WN9](../experiments/WN9_SOURCE_AUDIT.md).

Không ghi đè processed datasets và final features đã có. Dùng verify cho
dataset hiện hữu, giữ raw downloads/manifests. Không xóa hoặc viết lại benchmark
chỉ để một lệnh setup chạy thành công.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="bert-text-features"></a>
<a id="download-prepare-and-verify-data"></a>
<a id="local-layout-and-sources"></a>
<a id="structural-preprocessing"></a>
<a id="vit-image-features"></a>
<a id="wn9-from-pinned-mose--verified-wn18-features"></a>
