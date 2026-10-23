# Download, prepare and verify data

All commands run from the repository root with the environment activated.
Choose only datasets needed for your table. Dataset preprocessing is CPU-only;
encoding is a separate, explicitly invoked step. No pretrained graph checkpoints
or feature arrays are shipped in Git.

## Local layout and sources

Use any download location; the examples use ignored `raw/`. Pass its paths
explicitly instead of recreating another machine's `/workspace/...` directory.

```bash
mkdir -p raw
git clone https://github.com/yao8839836/kg-bert.git raw/kg-bert
git clone https://github.com/OreOZhao/MoSE4MKGC.git raw/mose
git -C raw/kg-bert rev-parse HEAD
git -C raw/mose rev-parse HEAD
```

Record source revisions with your experiment. WN9's audited preparation requires
the pinned MoSE revision; checkout that version before WN9 preparation:

```bash
git -C raw/mose fetch origin 192d9ff2882b5be6ef8239a37cbd90c5dd102b5a
git -C raw/mose checkout --detach 192d9ff2882b5be6ef8239a37cbd90c5dd102b5a
```

Primary sources:

| Asset | Source | What to obtain |
|---|---|---|
| DB15K, MKG-W, MKG-Y triples/text/links | [MMRNS](https://github.com/quqxui/MMRNS#datasets), [Drive](https://drive.google.com/drive/folders/1sFC-P9RKnikqNXjmLcj0IX7x5zvRs-Yj) | Original files below; HDF5 embeddings are not raw images |
| FB15K-237 and WN18 triples/text | [KG-BERT](https://github.com/yao8839836/kg-bert/tree/master/data) | Dataset subfolders, including descriptions |
| FB/WN image archive | [MKGformer](https://github.com/zjunlp/MKGformer#data-download), [Drive archive](https://drive.google.com/file/d/197c4fCLVC6F7sCBqZIDwm5tjWeGJnZ5-/view) | `fb15k_wn18_images.zip` |
| Alternative FB/WN images | [MoSE](https://github.com/OreOZhao/MoSE4MKGC#dataset-download) | Baidu link in README, code `4xmv`; verify layout/content, not assumed byte-identical |
| WN9 triples/text | [MoSE WN9](https://github.com/OreOZhao/MoSE4MKGC/tree/main/src_data/WN9), originally [IKRL](https://github.com/thunlp/IKRL) | `train`, `valid`, `test`, `ent_dscp.txt`, `ent_name.txt` |
| DB15K raw image URLs | [MMKB visual data](https://github.com/mniepert/mmkb#visual-data-for-fb15k-yago15k-and-dbpedia15k) | Published image URL archive plus DB15K SameAs links |

Google Drive/Baidu may require a browser, account, or quota wait. Download the
specific original assets and arrange them below; do not assume cloning MMRNS
downloads Drive data. If a source is unavailable, stop at that dataset and record
the missing asset. Do not silently substitute a similarly named dataset or H5
features. The documentation work did not redownload large archives.

Example canonical downloaded layout (before preprocessing):

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

The `descreption` spelling is upstream. If downloaded filenames differ, pass the
actual paths rather than changing parser assumptions. MMRNS can supply MKG-W
`ent_links` in its separate raw-data folder: copy that published file into the
example `raw/mmrns/MKG-W/` folder before the structural command below.

The shared FB/WN ZIP observed here is about 2.9 GiB compressed. Extraction,
encoder weights, features, and teacher/distiller checkpoints require additional
space; an entire experiment grid can use tens of GiB. Check `df -h .` and select
tables accordingly. Raw images are not committed or redistributed by this repo.

## Structural preprocessing

The portable preparer stages source files and calls the existing
`src/process_datasets.py`. It refuses existing processed outputs. The repository
already tracks FB15K-237/WN18 raw splits: existing staging is reused only if its
ordered triples match the downloaded source (newline differences allowed), and
is never rewritten. Source and staged hashes are both recorded. On success:

```text
src_data/DATASET/{train,valid,test,...}
data/DATASET/{ent_id,rel_id,train.pickle,valid.pickle,test.pickle,to_skip.pickle,probas.pickle,...}
```

Run only needed commands:

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

Verify existing outputs instead of rerunning preparation:

```bash
python scripts/prepare_dataset.py --dataset MKG-Y --verify-only
```

Repeat verification with the other dataset names. It reconstructs exact mapped
triples and reciprocal filters independently. Mapping order is sorted across
all splits, while NDKD training neighbors come from training only.

Expected counts for the actual benchmark files used here:

| Dataset | Entities | Original relations | Train | Valid | Test |
|---|---:|---:|---:|---:|---:|
| MMKB-DB15K | 12842 | 279 | 79222 | 9904 | 9902 |
| MKG-W | 15000 | 169 | 34196 | 4276 | 4274 |
| MKG-Y | 15000 | 28 | 21310 | 2665 | 2663 |
| FB15K-237 | 14541 | 237 | 272115 | 17535 | 20466 |
| WN18 | 40943 | 18 | 141442 | 5000 | 5000 |
| WN9 | 6555 | 9 | 11741 | 1337 | 1319 |

DB15K downloaded valid/test counts differ from their order in paper Table 5.
Keep the source file names: original validation has duplicates and overlaps;
do not deduplicate, swap or resplit it to match a table. See the
[source trace](../experiments/DB15K_CLEAN_SPLIT_SENSITIVITY.md). CleanV1 is a
separate sensitivity experiment, not the main benchmark dataset.

## BERT text features

`scripts/build_features.py` adapts explicit input paths to unchanged dataset
encoders. Default `--mode audit` checks paths and entity mapping without Torch
or downloads; it is a **path preflight**, not a full media-coverage audit.
`--mode extract` is the explicit encoding step. These examples use CUDA; text
supports `--device cpu` too, with much slower execution.

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

All use `bert-base-uncased`, penultimate hidden state (`hidden_states[-2]`),
attention-mask mean pooling and max length 512, with seed-2022 missing vectors.
For FB15K-237, the long description file is intentional. For MKG-W, Wikidata
structural URIs must be mapped to DBpedia media keys using `ent_links`. Bare
row-index matching is invalid. Extraction backends check their expected source
syntax/coverage; inspect logs/manifests before training.

## ViT image features

For MKG-Y/MKG-W, use their original raw ZIPs:

```bash
python scripts/build_features.py --dataset MKG-Y --modality image --mode extract \
  --input raw/mmrns/MKG-W-Y-raw_data/MKG-Y_img.zip --batch-size 8
python scripts/build_features.py --dataset MKG-W --modality image --mode extract \
  --input raw/mmrns/MKG-W-Y-raw_data/MKG-W_img.zip \
  --links src_data/MKG-W/ent_links --batch-size 8
```

For FB15K-237, extract the `FB15k-images/` subtree from the shared downloaded ZIP:

```bash
mkdir -p raw/fb-wn-images
unzip -n raw/fb15k_wn18_images.zip 'FB15k-images/*' -d raw/fb-wn-images
python scripts/build_features.py --dataset FB15K-237 --modality image --mode extract \
  --input raw/fb-wn-images/FB15k-images --batch-size 8
```

WN18 reads the other subtree directly inside the ZIP; no second extraction:

```bash
python scripts/build_features.py --dataset WN18 --modality image --mode extract \
  --input raw/fb15k_wn18_images.zip --batch-size 8
```

DB15K uses Freebase-keyed MMKB images aligned through its published SameAs links.
If you already have the raw images, pass their directory. Otherwise retrieve
the published URL archive from MMKB, then explicitly run the historical downloader
with **all paths supplied** (its legacy defaults are machine-specific):

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

Old URLs can be dead; record actual coverage. These recovered images are not
verified as the paper's original set. Do not replace them silently with MMRNS
BEiT or MMKB VGG H5 vectors. Image backends require CUDA, use
`pytorch_pretrained_vit/B_16_imagenet1k`, RGB/resize 384/Normalize(.5,.5), first
valid sorted image, and 1000-dimensional classifier logits. This differs from
the paper's stated 224 input; exact author feature equivalence is unverified.

The wrapper refuses final-output overwrite and binds new feature checkpoints
to input/mapping/backend fingerprints. Directory images use path/size/mtime
metadata, not full image-content hashes. A legacy checkpoint without this
fingerprint is not automatically resumed. Keep existing results; use their
original builder or a separate clean checkout to reconstruct changed inputs.

Verify both outputs for each prepared dataset:

```bash
python scripts/build_features.py --dataset MKG-Y --modality text --mode verify
python scripts/build_features.py --dataset MKG-Y --modality image --mode verify
```

Expected arrays are `(entities,768)` text and `(entities,1000)` image, float32,
finite, ordered by `ent_id`. Shape/finiteness alone is not proof of semantic
alignment; retain source links, alignment checks and feature manifests.

## WN9 from pinned MoSE + verified WN18 features

First finish WN18 preparation and both features above. Then:

```bash
python scripts/prepare_wn9_from_mose.py --mode audit \
  --source-repo raw/mose --parent-features data/WN18 \
  --parent-text raw/kg-bert/data/WN18/entity2text.txt

python scripts/prepare_wn9_from_mose.py --mode prepare \
  --source-repo raw/mose --parent-features data/WN18 \
  --parent-text raw/kg-bert/data/WN18/entity2text.txt

python scripts/prepare_wn9_from_mose.py --mode verify
```

The preparer requires the pinned MoSE files, matching synsets/descriptions and
the expected WN18 feature shapes; it gathers features by synset ID, not row
position. It preserves 6,547 real-image and 8 imputed rows in the recorded source.
Do not retrain/encode WN9 just to duplicate the same generic frozen features.
Do train new WN9 graph teacher/student checkpoints: WN18 graph training includes
many WN9 held-out facts. See [WN9 audit](../experiments/WN9_SOURCE_AUDIT.md).

Existing processed datasets and final features are never overwritten. Use verification for
existing datasets, and preserve raw downloads/manifests. Never delete or rewrite
an existing benchmark merely to make a setup command succeed.
