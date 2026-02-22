# DSoM4MKGC
Code for paper "Dark Side of Modalities: Reinforced Multimodal Distillation for Multimodal Knowledge Graph Reasoning", ACM MM, 2025.

## Dataset Download

For 
You can download dataset FB15K-237, WN18 from [kbc](https://github.com/facebookresearch/kbc). 
The descriptions of entities can be downloaded from [KG-BERT](https://github.com/yao8839836/kg-bert).
The images of entities can be downloaded from [MMKB](https://github.com/mniepert/mmkb), [RSME](https://github.com/wangmengsd/RSME), or [MKGformer](https://github.com/zjunlp/MKGformer).
The entity images of Freebase and WordNet can be found in BaiduNetDisk [MKG_img](https://pan.baidu.com/s/1cbexBtCwxiXM8MeDUvNpRg), code: 4xmv.

The triples, descriptions, and images of DB15K, MKG-W, and MKG-Y datasets are from [MMRNS](https://github.com/quqxui/MMRNS).

## Dataset Preprocess

For triples data, the data could be preprocessed by `src/process_datasets.py`.

For text data, we used bert-base-uncased model from [transformers](https://github.com/huggingface/transformers) as encoder.
We use the mean aggregation of token hidden states as text feature of entity.
The text encoder is in `src/text_encoder.py`.

For image data, we used pytorch_pretrained_vit model as encoder, which is the same as [RSME](https://github.com/wangmengsd/RSME).
The image encoder is in `src/img_encoder.py`.

For the missing descriptions or images, we impute the feature with `np.random.normal` vector.

After extracting features from fixed encoders, we save the text and image features of entities in a pickle file and save the file in `data/DATASET_NAME/`.

The expected project structure is:
```
DSoM4MKGC
 |-- src
 |-- src_data           # downloaded triples of dataset
 |    |-- DATASET_NAME
 |    |    |-- train
 |    |    |-- valid
 |    |    |-- test
 |    |    |-- ...      # other files
 |-- data
 |    |-- DATASET_NAME
 |    |    |-- ...      # files from process_datasets.py
 |    |    |-- YOUR_TEXT_FEATURE.pickle
 |    |    |-- YOUR_IMG_FEATURE.pickle
```

## How to run

### Step 1: Teacher Model Training

After preprocessing, you could run `src/learn.py` to train the modality split MKGR model.

```bash
cd src
CUDA_VISIBLE_DEVICES='1' python learn.py --model ComplExMDR --ckpt_dir YOUR_CKPT_DIR --dataset MMKB-DB15K --early_stopping 10 --fusion_dscp True --fusion_img True --modality_split True --img_info PATH_TO_YOUR_IMG_FEATURE.pickle --dscp_info PATH_TO_YOUR_TEXT_FEATURE.pickle
```

**Note that the displayed metrics during training are more like an upper bound of modality split KGs performance, which is not the final prediction performance.**

### Step2: Reinforced Distillation to Student Model

Update the saved teacher model path dict `teacher_model_path` in `src/kd_learn.py` as:

```python
teacher_model_path = {'MMKB-DB15K': 'YOUR_PATH_TO_TEACHER_MODEL_ComplExMDR.pth',
                      'FB15K-237': 'YOUR_FB15K-237_TEACHER_MODEL_PATH.pth'
                      }[args.dataset]
```

Then run the reinforced distillation to train the student model. 

```bash
CUDA_VISIBLE_DEVICES='1' python kd_learn.py --dataset MMKB-DB15K --kd_loss_type kdrl --kd_kl_weight 2 --ckpt_dir YOUR_CKPT_DIR
```

The performance of the student model is the final performance. 

## Citation
Please cite the following paper as reference if you find our work useful.

```bibtex
@inproceedings{zhao2025DSoM,
author = {Zhao, Yu and Zhang, Ying and Sui, Xuhui and Zhou, Baohang and Zhu, Haoze and Pan, Jeff Z. and Yuan, Xiaojie},
title = {Dark Side of Modalities: Reinforced Multimodal Distillation for Multimodal Knowledge Graph Reasoning},
year = {2025},
isbn = {9798400720352},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3746027.3754576},
doi = {10.1145/3746027.3754576},
booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
pages = {2506–2515},
numpages = {10},
keywords = {knowledge distillation, multimodal knowledge graphs, multimodal knowledge reasoning, multimodal learning},
location = {Dublin, Ireland},
series = {MM '25}
}
```