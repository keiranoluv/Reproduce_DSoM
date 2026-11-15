# WN9: nguồn dữ liệu và mapping đặc trưng

WN9 được chuẩn bị từ MoSE commit `192d9ff2882b5be6ef8239a37cbd90c5dd102b5a`. Ngày 2026-10-24, file nguồn, dữ liệu sau xử lý và mapping đặc trưng đã được kiểm tra. Teacher/student WN9 được train mới sau đó; xem [kết quả Table 7](../tables/table-7-wn9.md#recorded-full-run--2026-10-24).

## File nguồn

Bản MoSE dùng khi kiểm tra nằm ở `../dataset_audit/MoSE4MKGC_WN9/`. Năm file `src_data/WN9/{train,valid,test,ent_dscp.txt,ent_name.txt}` khớp từng byte với Git blobs của commit trên.

README [MoSE](https://github.com/OreOZhao/MoSE4MKGC) dẫn nguồn WN9 tới [IKRL](https://github.com/thunlp/IKRL). Số lượng khớp [DSoM Appendix C/Table 5](https://arxiv.org/html/2507.20738v1#A3):

| Item | Quan sát | DSoM paper |
|---|---:|---:|
| Entities | 6,555 | 6,555 |
| Relations trước reciprocal augmentation | 9 | 9 |
| Train triples | 11,741 | 11,741 |
| Valid triples | 1,337 | 1,337 |
| Test triples | 1,319 | 1,319 |

Các split không có duplicate row hoặc exact overlap; mọi entity xuất hiện trong train. Lần kiểm tra dùng MoSE snapshot, chưa tải riêng IKRL archive để so sánh.

## Inverse facts giữa các split

Các relation pair được đếm gồm hypernym/hyponym, has_part/part_of, member_holonym/member_meronym, topic-domain và derivational relation đối xứng, theo `inverse_relation_definition`:

| Nguồn / target split | Exact overlap | Same-relation reverse | Named-inverse reverse |
|---|---:|---:|---:|
| Train / valid | 0 | 5 | 1,198 |
| Train / test | 0 | 10 | 1,186 |
| Valid / test | 0 | 0 | 69 |

Có 1.186/1.319 test triple (89.92%) có named-inverse fact trong train. Đây là tính chất dữ liệu nguồn, được giữ khi tái lập.

Mọi synset WN9 có trong mapping WN18, nhưng graph và split khác nhau: 1.247 test fact WN9 đã có trong WN18 train. Training fact `(n03868863, _part_of, n03868863)` có trong MoSE nhưng vắng ở WN18 union hiện tại. Bộ chuẩn bị giữ self-loop này như nguồn.

## Lấy đặc trưng từ WN18

Script bỏ tiền tố `n` của noun synset WN9 rồi tra ID tám chữ số trong `data/WN18/ent_id`. Cả 6.555 ID đều khớp. Mô tả MoSE không rỗng cũng khớp mô tả KG-BERT đã normalize trong builder WN18.

| Modality | WN9 array shape | Nguồn |
|---|---|---|
| Text | `(6555, 768)`, float32 | Existing WN18 BERT features |
| Image | `(6555, 1000)`, float32 | Existing WN18 ViT classifier-logit features |

Ảnh được kiểm tra theo synset ID. Có 6.547 entity dùng ảnh thật và 8 hàng đã được điền từ WN18; các vector thiếu giữ nguyên RNG sequence của parent. Encoder ảnh là `B_16_imagenet1k`, resize 384×384. Text là bộ BERT được dựng lại trước đó.

Đây là đặc trưng từ frozen encoder. Graph checkpoint WN18 đã thấy nhiều held-out fact WN9, nên Table 7 train mới cả teacher và student trên WN9.

## File sau chuẩn bị và cách kiểm tra

`src_data/WN9/` lưu năm file MoSE nguyên bản. `data/WN9/` khoảng 46.3 MiB, gồm:

- `train.pickle`, `valid.pickle`, `test.pickle`.
- `ent_id`, `rel_id`, `to_skip.pickle`, `probas.pickle`.
- `img_vec.pickle`, `dscp_vec.pickle`.
- `wn18_feature_row_map.tsv`: 6.555 ánh xạ hàng cha và con.
- `preparation_manifest.json`: hash nguồn, input/output và cấu hình.

Preparer dùng `src/process_datasets.py` để tạo sorted ID, uint64 triples, reciprocal filters và training frequency. Nó từ chối output đã có. Verify dựng lại triples/filters/frequency, so từng giá trị đặc trưng với parent array và đối chiếu SHA256. NDKD vẫn dùng training-only neighbor map.

[Source trace](wn9_source_trace.json) được lưu trong Git; báo cáo đầy đủ trên máy là `runs/audit/wn9_sources_20261023_171406_058679/report.json` (timestamp UTC).

Các lệnh dùng ở máy cũ:

```bash
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/prepare_wn9_from_mose.py --mode verify
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python -m pytest -q tests/test_wn9_preparation.py
```

```bash
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/prepare_wn9_from_mose.py --mode prepare
```

Để chuẩn bị trên clone mới với đường dẫn cụ thể, dùng [hướng dẫn WN9 hiện tại](../../reproduction/data.md#wn9-from-pinned-mose--verified-wn18-features). Nếu output đã có, chọn verify.

## Kết quả liên quan

Table 7 ở `runs/table7/full/20261023_173538_401388/` đạt filtered MRR 0.919634, H@1 0.903336, H@10 0.949583; raw MRR là 0.618763 trên cùng checkpoint. Paper báo 0.9213/0.9052/0.9556. Cách đánh giá raw/filtered của paper còn cần xác nhận; thông tin nguồn ở đây giúp kiểm tra input đã dùng.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="feature-reuse-without-encoding"></a>
<a id="files-and-verification"></a>
<a id="inverse-facts-and-wn18-graph-comparison"></a>
<a id="source-and-scope"></a>
<a id="training-follow-up-and-remaining-protocol-question"></a>
<a id="wn9-source-audit-and-prepared-inputs--2026-10-24"></a>
