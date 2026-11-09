# WN9 — Audit nguồn và inputs đã chuẩn bị, 2026-10-24

Trạng thái: **preparation/kiểm tra CPU PASS**. Audit không encoder extraction
hoặc GPU training, giữ WN18 features/data/results cũ. Sau đó người dùng train
mới WN9 teacher/student Table 7; CPU artifact review PASS ngày 2026-10-24.
Xem [full result](TABLE7_WN9_REPRODUCTION.md#recorded-full-run--2026-10-24).

## Nguồn và phạm vi

Checkout audit `../dataset_audit/MoSE4MKGC_WN9/` đã có, không tải lại.
Repo [OreOZhao/MoSE4MKGC](https://github.com/OreOZhao/MoSE4MKGC), pin commit
`192d9ff2882b5be6ef8239a37cbd90c5dd102b5a`. Cả năm files `src_data/WN9/` được
so từng byte với Git blobs: `train`, `valid`, `test`, `ent_dscp.txt`, `ent_name.txt`.

README MoSE dẫn WN9 tới [IKRL](https://github.com/thunlp/IKRL), còn
[DSoM Appendix C/Table 5](https://arxiv.org/html/2507.20738v1#A3) báo:

| Item | Quan sát | DSoM paper |
|---|---:|---:|
| Entities | 6,555 | 6,555 |
| Relations trước reciprocal augmentation | 9 | 9 |
| Train triples | 11,741 | 11,741 |
| Valid triples | 1,337 | 1,337 |
| Test triples | 1,319 | 1,319 |

Mọi split rows unique, mọi entities xuất hiện training; exact intersections
giữa split pairs bằng zero. Không đổi valid/test hoặc deduplicate. Chưa tải
độc lập/so IKRL archive; nguồn verified là pinned MoSE snapshot.

## Inverse facts và đối chiếu graph WN18

Audit đếm reverse facts theo relation pairs trong `inverse_relation_definition`,
gồm hypernym/hyponym, has_part/part_of, member_holonym/member_meronym,
topic-domain và symmetric derivational relation. Khác với exact duplicates:

| Nguồn / target split | Exact overlap | Same-relation reverse | Named-inverse reverse |
|---|---:|---:|---:|
| Train / valid | 0 | 5 | 1,198 |
| Train / test | 0 | 10 | 1,186 |
| Valid / test | 0 | 0 | 69 |

Có 1,186 / 1,319 test triples (89.92%) có named-inverse fact trong train.
Đây là tính chất benchmark nguồn, không do preprocessing. Giữ nguyên splits
để reproduce, không phải clean sensitivity split.

Mọi WN9 synsets có trong WN18 mapping nhưng graph không đơn thuần là subset
giữ nguyên splits: 1,247 test facts WN9 có ở WN18 train. Một training fact
WN9 không có trong WN18 union: `(n03868863, _part_of, n03868863)`, self-loop
đã có ở MoSE commit và được giữ. Không thay WN9 triples bằng files WN18.

## Reuse features không encode lại

Mapping bỏ tiền tố `n` của noun synset WN9, tra ID tám chữ số trong
`data/WN18/ent_id`; không giả định row order/numeric IDs tương đương. Khớp đủ
6,555 IDs. Descriptions không rỗng ở MoSE khớp chính xác normalized KG-BERT
descriptions dùng trong WN18 feature builder.

WN9 features lấy đúng rows từ frozen-encoder feature arrays hiện có:

| Modality | WN9 array shape | Nguồn |
|---|---|---|
| Text | `(6555, 768)`, float32 | Existing WN18 BERT features |
| Image | `(6555, 1000)`, float32 | Existing WN18 ViT classifier-logit features |

Manifest kế thừa có 6,547 entities ảnh thật, 8 imputed vectors. Kiểm tra paths
ảnh theo synset IDs. Missing vectors kế thừa RNG sequence/order WN18, không
tạo lại bằng seed WN9 mới. Vẫn `B_16_imagenet1k`, resize 384×384, không khẳng
định khớp pipeline input 224 paper. Text features reuse reconstruction trước,
không phải features DSoM tác giả công bố.

Chỉ reuse generic frozen-encoder features. **Không reuse WN18 graph teacher/
student checkpoints cho WN9** vì đã thấy held-out facts. Table 7 hoàn thành
train mới cả hai graph models trên WN9.

## Files và kiểm chứng

Raw snapshot mới `src_data/WN9/` chứa năm MoSE files nguyên bản; processed
inputs mới `data/WN9/` khoảng 46.3 MiB:

- `train.pickle`, `valid.pickle`, `test.pickle`;
- `ent_id`, `rel_id`, `to_skip.pickle`, `probas.pickle`;
- `img_vec.pickle`, `dscp_vec.pickle`;
- `wn18_feature_row_map.tsv`: đủ 6,555 parent-to-child row assignments;
- `preparation_manifest.json`: source/input/output hashes và giới hạn protocol.

`src/process_datasets.py` hiện có tạo sorted entity/relation IDs, uint64
triples, reciprocal evaluation filters và training frequencies. Preparer chỉ
đổi tạm biến output-directory rồi khôi phục; từ chối WN9 output đã tồn tại.
Không sửa core training/evaluation.

Verify độc lập dựng mapped triples, split-union filters, training frequencies;
so mỗi feature value với parent array và mọi input/output SHA256 đã ghi.
NDKD neighbors vẫn dùng training-only reciprocal masks của dataset loader.

Bằng chứng versioned: [wn9_source_trace.json](../audits/wn9_source_trace.json).
Full report local `runs/audit/wn9_sources_20261023_171406_058679/report.json`.
Timestamp thư mục là UTC; ngày local 2026-10-24.

Kiểm tra inputs đã chuẩn bị, không Torch/GPU:

```bash
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/prepare_wn9_from_mose.py --mode verify
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python -m pytest -q tests/test_wn9_preparation.py
```

Default mode chỉ audit. Với fresh checkout có pinned MoSE snapshot và WN18
inputs khớp, chuẩn bị bằng:

```bash
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/prepare_wn9_from_mose.py --mode prepare
```

Không prepare lại outputs hiện tại; dùng verify. Git ignore raw/processed data.
Chỉ commit script/tests/docs/source-trace metadata.

## Training tiếp theo và vấn đề protocol còn mở

Workflow teacher/student tách stages, checkpoints mới hoàn thành tại
`runs/table7/full/20261023_173538_401388/`, giữ source hashes. Vẫn cần xác minh
evaluation setting: [DySarl §4.1.3](https://openreview.net/pdf?id=Gf94NMabeh)
báo raw, DSoM evaluator dùng filtered. Audit không xác nhận exact Table 7
protocol; báo cả hai, không âm thầm so khác settings. Không báo teacher oracle
minimum ranks như student performance.

Targets DSoM/Table 7: MRR .9213, Hits@1 .9052, Hits@10 .9556. Source-frozen
student đạt filtered MRR .919634/H@1 .903336/H@10 .949583, raw MRR .618763
trên cùng checkpoint. Audit xác lập provenance, không exact protocol equivalence.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="feature-reuse-without-encoding"></a>
<a id="files-and-verification"></a>
<a id="inverse-facts-and-wn18-graph-comparison"></a>
<a id="source-and-scope"></a>
<a id="training-follow-up-and-remaining-protocol-question"></a>
<a id="wn9-source-audit-and-prepared-inputs--2026-10-24"></a>
