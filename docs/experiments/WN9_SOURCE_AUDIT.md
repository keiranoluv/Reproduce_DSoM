# WN9 source audit and prepared inputs — 2026-10-24

Status: **data preparation and CPU verification PASS**. This source audit ran
without encoder extraction or GPU training and left existing WN18 features,
datasets, and benchmark results unchanged. Subsequently, the user completed
fresh WN9 teacher/student training for Table 7; CPU artifact review passed on
2026-10-24. See the [recorded full result](TABLE7_WN9_REPRODUCTION.md#recorded-full-run--2026-10-24).

## Source and scope

The existing audit checkout at
`../dataset_audit/MoSE4MKGC_WN9/` was already present; it was not downloaded again.
Repository: [OreOZhao/MoSE4MKGC](https://github.com/OreOZhao/MoSE4MKGC).
Pinned commit: `192d9ff2882b5be6ef8239a37cbd90c5dd102b5a`.
All five files in `src_data/WN9/` were compared byte-for-byte with their Git
blobs at that commit: `train`, `valid`, `test`, `ent_dscp.txt`, `ent_name.txt`.

MoSE's README links WN9 to [IKRL](https://github.com/thunlp/IKRL), while
[DSoM Appendix C/Table 5](https://arxiv.org/html/2507.20738v1#A3) reports:

| Item | Observed | DSoM paper |
|---|---:|---:|
| Entities | 6,555 | 6,555 |
| Relations before reciprocal augmentation | 9 | 9 |
| Train triples | 11,741 | 11,741 |
| Valid triples | 1,337 | 1,337 |
| Test triples | 1,319 | 1,319 |

All split rows are unique, all entities appear in training, and exact triple
intersections between every split pair are zero. No valid/test swapping or
deduplication was performed. The IKRL archive has not been independently
downloaded/compared; the verified source is the pinned MoSE snapshot.

## Inverse facts and WN18 graph comparison

The audit counts reversed facts under the explicitly listed relation pairs
in `inverse_relation_definition`, including hypernym/hyponym, has_part/part_of,
member_holonym/member_meronym, topic-domain relations, and the symmetric
derivational relation. These counts are distinct from exact duplicate triples:

| Source / target split | Exact overlap | Same-relation reverse | Named-inverse reverse |
|---|---:|---:|---:|
| Train / valid | 0 | 5 | 1,198 |
| Train / test | 0 | 10 | 1,186 |
| Valid / test | 0 | 0 | 69 |

Thus 1,186 / 1,319 test triples (89.92%) have a named-inverse fact in training.
This is a property of the downloaded benchmark, not introduced by preprocessing.
Splits remain unchanged for reproduction; this is not a clean sensitivity split.

WN9 synsets all exist in the current WN18 mapping, but the graph files are not
simply a split-preserving subset. For example, 1,247 WN9 test facts occur in
WN18 training. One WN9 training fact is absent from the current WN18 union:
`(n03868863, _part_of, n03868863)`. It is a self-loop already in the committed
MoSE source and was retained. Do not replace WN9 triples using WN18 files.

## Feature reuse without encoding

The mapping explicitly strips the leading `n` from each WN9 noun synset and
looks up that eight-digit ID in `data/WN18/ent_id`. No row-order or numeric-ID
equivalence is assumed. All 6,555 entity IDs match. Their nonempty descriptions
in MoSE exactly match the normalized KG-BERT descriptions used by our WN18
feature builder.

Prepared WN9 features select the corresponding rows from existing frozen-encoder
feature arrays:

| Modality | WN9 array shape | Source |
|---|---|---|
| Text | `(6555, 768)`, float32 | Existing WN18 BERT features |
| Image | `(6555, 1000)`, float32 | Existing WN18 ViT classifier-logit features |

According to the inherited image manifest, 6,547 entities have real-image
features and 8 have imputed vectors. The selected image paths were checked
against synset IDs. Missing-image vectors inherit the WN18 RNG sequence/order;
they were not regenerated using a new WN9 seed. Image preprocessing remains
`B_16_imagenet1k`, resize 384×384; it is not claimed to match the paper's exact
224-resolution feature pipeline. Text features are reused from our earlier
reconstruction, not author-released DSoM features.

This reuses only frozen generic-encoder features. **WN18 graph-trained teacher
or student checkpoints must not be reused for WN9**; they have seen many WN9
held-out facts. The completed Table 7 suite trained both graph models freshly
on WN9; only generic frozen-encoder features were reused.

## Files and verification

New raw snapshot: `src_data/WN9/`, containing the five unchanged MoSE files.
New processed inputs: `data/WN9/`, approximately 46.3 MiB:

- `train.pickle`, `valid.pickle`, `test.pickle`;
- `ent_id`, `rel_id`, `to_skip.pickle`, `probas.pickle`;
- `img_vec.pickle`, `dscp_vec.pickle`;
- `wn18_feature_row_map.tsv`: all 6,555 parent-to-child row assignments;
- `preparation_manifest.json`: source/input/output hashes and protocol limits.

The existing `src/process_datasets.py` generated sorted entity/relation IDs,
uint64 triples, reciprocal filtered-evaluation lists, and training frequencies.
The preparer changes only that module's output-directory variable temporarily,
restores it afterward, and refuses existing WN9 output directories.
No core training/evaluation implementation was edited.

Verification independently reconstructs mapped raw triples, split-union filters,
and training frequencies; it also checks every selected feature value against
the parent array and all recorded input/output SHA256s. Training NDKD neighbor
sets remain the existing dataset loader's training-only reciprocal masks.

Versioned evidence: [wn9_source_trace.json](../audits/wn9_source_trace.json).
Local full report: `runs/audit/wn9_sources_20261023_171406_058679/report.json`.
Runtime folder timestamps are UTC; the local date is 2026-10-24.

To verify the already prepared inputs without Torch or GPU:

```bash
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/prepare_wn9_from_mose.py --mode verify
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python -m pytest -q tests/test_wn9_preparation.py
```

Default mode is an audit only. On a fresh checkout with the pinned MoSE snapshot
and matching WN18 inputs available, preparation is:

```bash
CUDA_VISIBLE_DEVICES='' ./dsom/bin/python scripts/prepare_wn9_from_mose.py --mode prepare
```

Do not repeat prepare on the current outputs; use verify. Raw and processed data
are ignored by Git. Only script/tests/docs/source-trace metadata should be committed.

## Training follow-up and remaining protocol question

A separate teacher/student workflow with new checkpoints was completed in
`runs/table7/full/20261023_173538_401388/`, preserving these source hashes.
Evaluation settings still need reconciliation: [DySarl §4.1.3](https://openreview.net/pdf?id=Gf94NMabeh)
reports raw ranking, whereas the current DSoM dataset evaluator is filtered.
Table 7's exact author protocol is not established by this source audit; record
both raw and filtered results rather than silently comparing different settings.
Teacher oracle minimum ranks also cannot be reported as student performance.

Paper Table 7 targets for DSoM are MRR .9213, Hits@1 .9052, and Hits@10 .9556.
The completed source-frozen student reached filtered MRR .919634 / H@1 .903336 /
H@10 .949583 and raw MRR .618763 on the same checkpoint. This audit establishes
data provenance, not exact author-protocol equivalence.
