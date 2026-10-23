# DB15K raw-source trace and clean-split sensitivity

Date: 2026-10-23. Status: source audit, clean split, smoke test, and full
teacher-to-student retraining **PASS**. Original artifact hashes verified
unchanged after training. Saved student checkpoint independently reloaded and
evaluated on the full test set; metrics matched exactly.

## Source finding

The validation contamination is present in the **currently published MMRNS raw
files**, before this repository's preprocessing. This finding applies to the
public files inspected on this date; it does not establish which private files
were used to obtain the DSoM paper's numbers.

The [MMRNS README](https://github.com/quqxui/MMRNS#datasets) links to its
[Google Drive datasets](https://drive.google.com/drive/folders/1sFC-P9RKnikqNXjmLcj0IX7x5zvRs-Yj).
The DB15K subfolder is
[MMKB-DB15K](https://drive.google.com/drive/folders/1eWW4QquPkZU2hEVONkJfLGpsBY_IpwR7).

Three independently checked stages agree:

1. Fresh downloads from the public folder match the previously downloaded
   `../dataset_audit/MMRNS/MMKB-DB15K/{train,valid,test}.txt` byte-for-byte.
2. Those files match `src_data/MMKB-DB15K/{train,valid,test}` byte-for-byte.
3. Encoding their rows with the existing `ent_id` and `rel_id` reproduces the
   processed `data/MMKB-DB15K/{train,valid,test}.pickle`, including row order.

The fresh copies and retrieval record are saved under
`runs/audit/db15k_raw_source_20261023/`. Public file IDs, URLs, SHA256 hashes,
and audit counts are in [the source-trace report](../audits/db15k_source_trace.json).
`src/process_datasets.py` maps rows one-for-one and does not resample validation.

| Original split | Rows | Unique triples | Duplicate rows |
|---|---:|---:|---:|
| Train | 79,222 | 79,222 | 0 |
| Validation | 9,904 | 7,481 | 2,423 |
| Test | 9,902 | 9,902 | 0 |

Original unique exact overlaps: train/validation **6,584**,
validation/test **824**, train/test **0**. Only **73** validation triples occur
in neither train nor test. Separately, **131** original training facts reverse
an official test fact under the *same* relation ID.

## CleanV1 protocol

The original benchmark data, features, result JSONs, and baseline checkpoints
are preserved. New files use `MMKB-DB15K-CleanV1` in separate `data/`,
`src_data/`, and `runs/clean_sensitivity/` directories.

1. Preserve the official test triples and their order. The processed and raw
   test files are copied byte-for-byte.
2. Discard the original validation split completely. Do not add its 73 novel
   facts to the training pool or evaluation filters.
3. Remove the 131 training facts sharing `(min(head,tail), relation,
   max(head,tail))` with any test fact. Save these excluded facts explicitly.
4. From the remaining 79,091 training facts, deterministically hold out 9,904
   validation facts using NumPy `default_rng(2022)`. Shuffle sorted groups;
   approximate relation-proportional quotas first, then fill the remainder.
   Keep opposite directions under the same relation in the same split.
5. Accept a holdout group only if every entity and relation in the eligible
   training pool retains at least one training occurrence.
6. Preserve the complete entity/relation mappings and copy the existing BERT
   and ViT features without changing their values or row alignment.
7. Rebuild evaluation filters using **only the three new splits**. NDKD masks
   continue to use training triples and their reciprocals only, through
   `Dataset.train_to_skip`.

| CleanV1 split | Triples | Entities represented | Relations represented |
|---|---:|---:|---:|
| Train | 69,187 | 12,783 | 273 |
| Validation | 9,904 | 7,893 | 196 |
| Test | 9,902 | 7,989 | 191 |

All three split pairs have zero exact-triple and zero same-relation undirected
group overlap. There are no duplicate rows. The complete mapping still has
12,842 entities and 279 relations, matching the original feature alignment.
The original training split represented 12,785 entities and 273 relations;
two entities lose their only training occurrence when test reversals are
removed. Test entities absent from training increase from 57 to 59.

The full protocol, per-relation counts, input/output hashes, and preserved
original artifact hashes are in `data/MMKB-DB15K-CleanV1/split_manifest.json`.
Output directories are never overwritten by the builder.

## Retraining and result selection

Both teacher and student start fresh. The student loads **only the new teacher**.
The full configuration matches the original Exp01 settings: seed 2022,
ComplExMDR rank 2000, batch size 1000, Adagrad LR 0.1, N3 0.01, max 200 epochs,
validation every 3 epochs, and the existing early-stop behavior.

Student: NDKD, source-frozen RC, temperature 4, KD weight 2, alpha/beta 1/1,
seven strategies, policy hidden size 1024, binary reward +1/-10. The scheduler
remains `min` monitoring validation Hits@10, matching the baseline despite its
known direction mismatch. Changing it here would introduce another confound.

Teacher selection uses validation Hits@10 with the existing oracle-style
minimum modality rank evaluation. Student selection uses validation MRR.
Use the student's `final_result` for test comparison; teacher oracle scores
are not deployable student performance. Test metrics are logged during training
by the existing code but do not select checkpoints.

Local logging is enabled; W&B is opt-in with `DSOM_WANDB=1`. Each run saves exact
commands, a source-code archive and digest, Git HEAD, the working-tree patch,
split manifest, logs, checkpoints, and a comparison summary. The archive is
necessary because the new code need not already be committed.

## Commands

From the repository root:

```bash
# Optional fresh re-download into a NEW directory; requires the existing gdown package.
./dsom/bin/python scripts/download_db15k_raw_source.py \
  --output runs/audit/db15k_raw_source_fresh

# First-time preparation; refuses to overwrite an existing dataset.
./dsom/bin/python scripts/prepare_db15k_clean_split.py \
  --upstream-record runs/audit/db15k_raw_source_20261023/upstream_downloads.json

# Verify original artifacts, deterministic regeneration, disjointness, and filters.
./dsom/bin/python scripts/prepare_db15k_clean_split.py --verify-only

./dsom/bin/python -m pytest -q tests/test_db15k_clean_split.py
bash scripts/run_db15k_clean_sensitivity.sh --smoke
bash scripts/run_db15k_clean_sensitivity.sh --full
```

Both source tools accept `--source` pointing to another copy of the raw source
folder. To prepare from a new download, pass its `upstream_downloads.json` to
the builder instead of the original October 23 retrieval record.

Smoke pipeline passed at
`runs/clean_sensitivity/MMKB-DB15K-CleanV1/smoke/20261023_043857_3369024/`.
Its small training/evaluation subsets verify execution only and are not
benchmark results. Full run output is also collected in
`logs/db15k_clean_v1_full.log`.

Verification completed before full retraining:

- Public-source downloader executed successfully against all three file IDs.
- Two focused tests passed: reversal grouping/determinism/partition coverage,
  and retaining rare entities and relations in training.
- Generated data passed hash preservation, deterministic reconstruction,
  duplicate/overlap checks, exact test preservation, and evaluation-filter checks.
- The actual `Dataset` loader returned shape `(12842, 558, 12842)` and 138,374
  reciprocal training rows. Every held-out validation/test answer was checked
  to be absent from its training-neighbor mask in both prediction directions.
- Teacher-to-student smoke pipeline passed, including the source-frozen
  optimizer membership assertion and loading the newly generated teacher.
- Python lint, shell syntax, and `git diff --check` passed for the new workflow.

## Interpretation limits

This is a **sensitivity analysis**, not an estimate of the causal effect of
validation contamination alone. It also reduces training data, removes test
reversals, changes validation distribution, and changes the all-true evaluation
filters by discarding old validation-only facts. It retains the official test's
cold entities and does not detect semantic inverse relations with different
relation IDs. The reconstructed features and single training seed remain
limitations. A clean score must not replace the original paper-protocol result
without labeling this different split.

## Full results

Full run:
`runs/clean_sensitivity/MMKB-DB15K-CleanV1/full/20261023_043918_3369511/`.

The teacher finished 65 epochs in 763.98 seconds, selecting epoch 54 by
validation Hits@10. At that checkpoint, validation MRR is 0.422351 and
Hits@10 is 0.575323. These are the existing oracle-style teacher metrics.
The original contaminated-validation teacher had selected-checkpoint validation
MRR 0.944744; that numerical contrast is not a controlled causal estimate.

The student finished 38 epochs in approximately 1,523.26 seconds, selecting
epoch 27 by clean validation MRR. Both stages together took approximately
38.1 minutes of training/evaluation time. No original teacher checkpoint was
reused for this student's training.

| Student protocol | Best epoch | Selected validation MRR | Test MRR | Test H@1 | Test H@3 | Test H@10 |
|---|---:|---:|---:|---:|---:|---:|
| Original Exp01, contaminated validation | 63 | 0.941991 | 0.413046 | 0.331701 | 0.454656 | 0.565744 |
| CleanV1 sensitivity | 27 | 0.384404 | 0.380746 | 0.301101 | 0.418754 | 0.532721 |

The CleanV1 test MRR difference is **-0.032300**, or **-3.2300 percentage
points**. Test MR changes from 326.1553 to 386.2926. These values use the
unchanged official test triples, with each protocol's corresponding filters.

Artifacts relative to the full run folder above:

- `summary.json`: original/clean comparison and exact stage result paths.
- `teacher/checkpoints/0/result.json`: completed teacher metrics and settings.
- `student/checkpoints/0/result.json`: completed student metrics and settings.
- `checkpoint_verification.json`: independently reloaded best-student test
  evaluation; exact match to `final_result`.
- `split_manifest.json`: input/output and original artifact hashes.
- `environment.json`: Python 3.12.3, NumPy 2.5.3, PyTorch 2.7.0+cu128,
  CUDA 12.8, RTX 5090; training and split seeds both 2022.
- `source_snapshot.tar.gz`, its SHA256, Git HEAD, and working-tree patch:
  provenance for the source used at launch.

The launcher completed its final original-artifact verification and printed
`CLEAN SENSITIVITY PIPELINE: PASS`. This includes preserved original benchmark
features, mappings, split files, recorded result JSONs, and Exp01 checkpoints.

The validation scores now reflect a disjoint holdout and are substantially
lower than the contaminated scores. The test result shows that this pipeline
is sensitive to the changed protocol, but **does not attribute the entire
decrease to validation contamination**: less training data, test-reversal
removal, validation distribution, and filtering all change together. Treat the
single-seed comparison as exploratory and retain both labeled result sets.
