# DSoM Table 1 — MKG-Y and MKG-W Exp01 Reproduction Results

**Status:** Both full Teacher -> Student pipelines completed successfully.  
**Experiment IDs:** `T1_MKGY_Exp01`, `T1_MKGW_Exp01`  
**Method variant:** Source-Frozen RC baseline (not learned-policy REINFORCE).  
**Evaluation rule:** Report Student **test metrics from `final_result` in the best-validation-checkpoint `result.json`**, not the most recent W&B `test/*` summary.

## 1. Paper comparison (Student test set)

All entries are percentages. Delta is reproduction minus the paper value, in **percentage points (pp)**.

| Dataset | Metric | DSoM paper (%) | Exp01 reproduction (%) | Delta (pp) |
|:--|:--|--:|--:|--:|
| **MKG-Y** | **MRR** | 40.02 | **40.69** | **+0.67** |
| MKG-Y | Hits@1 | 35.67 | **36.01** | +0.34 |
| MKG-Y | Hits@3 | 41.70 | **42.41** | +0.71 |
| MKG-Y | Hits@10 | 47.71 | **48.99** | +1.28 |
| **MKG-W** | **MRR** | 40.90 | **40.58** | **-0.32** |
| MKG-W | Hits@1 | 33.56 | **33.54** | -0.02 |
| MKG-W | Hits@3 | 43.58 | **43.19** | -0.39 |
| MKG-W | Hits@10 | 55.28 | **53.73** | -1.55 |

**Interpretation:** Both reproduced Student runs are close to the paper's reported Table 1 figures. These are single-run results; comparable numerical scores **do not by themselves establish an exact methodological reproduction**.

## 2. Run summary

| Field | MKG-Y Exp01 | MKG-W Exp01 |
|:--|:--|:--|
| Entities | 15,000 | 15,000 |
| Original relations | 28 | 169 |
| Train / Valid / Test triples | 21,310 / 2,665 / 2,663 | 34,196 / 4,276 / 4,274 |
| Teacher best epoch | 36 | 72 |
| Teacher total epochs | 47 | 83 |
| Student best epoch | 42 | 54 |
| Student total epochs | 53 | 65 |
| Student best validation MRR | 0.38296 | 0.39081382006406784 |
| Teacher runtime (W&B) | ~122.1 s | ~345.6 s |
| Student runtime (W&B) | Not recorded in this summary | ~483.3 s |
| Training pipeline | **PASS** | **PASS** |
| Student checkpoint selection | Best validation MRR | Best validation MRR |

## 3. Exact best-checkpoint Student test metrics

The following are the unrounded outputs of each completed full pipeline:

| Dataset | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|:--|--:|--:|--:|--:|--:|
| MKG-Y | 468.45867919921875 | 0.4069194197654724 | 0.36012017726898193 | 0.4241456985473633 | 0.48986104130744934 |
| MKG-W | 232.71842193603516 | 0.4058402553200722 | 0.3354000747203827 | 0.43191391229629517 | 0.5373186469078064 |

These values came from the `STUDENT FULL: PASS` output, which reads `result["final_result"]` from the Student `result.json` in the checkpoint directory.

## 4. Training setup and provenance

Shared settings for the Exp01 pipelines:

- Teacher: `ComplExMDR`, using structural, image and description modalities.
- Student: structural model distilled from its corresponding Teacher checkpoint.
- `rank=2000`, `batch_size=1000`, optimizer `Adagrad`, initial learning rate `0.1`.
- `N3` regularizer with `reg=0.01`; scheduler mode `min`.
- `max_epochs=200`, validation every 3 epochs, early stopping patience 10.
- Student: `kd_loss_type=kdrl`, `teacher_strategy=rc`, `policy_optimizer=source_frozen`, KD temperature 4 and KL weight 2.
- Seed encoded in experiment names: `2022`.

**MKG-Y Git SHA confirmed in Student `result.json`:** `39c7886f8bfd728e9c79afb11a0800ae0ac4fad7`.

**MKG-W Git SHA confirmed in Student `result.json`:** `e148dbeb8330a36cf3f41d3e60fd103a942304b4`.

**MKG-Y W&B Student:** https://wandb.ai/keiranoluv-hcmus/DSoM-Reproduction/runs/2a9t5tjk

### Result artifact locations

- MKG-Y Student run ID: `20261020_174808_2562445`
- MKG-W Student run ID: `20261020_181221_2590139`
- Each Student `result.json` resides below: `runs/student/<DATASET>/full/<EXPERIMENT_NAME>/<RUN_ID>/checkpoints/0/result.json`
- The corresponding Teacher checkpoint path is saved in the Student `result.json` field `teacher_checkpoint`.

### Why W&B `test/*` differs from the final values

`test/*` in the W&B run summary is the **last logged evaluation**, not necessarily the metrics recomputed or stored for the **best Student checkpoint**. For MKG-W, W&B reports `test/MRR=0.4047639220952988` at epoch 63, whereas the best-checkpoint `final_result.MRR=0.4058402553200722` at best epoch 54. The run trained 65 total epochs. Thus, **use `final_result` in the local JSON for Table 1**.

## 5. Dataset and modality audit highlights

- **MKG-Y:** 15,000 entities; image vectors shape `(15000, 1000)` and description vectors shape `(15000, 768)`; structural and feature validation completed.
- **MKG-W:** Structural triples and entity mapping audited successfully; 15,000/15,000 Wikidata-to-DBpedia mappings, with zero missing links or conflicts.
- **MKG-W text:** 14,150 descriptions matched to entities; 850 missing; audited alignment **PASS**.
- **MKG-W raw images:** 8,959 entities had *candidate images* in the archive; 6,041 had no raw image candidates; audited alignment **PASS**. Candidate coverage is not the same as verified successfully decoded image count.
- Raw-image-based feature preparation may differ from the released MMRNS image embedding coverage. Preserve feature manifests to document actual encoded-image counts.

See the dedicated MKG-Y and MKG-W dataset preparation documents for the full preprocessing commands and input SHA256 values.

## 6. Reproduction limitations and next steps

1. **RC policy is frozen:** `source_frozen` samples policy actions but does not update policy weights; this is a baseline, not the paper's learned-policy RC implementation. Logs for MKG-W show 69/69 batches with nonzero policy gradients but 0 batches with policy weights updated, as expected.
2. **Feature pipeline differences:** BERT and ViT feature extraction, image selection and missing-modality initialization are reproduction choices; full equivalence with the authors' feature protocol is not verified.
3. **Single seed / single run:** Do not interpret the positive or negative deltas as statistical improvements or degradation without repeated-seed evaluation.
4. **Paper figures:** Paper benchmark values in this table were transcribed in the reproduction discussion; recheck them against the publisher's Table 1 before formal publication.
5. **Provenance completion:** Add the MKG-W Git SHA and the W&B run URL to this document once retrieved from local artifacts, without altering the SHA used during training.

**Conclusion:** `T1_MKGY_Exp01` and `T1_MKGW_Exp01` both reached `FULL PIPELINE: PASS`; the best-checkpoint Student test metrics closely match the reference Table 1 results under the documented Source-Frozen RC protocol.
