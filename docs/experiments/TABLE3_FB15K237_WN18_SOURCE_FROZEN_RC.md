# DSoM Table 3 Reproduction — Source-Frozen RC

## 1. Experiment information

- Paper: Dark Side of Modalities: Reinforced Multimodal Distillation for Multimodal Knowledge Graph Reasoning (ACM MM 2025).
- Paper: https://arxiv.org/abs/2507.20738
- Released repository: https://github.com/OreOZhao/DSoM
- Reproduction repository: https://github.com/keiranoluv/Reproduce_DSoM
- Reproduction branch: `experiment/wn18-table3`
- Source-Frozen RC baseline origin: `experiment/source-frozen-rc`
- Baseline commit (FB15K-237 Source-Frozen RC): `1dfabf9b41a3a5ef700170f4227ca2c4c48a0e36`
- Reproduction branch: `experiment/wn18-table3`
- Reproduction tag: `table3-source-frozen-20261020`
- Seed: 2022
- GPU: NVIDIA RTX 5090 32GB
- Datasets: FB15K-237 and WN18 (not WN18RR)
- Student policy optimizer: `source_frozen`

## 2. Student Table 3 comparison

Official reproduction results are taken from `final_result`
in the training `result.json`, corresponding to the checkpoint
selected using the best Validation MRR.

| Dataset | Source | MR ↓ | MRR ↑ | H@1 (%) | H@3 (%) | H@10 (%) |
|---|---|---:|---:|---:|---:|---:|
| FB15K-237 | Paper | 116.0000 | NR | 27.50 | 41.10 | 56.40 |
| FB15K-237 | Source-Frozen RC | 116.5569 | 0.373564 | 27.79 | 41.05 | 56.67 |
| WN18 | Paper | 7.0000 | NR | 95.00 | 96.20 | 97.40 |
| WN18 | Source-Frozen RC | 8.9588 | 0.955870 | 94.87 | 95.92 | 96.89 |

Paper Table 3 does not report MRR for these datasets.
`NR` means not reported.

### Differences: Reproduction minus Paper

| Dataset | Delta MR | Delta H@1 (pp) | Delta H@3 (pp) | Delta H@10 (pp) |
|---|---:|---:|---:|---:|
| FB15K-237 | +0.5569 | +0.29 | -0.05 | +0.27 |
| WN18 | +1.9588 | -0.13 | -0.28 | -0.51 |

- MR: lower is better.
- Hits@K: higher is better.
- `pp` means percentage points.
- Paper MR values are reported as integers.

## 3. Training configuration

The same Student baseline configuration was applied to both datasets.

| Parameter | Value |
|---|---|
| Student model | ComplExMDR, structure-only |
| Teacher modalities | Structure, Image, Description |
| Student rank | 2000 |
| Batch size | 1000 |
| Optimizer | Adagrad |
| Learning rate | 0.1 |
| Scheduler mode | min |
| Regularizer | N3 (0.01) |
| KD loss | kdrl |
| Teacher strategy | rc |
| RC strategy count | 7 |
| KD temperature | 4 |
| KD CE weight | 1 |
| KD KL weight | 2 |
| DKD alpha / beta | 1 / 1 |
| RC policy optimizer | source_frozen |
| RC policy temperature | 1 |
| RC hidden size | 1024 |
| RC reward / penalty | 1 / 10 |
| RC reward mode | binary |
| Validation interval | 3 epochs |
| Max epochs | 200 |
| Early stopping | 10 |
| Checkpoint selection | Best Validation MRR |

### Training summary

| Dataset | Best epoch | Total epochs | Best Valid MRR |
|---|---:|---:|---:|
| FB15K-237 | 60 | 71 | 0.379939 |
| WN18 | 21 | 32 | 0.953347 |

Both experiments terminated through early stopping.

## 4. WN18 Teacher

The WN18 Teacher was trained using ComplExMDR with
three modalities, Adagrad, batch size 1000, rank 2000,
learning rate 0.1, N3 regularization and scheduler `min`.

- Best epoch: 39
- Total epochs: 50
- Best Validation Hits@10: 97.49%
- Validation MRR at selected epoch: 0.959601
- Test MR: 7.4043
- Test MRR: 0.962368
- Test Hits@1: 95.48%
- Test Hits@3: 96.73%
- Test Hits@10: 97.62%

Teacher checkpoint selection uses Validation Hits@10.

Important: the released Teacher evaluation uses the minimum
of the modality-specific ground-truth ranks. This is an
oracle-style metric, not a deployable score-fusion inference
protocol. It should not be treated as directly comparable
to Student inference quality.

## 5. Paper-intended vs released implementation

### Paper-intended algorithm

The paper describes an RC policy trained using REINFORCE
to dynamically select helpful multimodal Teacher combinations.

The Student learns from multimodal Teacher logits through
knowledge distillation, including neighbor-decoupled KD.

### Released implementation

In the original released source, `Distiller.get_learnable_parameters()`
returns Student parameters only.

The released `KD_RL` does not override this method.

Consequently, the policy gradient loss may be computed, but
the RC policy parameters are not included in the corresponding
optimizer parameter list.

### Reproduction modification

The fork explicitly supports:

`--policy_optimizer source_frozen`

This mode preserves seven-strategy sampling, reward calculation
and REINFORCE loss computation, but updates Student parameters
without updating the policy network.

The WN18 smoke diagnostics recorded nonzero policy gradients
and zero policy weight updates.

Close reproduction of Table 3 under Source-Frozen RC does not
prove that the original authors trained a frozen policy, nor
does it establish the effectiveness of learned RC.

Controlled learned-policy ablations are needed to assess the
contribution of reinforcement learning.

## 6. Data and evaluation limitations

- WN18 structure was audited against KBC/WN.
- WN18 contains 40,943 entities and 18 original relations,
  expanded to 36 with reciprocals.
- WN18 image coverage is 7,801 / 40,943 entities (19.05%).
- Missing WN18 image features were randomly imputed.
- Exact equivalence of the reproduced image/text features
  to the authors' feature extraction is not established.
- WN18 includes inverse-relation leakage effects known
  to make link prediction substantially easier.
- Both reported Student results use one seed, without
  cross-seed variance estimates.
- W&B `test/*` at the final logging step is not necessarily
  the evaluation result of the best checkpoint.
- Results in this document use `final_result` instead.

For WN18 data provenance, see
[WN18 Dataset Preparation](../WN18_DATA_PREPARATION.md).

## 7. Local experiment artifacts

### FB15K-237

- Teacher checkpoint: `runs/teacher/FB15K-237/full/20261017_000249/checkpoints/0/m-20261017_0002-n-teacher_fb15k237_full_20261017_000249.pth`
- Student checkpoint: `runs/student/FB15K-237/table3_source_frozen_rc/full/20261020_121832/checkpoints/0/student-20261020_1218-n-table3_source_frozen_rc_full_20261020_121832.pth`
- Student result: `runs/student/FB15K-237/table3_source_frozen_rc/full/20261020_121832/checkpoints/0/result.json`

### WN18

- Teacher checkpoint: `runs/teacher/WN18/full/scheduler-min/20261020_190522/checkpoints/0/m-20261020_1905-n-teacher_wn18_full_scheduler-min_20261020_190522.pth`
- Teacher result: `runs/teacher/WN18/full/scheduler-min/20261020_190522/checkpoints/0/result.json`
- Student checkpoint: `runs/student/WN18/table3_source_frozen_rc/full/20261020_201526/checkpoints/0/student-20261020_2015-n-table3_source_frozen_rc_full_20261020_201526.pth`
- Student result: `runs/student/WN18/table3_source_frozen_rc/full/20261020_201526/checkpoints/0/result.json`

These paths refer to local training artifacts and are
intentionally excluded from Git tracking.

## 8. Conclusion

The Source-Frozen RC baseline closely reproduces the
published Student Table 3 metrics on both FB15K-237 and WN18.

This supports reproducibility of the released-code behavior,
but does not independently validate the paper-intended
REINFORCE policy learning mechanism.

Recommended future controls:
- Structure-only Student without KD.
- Fixed teacher-strategy KD.
- Learned-policy RC with matched training settings.
- Multiple seeds and statistical variability analysis.
