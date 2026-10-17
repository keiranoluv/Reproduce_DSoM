# Experiment L9 - Fixed Structural+Description Student

## Objective

Evaluate whether a fixed Structural+Description
teacher strategy improves Student distillation
compared with Experiment K baselines.

## Dataset

FB15K-237

## Teacher

Frozen pretrained MIN checkpoint:

runs/teacher/FB15K-237/full/20261017_000249/checkpoints/0/m-20261017_0002-n-teacher_fb15k237_full_20261017_000249.pth

## Configuration

- Teacher strategy: Structural+Description (0.5, 0, 0.5)
- Student: unimodal ComplExMDR
- Distillation: fixed_ndkd
- Training examples: 544230 reciprocal queries
- Epochs: 5
- Batch size: 16
- Rank: 2000
- Optimizer: Adagrad
- Learning rate: 0.1
- Regularizer: N3
- Regularization weight: 0.01
- CE weight: 1
- KD weight: 2
- Temperature: 4
- Scheduler: ReduceLROnPlateau, mode=min

## Full Filtered Evaluation

| Split | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|---:|
| Validation | 112.2216 | 0.362118 | 0.264271 | 0.399829 | 0.558968 |
| Test | 117.0304 | 0.357728 | 0.260945 | 0.393873 | 0.554481 |

## Comparison with Experiment K - Test

| Student | MR | MRR | Hits@1 | Hits@10 |
|---|---:|---:|---:|---:|
| K1 - Description | 119.17 | 0.35342 | 0.25731 | 0.54666 |
| K2 - Learned RC | 154.85 | 0.34586 | 0.25164 | 0.53765 |
| L9 - Structural+Description | 117.03 | 0.357728 | 0.260945 | 0.554481 |

## Comparison with DSoM Table 3 - Test

| Metric | Paper | L9 | Difference |
|---|---:|---:|---:|
| MR | 116 | 117.0304 | +1.0304 |
| Hits@1 | 0.275 | 0.260945 | -0.014055 |
| Hits@3 | 0.411 | 0.393873 | -0.017127 |
| Hits@10 | 0.564 | 0.554481 | -0.009519 |

## Findings

1. L9 improves Test MRR, Hits@1, and Hits@10
   over both K1 and K2 in the recorded runs.

2. A fixed Structural+Description strategy produces
   stronger Student results than the evaluated
   learned-RC configuration.

3. L9 approaches the paper's Table 3 metrics,
   but is not a faithful reproduction of DSoM RC.

## Limitations

- Single run per compared configuration.
- No multi-seed confidence intervals.
- Five training epochs are not yet verified
  against the complete paper protocol.
- The selected strategy is fixed, not learned.
- The Teacher checkpoint and modalities
  still require protocol-level verification.

## Conclusion

Close the L1-L9 diagnostic and ablation branch.

Next: return to faithful reproduction of
DSoM Table 3, focusing on code-paper alignment.
