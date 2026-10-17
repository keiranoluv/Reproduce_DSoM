# Experiment J — Fixed Strategy vs Learned RC

Dataset: FB15K-237

Configuration:
- 20,000 training examples
- 3 epochs
- Batch size 16
- Student: Adagrad, LR 0.1
- RC policy: Adam, LR 1e-5
- Evaluation: 1,024 queries per split
- Checkpoints selected by validation MRR

| Run | Valid MRR | Valid H@10 | Test MRR | Test H@10 |
|---|---:|---:|---:|---:|
| J0 Structural | 0.250256 | 0.399414 | 0.250696 | 0.393066 |
| J1 Description | 0.264426 | 0.408203 | 0.261604 | 0.416016 |
| J2 Average | 0.249199 | 0.387207 | 0.249323 | 0.391113 |
| J3 Learned RC | 0.257907 | 0.407715 | 0.263478 | 0.409668 |

## Findings

Description-only achieved the highest validation MRR.

Learned RC outperformed Structural-only on validation despite
previous diagnostics showing Structural-only policy collapse.

This suggests that final policy decisions alone do not fully
characterize the effect of dynamic teacher selection during training.

## Limitations

Results are from a medium-run with 20,000 examples, 3 epochs,
and sampled evaluation queries.

Initialization and minibatch-order equivalence should be verified.

No claims about full Table 3 reproduction can be made yet.
