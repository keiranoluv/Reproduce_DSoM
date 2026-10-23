# DSoM Reproduction Progress

Historical FB15K-237 diagnostic notes; pending/running labels below refer to
that experiment period. Current workflows are in [README](../../README.md).
Update 2026-10-24: the user's WN9 Table 7 full DSoM row completed and passed
artifact review; see the [result record](TABLE7_WN9_REPRODUCTION.md).

Dataset: FB15K-237
GPU: NVIDIA RTX 5090 (32 GB)
Branch: experiment/fixed-strategy-ablation

## 1. Goal

Reproduce DSoM Table 3 on FB15K-237.

Paper Student targets:
- MR: 116
- Hits@1: 27.5%
- Hits@3: 41.1%
- Hits@10: 56.4%

Pipeline:
Pretrained Teacher MIN -> Reinforced Curriculum (RC) -> Student.

## 2. Teacher Training

Teacher MIN training completed successfully.

Observed teacher results:
- MRR: 0.4334
- Hits@10: 62.40%

Checkpoint:
runs/teacher/FB15K-237/full/20261017_000249/

These teacher metrics are not directly comparable to
the Student results in Table 3.

## 3. Experiments A-I: RC Diagnostics

During Student training, we observed RC policy collapse.

The learned policy eventually selected Structural-only
for 100% of the audited validation queries.

Key findings:
- Adam policy LR 1e-5 delayed early saturation.
- Policy collapse still occurred during longer training.
- Structural-only was the lowest-CE strategy for only
  2.64% of the audited validation queries.
- Description frequently provided better teacher CE.
- Binary rewards often failed to distinguish strategies.
- Early policy updates included mostly negative advantages.

Conclusion:
Policy collapse was observed, but its exact cause
has not been conclusively established.

## 4. Experiment J: Fixed Teacher Ablation

Configuration:
- 20,000 training examples
- 3 epochs
- Batch size 16
- 1,024 evaluation queries

Results at best validation-MRR checkpoint:

| Strategy | Valid MRR | Test MRR |
|---|---:|---:|
| J0 Structural | 0.2503 | 0.2507 |
| J1 Description | 0.2644 | 0.2616 |
| J2 Average | 0.2492 | 0.2493 |
| J3 Learned RC | 0.2579 | 0.2635 |

Findings:
- Description-only achieved the best validation MRR.
- Learned RC outperformed fixed Structural-only.
- Final policy collapse does not necessarily imply
  that the Student learned only from Structural.

These are preliminary results from one medium-scale run.

## 5. Experiment K: Full-data Pilot

Current experiment:
- K1: Description-only
- K2: Learned RC
- Training examples: 544,230
- Batch size: 16
- Maximum epochs: 5
- Evaluation queries: 1,024
- K1 and K2 run sequentially in tmux.

Purpose:
Check whether Experiment J findings persist
when training on the full FB15K-237 dataset.

Status:
Running / results pending.

Logs:
runs/student/FB15K-237/experiment_k/

## 6. Training Cost Observation

RTX 5090 usage observed during Student training:
- VRAM: approximately 3.3 GB
- GPU utilization: approximately 88%
- Power: approximately 427 W

With batch size 16:
- 544,230 examples per epoch
- Approximately 34,015 optimizer steps per epoch
- Estimated 8-13 minutes per epoch depending on strategy.

The original code has a default maximum of 200 epochs
and batch size 1000, but the exact configuration used
for the paper's Table 3 still needs verification.

Important:
Low VRAM usage does not mean the GPU has idle compute.

## 7. Next Steps

1. Finish Experiment K and compare validation results.
2. Audit K2 policy for collapse on full data.
3. Benchmark batch sizes: 128, 256, 512, 1000.
4. Measure throughput and peak GPU memory.
5. Verify the original paper's hyperparameters.
6. Run full training and full evaluation for Table 3.

Do not change the RC algorithm before establishing
a reliable reproduction baseline.

## 8. Current Conclusion

Teacher training is complete.

RC collapse was observed in medium-scale experiments.

Description-only is a strong Student baseline,
but full-data results are still pending.

The priority is now reproduction correctness,
training efficiency, and Table 3 evaluation.
