# Experiment K - Full-data Student Ablation

## Setup

- Dataset: FB15K-237
- GPU: NVIDIA RTX 5090
- Teacher: Pretrained MIN
- Training examples: 544,230 (including reciprocal triples)
- Batch size: 16
- Epochs: 5
- Best checkpoint: Epoch 4 for both runs
- Student optimizer: Adagrad, LR 0.1
- K2 policy optimizer: Adam, LR 1e-5
- Validation during training: 1,024 queries per direction
- Final evaluation: Full filtered validation and test

## Methods

- K1: Description-only teacher with NDKD
- K2: Learned RC with NDKD and REINFORCE

## Full Validation

| Metric | K1 | K2 |
|---|---:|---:|
| MR | 115.219845 | 143.361359 |
| MRR | 0.359133 | 0.353590 |
| Hits@1 | 0.264015 | 0.259338 |
| Hits@3 | 0.394668 | 0.387111 |
| Hits@10 | 0.550157 | 0.544454 |

## Full Test

| Metric | Paper DSoM | K1 | K2 |
|---|---:|---:|---:|
| MR | 116 | 119.167278 | 154.851830 |
| Hits@1 | 0.275 | 0.257305 | 0.251637 |
| Hits@3 | 0.411 | 0.389011 | 0.379507 |
| Hits@10 | 0.564 | 0.546663 | 0.537648 |

Test MRR:
- K1: 0.353423
- K2: 0.345857

## Runtime

- K1: 478.88 sec/epoch (~39.9 min total)
- K2: 603.36 sec/epoch (~50.3 min total)
- Full evaluation: ~7 seconds for both models and splits

## Findings

1. K1 outperformed K2 on all full validation and test metrics.
2. K1 was approximately 26% faster than K2 during training.
3. Neither run matched all published Table 3 metrics.
4. Both experiments used one seed and five epochs.
5. Best checkpoints were selected using sampled validation MRR.
6. The experiments differ in teacher selection and RL optimization;
   they do not isolate the causal impact of RC policy collapse.

## Next Steps

1. Audit K2 RC strategy distribution and policy entropy.
2. Compare selected strategy CE against oracle strategy CE.
3. Benchmark larger batch sizes.
4. Verify the paper's training and evaluation protocol.
5. Run additional seeds before drawing general conclusions.
