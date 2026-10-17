# RC Policy Optimizer Ablation — FB15K-237

## Objective

Investigate RC policy saturation during DSoM student training.

## Setup

- Teacher: FB15K-237 pretrained MIN checkpoint
- Student optimizer: Adagrad, LR=0.1
- Training examples: 2048
- Batch size: 16
- Epochs: 1
- Evaluation queries: 64
- Policy network: 43623 -> 1024 -> 7
- KD temperature: 4
- RC reward: +1 / -10

## Fixed-Batch Probe (Batch 3)

| Metric | A: Adagrad 1e-3 | B: Adam 1e-4 | C: Adam 1e-3 | D: Adam 1e-5 |
|---|---:|---:|---:|---:|
| Entropy after update | 0.000378 | 0.845803 | 0.010450 | 1.745250 |
| Hidden RMS after update | 21.264126 | 1.627662 | 12.482269 | 0.842773 |
| Logit range after update | 213.857391 | 13.630247 | 120.607147 | 2.191428 |
| Batches with nonzero policy gradient | 22 | 34 | 19 | 31 |
| Batches with updated fc2 weights | 16 | 125 | 125 | 125 |

All four experiments had the same pre-update probe:
- Entropy: 1.857928
- Hidden RMS: 0.829591
- Logit range: 1.246729

## Student Results (Smoke Tests)

| Metric | B: Adam 1e-4 | C: Adam 1e-3 | D: Adam 1e-5 |
|---|---:|---:|---:|
| Valid MRR | 0.1572 | 0.1765 | 0.1617 |
| Valid Hits@10 | 0.2344 | 0.2500 | 0.2578 |
| Test MRR | 0.1138 | 0.1219 | 0.1223 |
| Test Hits@10 | 0.1641 | 0.1797 | 0.1797 |

## Findings

1. The Adagrad policy at LR=1e-3 experiences severe saturation
   immediately after its first substantial optimizer update.
2. Adam at LR=1e-3 also causes rapid saturation.
3. Adam at LR=1e-4 reduces the initial activation explosion,
   but the policy approaches deterministic behavior later.
4. Adam at LR=1e-5 offers the strongest initial stability
   among the tested configurations.
5. Higher policy entropy does not guarantee better student MRR.

## Limitations

- Only 2048 training examples and one epoch were used.
- Evaluation was restricted to 64 queries per side.
- These results are not directly comparable to DSoM Table 3.
- The policy optimizer experiments modify the reproduction
  implementation and are not claimed to match the original paper.
- Longer training is needed to assess sustained stability.

## Next Experiment

Adam policy LR=1e-5, 20000 training examples, 3 epochs,
with expanded validation evaluation.
