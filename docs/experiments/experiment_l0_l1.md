# Experiment L0/L1 - Early RC Policy Concentration

## Setup

Dataset: FB15K-237
Training examples: 2048
Batch size: 16
Optimizer steps: 128
Epochs: 1
Policy learning rate: 1e-5
Reward: +1 / -10

L0: Adam beta1=0.9
L1: Adam beta1=0.0

## Results

| Metric | L0 | L1 |
|---|---:|---:|
| Initial entropy | 1.864328 | 1.864328 |
| Final entropy | 0.169272 | 0.063810 |
| Final max probability | 0.944511 | 0.976755 |
| Zero-advantage batches | 97/128 | 90/128 |
| Zero-gradient weight updates | 94 | 0 |

L0 nonzero-advantage sample fraction: 1.8066%.

## MLP Diagnostics

| Metric | L0 batch 0 | L0 batch 127 | L1 batch 127 |
|---|---:|---:|---:|
| Teacher std | 2.1279 | 2.1016 | 2.1016 |
| Hidden RMS | 0.8562 | 1.8736 | 1.8483 |
| Policy logit range | 1.2292 | 17.1699 | 18.0931 |

## Interpretation

- Policy concentration occurs within 128 training steps.
- Removing Adam first-moment momentum does not prevent concentration.
- Teacher logit std stays relatively stable.
- RC policy logit range increases substantially.
- The source of logit amplification remains unconfirmed.

## Limitations

These are short diagnostic runs, not full-data experiments.
The change to beta1 also changes the training trajectory.
No causal explanation has yet been established.

## Next

Experiment L2: Normalize policy input while leaving
teacher logits used by reward and knowledge distillation unchanged.
