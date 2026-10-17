# Experiment L2 - Normalized RC Policy Input

## Setup

Dataset: FB15K-237
Training examples: 2048
Epochs: 1
Batch size: 16
Optimizer steps: 128
Policy optimizer: Adam, LR 1e-5
Reward: original binary +1/-10
Policy temperature: 1

L0: raw teacher logits to RC MLP
L2: per-query standardized teacher logits to RC MLP

Teacher logits for reward and KD remain unchanged.

## Training Diagnostics

| Metric | L0 | L2 |
|---|---:|---:|
| Initial entropy | 1.864328 | 1.928094 |
| Final entropy | 0.169272 | 0.401179 |
| Final max probability | 0.944511 | 0.861394 |
| Final logit range | 17.169900 | 10.566500 |

## Matched 1024-Query Validation Audit

| Metric | L0 | L2 |
|---|---:|---:|
| Policy entropy | 0.140282 | 0.424915 |
| Policy-oracle agreement | 9.6680% | 10.9375% |
| Student CE | 7.139151 | 7.153788 |
| Policy CE | 5.539306 | 5.647519 |
| Oracle CE | 4.651416 | 4.651416 |
| Policy CE regret | 0.887890 | 0.996103 |
| Positive reward rate | 75.3906% | 77.4414% |

## Strategy Distribution

| Strategy | L0 | L2 |
|---|---:|---:|
| Structural | 29.20% | 47.36% |
| Image | 0.00% | 0.00% |
| Description | 1.95% | 0.39% |
| Structural+Image | 0.00% | 0.00% |
| Structural+Description | 67.19% | 45.41% |
| Image+Description | 1.66% | 6.84% |
| All three | 0.00% | 0.00% |

## Findings

1. Input normalization reduces early policy concentration.
2. L2 has higher entropy but worse mean CE regret than L0.
3. Neither policy frequently selects Description-only.
4. Higher policy diversity does not guarantee better selection quality.

## Limitations

- One short training run per configuration.
- Only 128 optimizer steps.
- Validation audit uses 1024 sampled queries.
- No full-data link-prediction comparison.
- Reward design remains unchanged.

## Next Experiment

L3a: raw RC input + continuous CE reward.
L3b: normalized RC input + continuous CE reward.
