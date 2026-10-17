# Experiment L4 - Policy Learning Rate Ablation

## Objective

Investigate whether reducing the RC policy learning rate
mitigates early policy concentration and improves
teacher strategy selection under continuous CE reward.

## Experimental Setup

Dataset: FB15K-237
Training examples: 2048
Epochs: 1
Batch size: 16
Optimizer steps: 128

Reward: Continuous CE
Policy optimizer: Adam
Student optimizer: Adagrad, LR 0.1

Configurations:
- L3a: Raw input, policy LR 1e-5
- L4a: Raw input, policy LR 1e-6
- L3b: Normalized input, policy LR 1e-5
- L4b: Normalized input, policy LR 1e-6

## Training Diagnostics

| Metric | L3a | L4a | L3b | L4b |
|---|---:|---:|---:|---:|
| Final entropy | 0.056885 | 0.401380 | 0.227821 | 0.650564 |
| Final max probability | 0.981713 | 0.852131 | 0.911043 | 0.785581 |
| Final logit range | 21.9802 | 7.0258 | 12.7707 | 5.2009 |
| Mean gradient norm | 1.1808 | 4.3228 | 1.1928 | 3.5271 |
| Nonzero advantage | 99.76% | 97.46% | 99.12% | 95.65% |

Gradient norm refers to RC MLP fc2.weight.

## Matched Validation Audit

1024 validation queries, policy argmax, seed 2022.

| Metric | L3a | L4a | L3b | L4b |
|---|---:|---:|---:|---:|
| Policy entropy | 0.022224 | 0.284409 | 0.109288 | 0.569393 |
| Oracle agreement | 2.6367% | 3.8086% | 2.7344% | 4.1992% |
| Policy CE | 6.233175 | 6.206656 | 6.223771 | 6.159985 |
| CE regret | 1.581759 | 1.555240 | 1.572355 | 1.508569 |
| Structural selection | 96.97% | 95.90% | 95.31% | 92.38% |

## Findings

1. Reducing policy LR from 1e-5 to 1e-6 increases
   policy entropy in both input configurations.
2. Policy logit range decreases substantially.
3. Oracle agreement improves slightly.
4. CE regret decreases slightly.
5. Structural remains the dominant argmax strategy,
   selected on over 92% of validation queries.
6. Lower policy LR alone does not resolve
   the observed strategy-selection mismatch.

## Limitations

- Single short training run per configuration.
- Only 128 optimizer steps.
- Evaluation uses 1024 sampled validation queries.
- Results do not establish the cause of policy collapse.
- No full-data link-prediction results for L4.

## Next Experiment

L5: Policy Gradient Direction Audit.

Investigate:
- Advantage distribution by selected strategy.
- Policy probability changes after optimizer steps.
- Teacher CE per strategy on training minibatches.
