# Experiment L3 - Continuous CE Reward Ablation

## Setup

Dataset: FB15K-237
Training examples: 2048
Epochs: 1
Batch size: 16
Training steps: 128
Policy optimizer: Adam, LR 1e-5

Configurations:
- L0: Binary reward, raw RC input
- L2: Binary reward, normalized RC input
- L3a: Continuous CE reward, raw RC input
- L3b: Continuous CE reward, normalized RC input

Continuous reward:
R(s) = CE(Student) - CE(Teacher_s)

Advantage:
A(s) = CE(Teacher_average) - CE(Teacher_s)

## Training Diagnostics

| Metric | L0 | L2 | L3a | L3b |
|---|---:|---:|---:|---:|
| Final entropy | 0.169272 | 0.401179 | 0.056885 | 0.227821 |
| Final max probability | 0.944511 | 0.861394 | 0.981713 | 0.911043 |
| Final logit range | 17.1699 | 10.5665 | 21.9802 | 12.7707 |
| Nonzero advantage | 1.81% | 1.42% | 99.76% | 99.12% |

## Matched Validation Audit (1024 queries)

| Metric | L0 | L2 | L3a | L3b |
|---|---:|---:|---:|---:|
| Policy entropy | 0.140282 | 0.424915 | 0.022224 | 0.109288 |
| Oracle agreement | 9.6680% | 10.9375% | 2.6367% | 2.7344% |
| Policy CE | 5.539306 | 5.647519 | 6.233175 | 6.223771 |
| CE regret | 0.887890 | 0.996103 | 1.581759 | 1.572355 |
| Structural selection | 29.20% | 47.36% | 96.97% | 95.31% |

## Policy Gradient Diagnostics

Gradient norm measured on RC MLP fc2.weight.

| Metric | L0 | L2 | L3a | L3b |
|---|---:|---:|---:|---:|
| Mean gradient | 2.8690 | 2.0686 | 1.1808 | 1.1928 |
| Median gradient | 0.0000 | 0.0000 | 0.2814 | 0.6525 |
| Max gradient | 38.6851 | 24.4666 | 18.8057 | 9.5310 |
| Nonzero-gradient batches | 24.22% | 19.53% | 100% | 100% |

## Findings

1. Continuous CE reward nearly eliminates advantage sparsity.
2. L3a and L3b concentrate strongly on Structural.
3. Both L3 variants have higher CE regret than their binary baselines.
4. Continuous reward has lower mean fc2 gradient norm.
5. Input normalization reduces concentration in both reward settings.

## Limitations

- Single run per configuration.
- Only 128 optimizer steps.
- Evaluation uses 1024 sampled validation queries.
- Reward scale and policy learning rate are not separately controlled.
- Findings do not establish the cause of policy collapse.
- No full-data link-prediction results for L3.

## Next Experiment

L4: Policy learning-rate ablation with continuous CE reward.
Compare policy_lr=1e-5 against 1e-6.
