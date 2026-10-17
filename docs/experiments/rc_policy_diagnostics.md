# DSoM RC Policy Diagnostics — FB15K-237

## Scope

These experiments investigate RC policy behavior during student
distillation. They are diagnostics and ablations, not claims of
faithful reproduction of every detail in the original paper.

Baseline branch: student-pipeline-fixes

Experimental branch: experiment/rc-policy-stabilization

Teacher: FB15K-237 pretrained MIN checkpoint.

## Experiment E — Medium Training

Configuration:

- Student optimizer: Adagrad, learning rate 0.1
- Policy optimizer: Adam, learning rate 1e-5
- Training examples: 20,000
- Batch size: 16
- Epochs: 3
- Evaluation queries: 1,024

Results:

| Epoch | Valid MRR | Test MRR | Test Hits@10 |
|---|---:|---:|---:|
| 1 | 0.238054 | 0.226723 | 0.367188 |
| 2 | 0.225990 | 0.241456 | 0.380859 |
| 3 | 0.257907 | 0.263478 | 0.409668 |

Policy entropy at the final minibatch:

| Epoch | Entropy | Near-deterministic |
|---|---:|---:|
| 1 | 0.081332 | 13/16 |
| 2 | 0.000929 | 15/16 |
| 3 | 0.000003 | 16/16 |

Interpretation: Student performance improved, but policy
became nearly deterministic.

## Experiment F — Strategy Quality Audit

Checkpoint: Experiment E best distiller.

Validation queries: 1,024, including reciprocal queries.
Policy decision: argmax.

- Policy chose Structural-only: 100%
- Policy-oracle agreement: 2.6367%
- Student mean CE: 6.519696
- Policy mean CE: 6.272866
- Oracle mean CE: 4.651416
- Policy mean CE regret: 1.621450

Oracle strategy distribution:

| Strategy | Frequency |
|---|---:|
| Structural | 2.64% |
| Image | 5.47% |
| Description | 50.78% |
| Structural+Image | 3.71% |
| Structural+Description | 12.11% |
| Image+Description | 20.61% |
| All three | 4.69% |

The oracle uses ground-truth labels and is diagnostic only.

## Experiment G — Reward Signal Audit

Same checkpoint and validation setup as Experiment F.

Reward: +1 when strategy CE < student CE, otherwise -10.
Baseline: reward from mean of all three teacher modalities.

| Strategy | Mean CE | Positive reward |
|---|---:|---:|
| Structural | 6.2729 | 55.76% |
| Image | 6.3664 | 55.47% |
| Description | 4.9021 | 88.87% |
| Structural+Image | 5.6744 | 74.41% |
| Structural+Description | 5.2009 | 85.64% |
| Image+Description | 5.2135 | 85.16% |
| All three | 5.2242 | 86.52% |

Additional findings:

- Identical reward across all strategies: 43.55% of queries
- At least two strategies receive +1: 91.89%
- Mean CE gap among positively rewarded strategies: 1.411396
- Policy reward worse than best available reward: 38.09%

These observations suggest limited reward discrimination.
They do not by themselves establish the cause of policy collapse.

## Experiment H — Update Direction Audit

Student: Adagrad 0.1.
Policy: Adam 1e-5.
Training: 2,048 examples, one epoch.

Batch 3:
- One negative advantage
- P(Description) change: -0.091313
- P(Structural) change: +0.002867

Batch 8:
- Two negative advantages
- P(Image) change: -0.108906
- P(Structural) change: +0.039095

Probabilities were compared on the same minibatch before
and after optimizer updates, with policy dropout disabled.

## Experiment I — Advantage Attribution

First 20 batches, 320 examples:

- Positive advantage samples: 0
- Negative advantage samples: 7
- Zero advantage samples: 313

Negative advantage actions:

- Image: 5
- Description: 1
- Image+Description: 1

Batch 3: Description received advantage -11.
Batch 8: Two Image selections each received advantage -11.

These results are consistent with REINFORCE penalizing the
sampled strategies. They do not prove that the early updates
alone caused the final Structural-only policy.

## Limitations

- Experiment E used only 20,000 training examples and 3 epochs.
- Evaluation and audits used sampled query subsets.
- Experiments F/G evaluated the trained policy on validation.
- The CE oracle requires ground-truth labels.
- Experiments H/I measured only the first 20 batches.
- No conclusion about full Table 3 reproduction follows yet.

## Next: Experiment J

Compare student distillation under fixed strategies:

- J0: Structural-only
- J1: Description-only
- J2: Average of all modalities
- J3: Learned RC (Adam 1e-5)

Keep the same student initialization, training subset,
optimizer, NDKD configuration, epochs, and evaluation
protocol to the extent possible.

Use validation metrics for model comparison.
Do not select experimental configurations using test metrics.
