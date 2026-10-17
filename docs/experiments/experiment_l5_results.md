# Experiment L5 - Counterfactual Teacher Strategy Audit

## Objective

Investigate why the RC policy selects Structural during
the first training steps of continuous CE reward training.

## Setup

- Dataset: FB15K-237
- Baseline: L3a (continuous CE reward, raw policy input)
- Training examples: 2048
- Batch size: 16
- Epochs: 1
- Total optimizer steps: 128
- Diagnostic batches: first 20
- Audited training queries: 320
- Teacher: frozen pretrained MIN

## Counterfactual Advantage

For each training query, evaluate all seven teacher strategies.

Advantage(strategy) =
    CE(average teacher) - CE(selected strategy)

Positive advantage means the strategy has lower CE than
the average-teacher baseline.

## Results

| Strategy | Mean CE | Mean Advantage | Oracle % | Sampled % |
|---|---:|---:|---:|---:|
| Structural | 2.7072 | +1.1887 | 79.38% | 80.94% |
| Image | 6.4623 | -2.5664 | 0.94% | 1.56% |
| Description | 4.7788 | -0.8829 | 0.94% | 4.69% |
| Structural+Image | 3.7623 | +0.1336 | 7.50% | 1.56% |
| Structural+Description | 3.2780 | +0.6179 | 7.50% | 3.75% |
| Image+Description | 5.1580 | -1.2622 | 0.31% | 5.00% |
| All three | 3.8959 | 0.0000 | 3.44% | 2.50% |

## Findings

1. Structural has the lowest mean teacher CE on the
   first 320 training queries.
2. Structural is the CE oracle for 79.38% of these queries.
3. The policy samples Structural on 80.94% of these queries.
4. Structural receives positive mean counterfactual advantage.
5. Early preference for Structural is consistent with the
   continuous CE reward signal observed on training queries.

## Comparison With Validation

In the earlier 1024-query validation audit:
- Structural oracle: 2.64%
- Description oracle: 50.78%

The difference suggests a possible train-validation
teacher-quality gap.

## Limitations

- Only the first 20 training batches were audited.
- Model states change across training steps.
- Validation audit uses a fixed checkpoint.
- This does not establish causality or prove overfitting.
- A frozen-teacher train/validation/test audit is needed.

## Next Step

L6: Teacher generalization gap audit using the same frozen
Teacher MIN checkpoint across train, validation and test.
