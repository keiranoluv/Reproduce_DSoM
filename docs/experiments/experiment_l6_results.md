# Experiment L6 - Teacher Generalization Gap Audit

## Objective

Measure teacher strategy quality on Train, Validation,
and Test using the same frozen Teacher MIN checkpoint.

## Setup

Dataset: FB15K-237
Evaluation: full dataset
Teacher: pretrained MIN, frozen
Queries: original + reciprocal
Metric: unfiltered cross-entropy (CE)

Queries:
- Train: 544230
- Validation: 35070
- Test: 40932

## Mean Cross-Entropy

| Strategy | Train | Valid | Test |
|---|---:|---:|---:|
| Structural | 2.6913 | 6.3162 | 6.3383 |
| Image | 6.1023 | 6.4193 | 6.4497 |
| Description | 4.6177 | 4.9181 | 4.9370 |
| Structural+Image | 3.6250 | 5.7175 | 5.7435 |
| Structural+Description | 3.2194 | 5.2440 | 5.2609 |
| Image+Description | 4.9135 | 5.2477 | 5.2702 |
| All three | 3.7595 | 5.2626 | 5.2836 |

## CE Oracle Strategy Distribution (%)

| Strategy | Train | Valid | Test |
|---|---:|---:|---:|
| Structural | 73.74 | 2.44 | 2.42 |
| Image | 0.89 | 5.32 | 5.37 |
| Description | 2.01 | 50.29 | 50.30 |
| Structural+Image | 9.35 | 4.01 | 3.92 |
| Structural+Description | 10.69 | 11.59 | 11.83 |
| Image+Description | 1.34 | 21.91 | 21.21 |
| All three | 1.99 | 4.44 | 4.95 |

## Train-to-Validation CE Gap

| Strategy | Gap |
|---|---:|
| Structural | +3.6249 |
| Image | +0.3170 |
| Description | +0.3004 |
| Structural+Image | +2.0925 |
| Structural+Description | +2.0247 |
| Image+Description | +0.3342 |
| All three | +1.5031 |

## Pairwise Comparison

Structural has lower CE than Description:
- Train: 92.48%
- Validation: 9.69%
- Test: 9.62%

Description has lower CE than Structural:
- Train: 7.52%
- Validation: 90.31%
- Test: 90.38%

## Interpretation

The frozen Teacher exhibits a substantial
train-to-validation CE gap for Structural predictions.

The relative quality of Structural and Description
reverses between training and unseen triples.

This provides evidence for a teacher-quality mismatch
that can contribute to RC preference for Structural
during training and poor teacher selection on validation.

## Limitations

- CE is unfiltered; it is not the filtered ranking metric
  reported in DSoM Table 3.
- The findings do not prove the cause of overfitting.
- They do not establish that an alternative RC strategy
  improves Student MRR or Hits@K.

## Next Experiment

L7: Compare teacher ranking quality, prediction confidence,
and temperature sensitivity on held-out data.
