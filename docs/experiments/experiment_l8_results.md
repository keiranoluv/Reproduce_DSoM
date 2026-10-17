# Experiment L8 - Filtered Teacher Ranking Audit

## Objective

Evaluate all seven frozen Teacher strategies using
the filtered ranking protocol on FB15K-237.

## Setup

- Dataset: FB15K-237
- Teacher: frozen pretrained MIN
- Split: full validation
- Original triples: 17535
- Evaluation queries: 35070 (RHS + LHS)
- Protocol: filtered ranking using dataset.to_skip

## Results - Both Directions

| Strategy | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|---:|
| Structural | 152.03 | 0.3643 | 0.2714 | 0.3980 | 0.5515 |
| Image | 960.54 | 0.2193 | 0.1641 | 0.2392 | 0.3284 |
| Description | 152.32 | 0.3394 | 0.2566 | 0.3681 | 0.5045 |
| Structural+Image | 197.61 | 0.3205 | 0.2357 | 0.3513 | 0.4889 |
| Structural+Description | 110.80 | 0.3790 | 0.2855 | 0.4159 | 0.5663 |
| Image+Description | 193.01 | 0.3098 | 0.2329 | 0.3366 | 0.4606 |
| All three | 130.62 | 0.3498 | 0.2618 | 0.3825 | 0.5246 |

## Findings

1. Structural+Description achieves the best filtered
   MR, MRR, Hits@1, Hits@3 and Hits@10.
2. Structural has higher filtered MRR than Description.
3. This contrasts with L6/L7 unfiltered CE and MRR,
   where Description performs better.
4. The distinction between filtered and unfiltered
   evaluation materially affects teacher comparisons.

## Limitations

- Results evaluate Teacher strategies, not Student.
- Better Teacher filtered MRR does not guarantee
  better performance after distillation.
- Single pretrained Teacher checkpoint.

## Next Experiment

L9: Train a Student with fixed Structural+Description
and compare against K1 Description and K2 RC.
