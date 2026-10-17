# Experiment L7 - Teacher Calibration and Ranking Audit

## Objective

Determine whether Structural's high validation CE
comes from poorer ranking, excessive confidence, or both.

## Setup

Dataset: FB15K-237
Teacher: frozen pretrained MIN
Split: full validation
Queries: 35070 (original + reciprocal)

Metrics:
- Unfiltered Cross-Entropy
- Unfiltered MRR and Hits@1
- Top-1 confidence
- Expected Calibration Error (ECE, 10 bins)
- Cross-Entropy at temperatures 1, 2 and 4

## Results

| Strategy | CE | MRR | Hits@1 | Confidence | ECE | CE T=2 | CE T=4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Structural | 6.3162 | 0.1681 | 0.1005 | 0.1959 | 0.1021 | 6.7524 | 8.0465 |
| Image | 6.4193 | 0.1586 | 0.1091 | 0.1088 | 0.0067 | 7.3782 | 8.3677 |
| Description | 4.9181 | 0.2081 | 0.1376 | 0.1414 | 0.0154 | 6.0726 | 7.6092 |
| Structural+Image | 5.7175 | 0.1628 | 0.1022 | 0.1509 | 0.0509 | 6.9337 | 8.1791 |
| Structural+Description | 5.2440 | 0.1765 | 0.1088 | 0.1749 | 0.0665 | 6.3178 | 7.8061 |
| Image+Description | 5.2477 | 0.1935 | 0.1278 | 0.1377 | 0.0156 | 6.5539 | 7.9476 |
| All three | 5.2626 | 0.1760 | 0.1111 | 0.1534 | 0.0466 | 6.5573 | 7.9675 |

## Findings

1. Description has lower validation CE than Structural.
2. Description also has higher unfiltered MRR and Hits@1.
3. Structural has greater top-1 confidence despite lower accuracy.
4. Structural has higher ECE than Description.
5. Increasing temperature from 1 to 2 or 4 worsens CE
   for both modalities.

## Interpretation

The Structural/Description gap cannot be explained
by confidence alone: ranking quality also differs.

The temperature experiment tests only T=1,2,4.
It does not rule out other calibration temperatures.

## Limitations

- Unfiltered ranking is not Table 3 filtered ranking.
- These metrics evaluate Teacher, not Student.
- ECE is sensitive to binning and top-1 accuracy.

## Next Experiment

L8: Full filtered ranking audit for all seven
teacher strategies using FB15K-237 to_skip filters.
