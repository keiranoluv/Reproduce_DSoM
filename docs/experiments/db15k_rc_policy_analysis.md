# DB15K — Audit RC policy A5/A9/A10

**Trạng thái:** Hoàn tất, exploratory, một seed (2022).

## Phát hiện chính

- Learned RC gần deterministic, Structural chiếm ưu thế.
- A5 chọn S+D thưa, tập trung `rdf:type` LHS queries.
- A5 cải thiện aggregate test teacher CE so Structural-only.
- Lợi thế đảo ngược khi so random phân tầng relation-direction.
- Chưa xác minh adaptation theo query hoặc lợi ích nhân quả với student MRR.

## Relation mapping đã xác minh

- Relation 275: `http://www.w3.org/1999/02/22-rdf-syntax-ns#type`.
- RDF `rdf:type` nối entity với class/type.
- RHS dự đoán type từ entity; LHS dự đoán entity từ type.
- Bất đối xứng hai chiều là giả thuyết, chưa giải thích nhân quả.

## Metrics của student checkpoint được chọn

| Experiment | Loss | Epoch | MRR | H@1 | H@3 | H@10 |
|---|---|---:|---:|---:|---:|---:|
| A5_NDKD | NDKD | 63 | 0.414221 | 0.335185 | 0.453444 | 0.564431 |
| A9_KL | Vanilla KD | 51 | 0.413372 | 0.334478 | 0.452282 | 0.565189 |
| A10_DKD | DKD | 60 | 0.413027 | 0.333367 | 0.452333 | 0.564936 |

## Hành vi policy theo split

Greedy argmax, tắt dropout.

| Model | Split | N | Structural | S+D | Entropy | Oracle match |
|---|---|---:|---:|---:|---:|---:|
| A5_NDKD | train | 158444 | 95.47% | 4.52% | 0.000211 | 80.36% |
| A5_NDKD | valid | 19808 | 96.17% | 3.83% | 0.000357 | 72.55% |
| A5_NDKD | test | 19804 | 95.22% | 4.77% | 0.001724 | 21.08% |
| A9_KL | train | 158444 | 99.85% | 0.15% | 0.000157 | 83.22% |
| A9_KL | valid | 19808 | 99.89% | 0.11% | 0.000428 | 74.56% |
| A9_KL | test | 19804 | 99.75% | 0.24% | 0.001687 | 21.17% |
| A10_DKD | train | 158444 | 99.88% | 0.12% | 0.000106 | 83.21% |
| A10_DKD | valid | 19808 | 99.93% | 0.07% | 0.000320 | 74.59% |
| A10_DKD | test | 19804 | 99.78% | 0.21% | 0.001612 | 21.16% |

## Phân phối CE oracle

Oracle dùng ground-truth labels, chỉ diagnostic.

| Split | Structural | Description | S+D |
|---|---:|---:|---:|
| train | 83.16% | 0.44% | 12.14% |
| valid | 74.58% | 2.92% | 14.19% |
| test | 21.24% | 22.35% | 17.43% |

## Phân tích đóng góp teacher CE của A5

CE difference âm nghĩa là cải thiện.

| Split | S+D rate | S+D precision | A5-S CE | S+D contribution | Random same-rate CE |
|---|---:|---:|---:|---:|---:|
| train | 4.52% | 18.97% | +0.006093 | +0.006379 | +0.039142 |
| valid | 3.83% | 28.63% | +0.002662 | +0.002662 | +0.023941 |
| test | 4.77% | 92.59% | -0.029112 | -0.028994 | -0.022886 |

## Phân rã hai chiều relation 275

860 trên 945 S+D choices thuộc `rdf:type`. **859 trên 945 thuộc reciprocal/LHS queries.**

| Group | Queries | S+D | Rate | Mean delta | Chosen delta | CE contribution |
|---|---:|---:|---:|---:|---:|---:|
| rdf:type RHS | 1345 | 1 | 0.07% | +0.4942 | -0.4137 | -0.000021 |
| rdf:type LHS | 1345 | 859 | 63.87% | -0.9486 | -0.8261 | -0.035830 |
| Other relations | 17114 | 85 | 0.50% | -0.5193 | +1.5976 | +0.006857 |
| All test | 19804 | 945 | 4.77% | -0.4796 | -0.6076 | -0.028994 |

Trong rdf:type/LHS, random S+D có mean CE benefit lớn hơn subset A5 chọn.

## Độ vững thống kê

Bootstrap ghép hai chiều reciprocal của unique original triples; conditional
permutations giữ selection rates theo relation và direction.

| Split | Unique triples | A5-S CE | Bootstrap 95% CI | Global gain | Conditional gain | One-sided p |
|---|---:|---:|---|---:|---:|---:|
| valid | 7481 | +0.003039 | [+0.001714, +0.004347] | +0.023925 | +0.007363 | 0.000500 |
| test | 9902 | -0.029112 | [-0.032290, -0.025867] | +0.006108 | -0.012748 | 1.000000 |

## Độ đồng thuận policy trên test

| Pair | Greedy agreement | Mean soft L1 |
|---|---:|---:|
| A5_NDKD_vs_A9_KL | 95.33% | 0.092779 |
| A5_NDKD_vs_A10_DKD | 95.29% | 0.093614 |
| A9_KL_vs_A10_DKD | 99.70% | 0.005541 |

## Giới hạn

- CE oracle cần nhãn.
- Teacher CE unfiltered, student MRR filtered.
- Teacher selection diễn ra khi train student, không khi student inference.
- DB15K validation có duplicates/split leakage.
- Tests exploratory/post-hoc.
- Một seed và features tái dựng.
- Learned RC khác released-code source-frozen RC.

## Các bước tiếp theo

- Verify NDKD neighbor/non-neighbor decomposition trước Table 4 ablations.
- Chạy lại thí nghiệm chính với thêm seeds.
- Đánh giá sensitivity với DB15K validation leakage.

## Nguồn gốc

- Policy audit: `runs/audit/db15k_rc_policy/20261021_194427_3194329`
- Selection attribution: `runs/audit/db15k_a5_selection/20261021_200624_3249374`
- Relation mapping: `data/MMKB-DB15K/rel_id`
- Sinh bởi `scripts/write_db15k_rc_report.py`.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="a5-teacher-ce-attribution"></a>
<a id="ce-oracle-distribution"></a>
<a id="db15k-rc-policy-audit--a5-a9-a10"></a>
<a id="limitations"></a>
<a id="main-findings"></a>
<a id="next-steps"></a>
<a id="policy-agreement-on-test"></a>
<a id="policy-behavior-by-split"></a>
<a id="provenance"></a>
<a id="relation-275-direction-decomposition"></a>
<a id="statistical-robustness"></a>
<a id="student-best-checkpoint-metrics"></a>
<a id="verified-relation-mapping"></a>
