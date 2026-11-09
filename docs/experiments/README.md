# Kết quả nghiên cứu và provenance

Khu vực này lưu cấu hình, source provenance, kết quả và giới hạn khoa học.
Để chạy mới, dùng [RUNNING.md](../reproduction/RUNNING.md); lệnh trong các báo cáo
lịch sử chỉ mô tả thí nghiệm cũ, không phải luồng setup thay thế.
Baseline `source_frozen` và các learned-RC adaptations phải được báo cáo riêng.

## Báo cáo chính

- Table 1: [MKG-Y/MKG-W](TABLE1_MKGY_MKGW_EXP01_RESULTS.md);
  DB15K có trong [báo cáo baseline/sensitivity](DB15K_CLEAN_SPLIT_SENSITIVITY.md).
- Table 2: [teacher selection và MulDE adaptation](TABLE2_REPRODUCTION.md).
- Table 3: [FB15K-237/WN18](TABLE3_FB15K237_WN18_SOURCE_FROZEN_RC.md).
- Table 4: [component students](TABLE4_COMPONENT_ABLATIONS.md),
  [teacher ensembles](TABLE4_TEACHER_ENSEMBLES.md).
- Table 6: [modality ablation](TABLE6_REPRODUCTION.md),
  [metric snapshot và hashes](table6_results_20261023.json).
- Table 7: [WN9 result/protocol](TABLE7_WN9_REPRODUCTION.md),
  [metric snapshot và hashes](table7_results_20261024.json).

Chỉ Table 7 tự reload selected student checkpoint để evaluation độc lập trong
workflow; các bảng khác phải được diễn giải theo scope verification đã ghi nhận.
Các artifacts `runs/` không được đóng gói vào fresh clone.

## Nguồn dữ liệu và method audit

[DB15K contamination/CleanV1](DB15K_CLEAN_SPLIT_SENSITIVITY.md) ·
[WN9 source và feature reuse](WN9_SOURCE_AUDIT.md) ·
[DB15K policy analysis](db15k_rc_policy_analysis.md).
Nguồn-trace metadata: [DB15K](../audits/db15k_source_trace.json),
[WN9](../audits/wn9_source_trace.json); [Table 3 code audit](../audits/table3_code_audit.txt).

## Các thí nghiệm chẩn đoán

Các báo cáo dưới đây vẫn có giá trị nghiên cứu, không bị deprecated chỉ vì được
viết trước các full-table workflows. Chúng không chứng minh mọi adaptation là
protocol tác giả hay thay thế các main baseline results.

- RC: [optimizer ablation](rc_policy_optimizer_ablation.md),
  [diagnostics E–I](rc_policy_diagnostics.md),
  [first-step saturation](table3_first_step_policy_saturation.md).
- [J: fixed strategy](fixed_strategy_ablation.md),
  [K: full-data pilot](experiment_k_results.md),
  [L0/L1: concentration](experiment_l0_l1.md).
- [L2: input normalization](experiment_l2_results.md),
  [L3: continuous reward](experiment_l3_results.md),
  [L4: policy LR](experiment_l4_results.md),
  [L5: counterfactual audit](experiment_l5_results.md).
- [L6: generalization gap](experiment_l6_results.md),
  [L7: calibration/ranking](experiment_l7_results.md),
  [L8: filtered ranking](experiment_l8_results.md),
  [L9: structural+description](experiment_l9_results.md).

Các handoff/setup bị thay thế nằm trong [deprecated](../deprecated/README.md).

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="diagnostic-experiments"></a>
