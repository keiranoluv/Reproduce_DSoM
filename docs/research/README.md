# Báo cáo và hồ sơ nghiên cứu

Thư mục này lưu báo cáo bảng, phân tích bổ sung, cấu hình tham chiếu và provenance. Đây là tài liệu nghiên cứu, không phải hướng dẫn setup hoặc chạy workflow.

## Báo cáo theo bảng

- Table 1: [MKG-Y/MKG-W](tables/TABLE1_MKGY_MKGW_EXP01_RESULTS.md), [DB15K và sensitivity](tables/DB15K_CLEAN_SPLIT_SENSITIVITY.md).
- Table 2: [teacher selection](tables/TABLE2_REPRODUCTION.md).
- Table 3: [FB15K-237/WN18](tables/TABLE3_FB15K237_WN18_SOURCE_FROZEN_RC.md).
- Table 4: [component ablations](tables/TABLE4_COMPONENT_ABLATIONS.md), [teacher ensembles](tables/TABLE4_TEACHER_ENSEMBLES.md).
- Table 6: [modality ablation](tables/TABLE6_REPRODUCTION.md).
- Table 7: [WN9](tables/TABLE7_WN9_REPRODUCTION.md).
- Snapshot JSON và targets được giải thích trong [chỉ mục dữ liệu bảng](tables/reference/README.md).

## Phân tích và protocol

- [Phân tích chẩn đoán và các thí nghiệm bổ sung](analysis/README.md) — policy, ablation và các lượt thử nghiệm ngoài bảng chính.
- [Protocol và giới hạn](methodology/PROTOCOL.md) — cấu hình, khác biệt dữ liệu và các lưu ý khi diễn giải.

## Provenance

- [Chỉ mục provenance](provenance/README.md) và [audit nguồn WN9](provenance/WN9_SOURCE_AUDIT.md).
- [DB15K source trace](provenance/db15k_source_trace.json), [WN9 source trace](provenance/wn9_source_trace.json) và [Table 3 code audit](provenance/table3_code_audit.txt).

Các handoff và hướng dẫn đã bị thay thế được giữ trong [archive](../archive/README.md); không dùng chúng thay cho hướng dẫn chạy hiện tại.
