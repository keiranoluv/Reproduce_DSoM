# Tiến độ reproduction DSoM

Ghi chép diagnostics FB15K-237 lịch sử; nhãn pending/running bên dưới thuộc
thời điểm thí nghiệm. Workflow hiện tại ở [README](../../../README.md).
Cập nhật 2026-10-24: full WN9 Table 7 hoàn tất, artifact review PASS;
xem [báo cáo](../../experiments/TABLE7_WN9_REPRODUCTION.md).

Dataset: FB15K-237
GPU: NVIDIA RTX 5090 (32 GB)
Nhánh: experiment/fixed-strategy-ablation

## 1. Mục tiêu

Reproduce DSoM Table 3 trên FB15K-237.

Student targets paper:
- MR: 116
- Hits@1: 27.5%
- Hits@3: 41.1%
- Hits@10: 56.4%

Pipeline:
Pretrained Teacher MIN -> Reinforced Curriculum (RC) -> Student.

## 2. Training teacher

Training Teacher MIN hoàn tất thành công.

Teacher results quan sát:
- MRR: 0.4334
- Hits@10: 62.40%

Checkpoint:
runs/teacher/FB15K-237/full/20261017_000249/

Teacher metrics không so trực tiếp với student results Table 3.

## 3. Thí nghiệm A-I: chẩn đoán RC

Quan sát RC policy collapse khi train student.

Learned policy cuối chọn Structural-only cho 100% validation queries được audit.

Phát hiện chính:
- Adam policy LR 1e-5 trì hoãn early saturation.
- Training dài hơn vẫn collapse.
- Structural-only là lowest-CE strategy cho chỉ 2.64% validation queries được audit.
- Description thường có teacher CE tốt hơn.
- Binary rewards thường không phân biệt strategies.
- Early updates chủ yếu có advantage âm.

Kết luận: có policy collapse nhưng chưa xác lập chính xác nguyên nhân.

## 4. Thí nghiệm J: ablation fixed teacher

Cấu hình:
- 20,000 training examples
- 3 epochs
- Batch size 16
- 1,024 evaluation queries

Kết quả tại best validation-MRR checkpoint:

| Chiến lược | Valid MRR | Test MRR |
|---|---:|---:|
| J0 Structural | 0.2503 | 0.2507 |
| J1 Description | 0.2644 | 0.2616 |
| J2 Average | 0.2492 | 0.2493 |
| J3 Learned RC | 0.2579 | 0.2635 |

Phát hiện:
- Description-only có validation MRR tốt nhất.
- Learned RC hơn fixed Structural-only.
- Final collapse không có nghĩa student chỉ học từ Structural.

Kết quả sơ bộ từ một medium run.

## 5. Thí nghiệm K: pilot full data

Thí nghiệm tại thời điểm viết:
- K1: Description-only
- K2: Learned RC
- Mẫu training: 544,230
- Batch size: 16
- Tối đa epochs: 5
- Queries evaluation: 1,024
- K1/K2 chạy tuần tự trong tmux.

Mục đích: kiểm tra phát hiện J còn đúng khi train full FB15K-237 không.

Trạng thái tại thời điểm viết:
Đang chạy / kết quả còn chờ.

Logs:
runs/student/FB15K-237/experiment_k/

## 6. Quan sát chi phí training

Quan sát RTX 5090 khi train student:
- VRAM: khoảng 3.3 GB
- GPU utilization: khoảng 88%
- Power: khoảng 427 W

Với batch 16:
- 544,230 examples/epoch
- Khoảng 34,015 optimizer steps/epoch
- Ước lượng 8-13 phút/epoch tùy strategy.

Code gốc mặc định tối đa 200 epochs, batch 1000; cấu hình chính xác dùng
cho paper Table 3 vẫn cần xác minh.

Lưu ý: ít VRAM sử dụng không có nghĩa GPU còn nhàn rỗi về tính toán.

## 7. Các bước tiếp theo

1. Hoàn tất K, so validation.
2. Audit K2 collapse trên full data.
3. Benchmark batches 128, 256, 512, 1000.
4. Đo throughput/peak memory.
5. Xác minh paper hyperparameters.
6. Full training/evaluation Table 3.

Chưa đổi RC algorithm trước khi thiết lập baseline reproduction đáng tin cậy.

## 8. Kết luận tại thời điểm viết

Teacher training hoàn tất.

Quan sát RC collapse trong medium experiments.

Description-only là baseline mạnh, nhưng full-data results còn chờ tại thời điểm viết.

Ưu tiên lúc này: reproduction đúng, training hiệu quả và Table 3 evaluation.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="1-goal"></a>
<a id="2-teacher-training"></a>
<a id="3-experiments-a-i-rc-diagnostics"></a>
<a id="4-experiment-j-fixed-teacher-ablation"></a>
<a id="5-experiment-k-full-data-pilot"></a>
<a id="6-training-cost-observation"></a>
<a id="7-next-steps"></a>
<a id="8-current-conclusion"></a>
<a id="dsom-reproduction-progress"></a>
