# Tiến độ FB15K-237 trước khi hoàn tất thí nghiệm K

Ghi chép này thuộc nhánh `experiment/fixed-strategy-ablation` trên RTX 5090 32 GB. Tại thời điểm viết, teacher đã train xong, J hoàn tất và K đang chạy. [Kết quả K](../experiments/k-full-data.md) và [baseline Table 3](../../research/tables/table-3-fb15k237-wn18.md) được lưu sau đó.

## Teacher và mục tiêu Table 3

Paper báo student MR 116, H@1 27.5%, H@3 41.1%, H@10 56.4%. Quy trình đang thử là pretrained MIN teacher → RC → structural student.

Teacher ở `runs/teacher/FB15K-237/full/20261017_000249/` có MRR 0.4334, H@10 62.40%. Teacher metric dùng cách đánh giá khác student, được giữ riêng khi so paper.

## Những thử nghiệm trước K

Các kiểm tra A-I cho thấy Adam policy LR `1e-5` trì hoãn saturation đầu nhưng policy cuối vẫn chọn structural-only ở 100% sampled validation query. Trong mẫu đó, structural chỉ là CE oracle ở 2.64% query; description thường có CE tốt hơn. Binary reward ít phân biệt strategy và early updates chủ yếu có advantage âm.

J so bốn cách chọn teacher trên 20.000 example, ba epoch, batch 16 và 1.024 evaluation query:

| Chiến lược | Valid MRR | Test MRR |
|---|---:|---:|
| J0 Structural | 0.2503 | 0.2507 |
| J1 Description | 0.2644 | 0.2616 |
| J2 Average | 0.2492 | 0.2493 |
| J3 Learned RC | 0.2579 | 0.2635 |

Description có validation MRR tốt nhất; learned RC tốt hơn fixed structural. Policy cuối tập trung vào structural chưa mô tả hết teacher selection trong training.

## K đang chạy lúc ghi chép

K1 description và K2 learned RC dùng đủ 544.230 training example, batch 16, tối đa năm epoch và 1.024 query để validation. Hai run được xếp tuần tự trong tmux, log ở `runs/student/FB15K-237/experiment_k/`.

Khi quan sát trên RTX 5090, student dùng khoảng 3.3 GB VRAM, GPU utilization 88% và công suất 427 W. Batch 16 tương ứng khoảng 34.015 step/epoch, ước lượng 8-13 phút/epoch tùy strategy. Các quan sát này thuộc cấu hình pilot; code gốc dùng batch 1000 và tối đa 200 epoch.

Các bước dự định lúc đó gồm hoàn tất K, kiểm tra policy, đo throughput/peak memory ở batch lớn hơn và đối chiếu cấu hình paper. Các kết quả tiếp nối nằm trong [archive thí nghiệm](../experiments/README.md); lệnh mới dùng [hướng dẫn chạy](../../reproduction/running.md).

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="1-goal"></a>
<a id="2-teacher-training"></a>
<a id="3-experiments-a-i-rc-diagnostics"></a>
<a id="4-experiment-j-fixed-teacher-ablation"></a>
<a id="5-experiment-k-full-data-pilot"></a>
<a id="6-training-cost-observation"></a>
<a id="7-next-steps"></a>
<a id="8-current-conclusion"></a>
<a id="dsom-reproduction-progress"></a>
