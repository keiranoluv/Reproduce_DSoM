# Table 1: MKG-Y và MKG-W

Hai cấu hình `T1_MKGY_Exp01` và `T1_MKGW_Exp01` đã train xong teacher và student. Trang này lấy chỉ số test từ `final_result` của checkpoint student được chọn theo validation MRR. Cả hai dùng RC `source_frozen`.

## So sánh với paper

Các giá trị dưới đây là phần trăm; Δ = tái lập − paper, tính bằng điểm phần trăm (pp).

| Dataset | Chỉ số | DSoM paper (%) | Exp01 reproduction (%) | Delta (pp) |
|:--|:--|--:|--:|--:|
| **MKG-Y** | **MRR** | 40.02 | **40.69** | **+0.67** |
| MKG-Y | Hits@1 | 35.67 | **36.01** | +0.34 |
| MKG-Y | Hits@3 | 41.70 | **42.41** | +0.71 |
| MKG-Y | Hits@10 | 47.71 | **48.99** | +1.28 |
| **MKG-W** | **MRR** | 40.90 | **40.58** | **-0.32** |
| MKG-W | Hits@1 | 33.56 | **33.54** | -0.02 |
| MKG-W | Hits@3 | 43.58 | **43.19** | -0.39 |
| MKG-W | Hits@10 | 55.28 | **53.73** | -1.55 |

MRR trên MKG-Y cao hơn paper 0.67 pp; trên MKG-W thấp hơn 0.32 pp. Đây là kết quả một seed và có các khác biệt về policy, đặc trưng được nêu bên dưới.

## Cấu hình và thời gian chạy

| Trường | MKG-Y Exp01 | MKG-W Exp01 |
|:--|:--|:--|
| Entities | 15,000 | 15,000 |
| Base relations | 28 | 169 |
| Train / Valid / Test triples | 21,310 / 2,665 / 2,663 | 34,196 / 4,276 / 4,274 |
| Teacher best epoch | 36 | 72 |
| Teacher total epochs | 47 | 83 |
| Student best epoch | 42 | 54 |
| Student total epochs | 53 | 65 |
| Student best validation MRR | 0.38296 | 0.39081382006406784 |
| Teacher runtime (W&B) | ~122.1 s | ~345.6 s |
| Student runtime (W&B) | Không ghi trong bản tổng hợp này | ~483.3 s |
| Training pipeline | **PASS** | **PASS** |
| Student checkpoint selection | Best validation MRR | Best validation MRR |

Teacher là `ComplExMDR` dùng cấu trúc, ảnh và mô tả; student chỉ dùng cấu trúc. Cấu hình chung: rank 2000, batch 1000, Adagrad LR 0.1, N3 0.01, scheduler `min`, tối đa 200 epoch, validation mỗi 3 epoch, early stopping 10.

Student dùng `kd_loss_type=kdrl`, `teacher_strategy=rc`, `policy_optimizer=source_frozen`, temperature 4 và KD weight 2. Seed trong tên thí nghiệm là 2022. Policy vẫn lấy mẫu action nhưng giữ nguyên trọng số. Log MKG-W ghi 69/69 batch có policy gradient khác không và không batch nào cập nhật policy weights, đúng với cấu hình này.

## Số liệu đầy đủ và vị trí lưu

| Dataset | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|:--|--:|--:|--:|--:|--:|
| MKG-Y | 468.45867919921875 | 0.4069194197654724 | 0.36012017726898193 | 0.4241456985473633 | 0.48986104130744934 |
| MKG-W | 232.71842193603516 | 0.4058402553200722 | 0.3354000747203827 | 0.43191391229629517 | 0.5373186469078064 |

Các số trên được đọc từ `result["final_result"]` sau khi full student hoàn tất. Kết quả nằm ở `runs/student/<DATASET>/full/<EXPERIMENT_NAME>/<RUN_ID>/checkpoints/0/result.json`. Trường `teacher_checkpoint` trỏ tới teacher đã dùng.

| Dataset | Run ID | Commit trong result.json |
|---|---|---|
| MKG-Y | `20261020_174808_2562445` | `39c7886f8bfd728e9c79afb11a0800ae0ac4fad7` |
| MKG-W | `20261020_181221_2590139` | `e148dbeb8330a36cf3f41d3e60fd103a942304b4` |

[W&B student MKG-Y](https://wandb.ai/keiranoluv-hcmus/DSoM-Reproduction/runs/2a9t5tjk). Bản tổng hợp chưa có URL W&B MKG-W.

W&B `test/*` là lần đánh giá được log cuối. Chẳng hạn MKG-W có `test/MRR=0.4047639220952988` tại epoch 63, còn `final_result.MRR=0.4058402553200722` thuộc epoch 54. Khi điền bảng, dùng `final_result`.

## Dữ liệu và đặc trưng

MKG-Y có 15.000 entity, đặc trưng ảnh `(15000, 1000)` và văn bản `(15000, 768)`. Dữ liệu graph và hai mảng đặc trưng đã được kiểm tra.

MKG-W ánh xạ đủ 15.000 URI Wikidata sang DBpedia, không thiếu link hoặc có xung đột. Có 14.150 mô tả khớp và 850 mô tả thiếu. Archive có ảnh ứng viên cho 8.959 entity, còn 6.041 entity không có; số ảnh encode được có thể thấp hơn số ứng viên. Manifest lưu độ phủ thực tế.

Đặc trưng BERT/ViT được dựng lại từ dữ liệu gốc; cách chọn ảnh và điền modality thiếu chưa được xác nhận trùng với tác giả. Các giá trị paper trong bản tổng hợp được chép khi tái lập, cần đối chiếu Table 1 bản xuất bản trước khi dùng để công bố. Xem [cấu hình thực nghiệm](../methodology/protocol.md) và [cách chạy lại](../../reproduction/running.md).

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="1-paper-comparison-student-test-set"></a>
<a id="2-run-summary"></a>
<a id="3-exact-best-checkpoint-student-test-metrics"></a>
<a id="4-training-setup-and-provenance"></a>
<a id="5-dataset-and-modality-audit-highlights"></a>
<a id="6-reproduction-limitations-and-next-steps"></a>
<a id="dsom-table-1--mkg-y-and-mkg-w-exp01-reproduction-results"></a>
<a id="result-artifact-locations"></a>
<a id="why-wb-test-differs-from-the-final-values"></a>
