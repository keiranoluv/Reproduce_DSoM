# RC policy sau optimizer step đầu

Ba cấu hình R1/R2/R4 trên FB15K-237 cho thấy policy có thể tập trung gần hoàn toàn sau một optimizer step. Trang này ghi probe trước/sau cùng batch và phân biệt hiện tượng đó với việc policy tiếp tục tập trung ở cuối training.

## Cấu hình và cách đo

Teacher là pretrained multimodal ComplExMDR; student chỉ dùng cấu trúc, rank 2000, batch 1024, Adagrad LR 0.1, N3 0.01. Distillation dùng temperature 4, KD weight 2, alpha/beta 1/1. Policy MLP có hidden 1024, bảy strategy, binary reward +1/-10 và temperature 1.

| Run | Policy optimizer | Policy LR | Input normalization |
|---|---|---:|---|
| R1 | Shared Adagrad | 0.1 | No |
| R2 | Separate Adam | 0.00001 | No |
| R4 | Shared Adagrad | 0.1 | Yes |

R1/R2 có cùng diagnostics và gradient ở batch đầu. R2 đổi cả thuật toán optimizer và learning rate, nên phép so chưa tách riêng hai yếu tố này.

Probe tính lại teacher prediction, tắt dropout, rồi đo entropy, hidden RMS và logit range ngay trước/sau `optimizer.step()` trên cùng minibatch. Nó không thêm bước tối ưu policy. Entropy tối đa của bảy strategy là `ln(7) ≈ 1.94591`; gần zero nghĩa là xác suất tập trung gần một action.

## Kết quả bước đầu

Entropy:

| Run | Trước update | Sau update |
|---|---:|---:|
| R1 | 1.832511 | 0.004147 |
| R2 | 1.832511 | 1.737293 |
| R4 | 1.919530 | 0.002966 |

Hidden RMS:

| Run | Trước update | Sau update |
|---|---:|---:|
| R1 | 0.857350 | 2032.694092 |
| R2 | 0.857350 | 0.895919 |
| R4 | 0.405769 | 997.501831 |

Khoảng logits:

| Run | Trước update | Sau update |
|---|---:|---:|
| R1 | 1.446046 | 246115.843750 |
| R2 | 1.446046 | 3.250485 |
| R4 | 0.689721 | 120498.843750 |

Gradient và weight update:

| Run | Gradient norm | Maximum FC2 weight change | Nonzero advantages |
|---|---:|---:|---:|
| R1 | 1.977970 | 0.10000002 | 11/1024 |
| R2 | 1.977970 | 0.00001000 | 11/1024 |
| R4 | 1.251625 | 0.10000002 | 14/1024 |

R1/R4 gần saturation ngay sau update, còn R2 giữ entropy cao hơn. Maximum FC2 update khoảng 0.1 với shared Adagrad và `1e-5` với Adam riêng. Probe chưa đo toàn bộ độ lớn update FC1.

## Normalize input có hoạt động không?

R4 bật `policy_input_norm=True`. Input ban đầu có mean -0.0000, std 1.0000 và absmax 13.5943. Sau step, hidden RMS lên 997.5018, logit range 120498.8438 và entropy còn 0.002966. Chuẩn hóa đã hoạt động, nhưng chưa ngăn được saturation với shared Adagrad LR 0.1.

Ở step đầu, Adagrad gần với `delta_w = -lr * gradient / (abs(gradient) + epsilon)`. Với gradient lớn hơn epsilon, độ lớn update có thể gần learning rate. Mạng policy nhận scores của ba modality trên toàn bộ entity; mapping FB15K-237 có 14.541 entity, tức 43.623 input. Ghi chép ban đầu dùng 14.951/44.853 khi ước lượng kích thước; số đó không khớp mapping được dùng trong repo.

Update lớn có thể khuếch đại hidden activation/logits trong mạng này, phù hợp probe quan sát. Chưa tách được đóng góp từng layer hoặc optimizer algorithm khỏi learning rate.

## Policy ở cuối training

Trong cửa sổ log cuối đã kiểm tra, R1/R4 chọn structural khoảng 100%, R2 khoảng 99.41%. Đây là số liệu của cửa sổ log, chưa đại diện mọi query toàn dataset. Adam riêng tránh saturation nặng ở bước đầu nhưng policy vẫn tập trung về sau.

## Kết quả student

Full filtered test:

| Chỉ số | R1 | R2 | R4 |
|---|---:|---:|---:|
| MR | 162.53 | 159.58 | 160.15 |
| MRR | 0.35958 | 0.35708 | 0.35718 |
| Hits@1 (%) | 26.57 | 26.24 | 26.26 |
| Hits@3 (%) | 39.45 | 39.35 | 39.31 |
| Hits@10 (%) | 54.85 | 54.59 | 54.70 |

Paper Table 3 báo MR 116, H@1 27.5%, H@3 41.1%, H@10 56.4%. R2 tránh được saturation tức thời nhưng MRR/H@1/H@10 chưa cao hơn R1 trong các lần chạy đã lưu. Chưa đủ bằng chứng rằng chỉ ổn định step đầu sẽ thu hẹp khoảng cách với paper.

## Log để đối chiếu

- R1: `runs/student/FB15K-237/table3_batch1024/full/train.log`.
- R2: `runs/student/FB15K-237/table3_separate_adam/full/train.log`.
- R4: `runs/student/FB15K-237/table3_r4_input_norm/full/train.log`.

Tìm các marker `===== FIXED-BATCH POLICY PROBE =====`, `RC batch=0` và `MLP batch=0`. Phần ghi diagnostics nằm trong `src/kd.py`, `src/kd_optimizers.py`, `src/kd_learn.py`.

Các kết quả dựa trên log đã có. Baseline source-frozen được ghi riêng trong [kết quả Table 3](../../research/tables/table-3-fb15k237-wn18.md); những cấu hình learned RC ở đây là các lần thử để hiểu optimization.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="1-objective"></a>
<a id="10-findings-and-confidence"></a>
<a id="11-implications-for-reproduction"></a>
<a id="12-conclusion"></a>
<a id="2-experimental-setup"></a>
<a id="3-diagnostic-method"></a>
<a id="4-first-step-results"></a>
<a id="41-policy-entropy"></a>
<a id="42-hidden-activation-rms"></a>
<a id="43-policy-output-logit-range"></a>
<a id="44-policy-gradient-and-parameter-update"></a>
<a id="5-input-normalization-verification"></a>
<a id="6-mechanistic-interpretation"></a>
<a id="7-immediate-saturation-vs-long-term-collapse"></a>
<a id="71-immediate-saturation"></a>
<a id="72-long-term-collapse"></a>
<a id="8-student-evaluation-results"></a>
<a id="9-evidence-locations"></a>
<a id="dsom-table-3--first-step-rc-policy-saturation-analysis"></a>
