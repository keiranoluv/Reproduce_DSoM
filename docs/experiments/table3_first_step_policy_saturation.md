# DSoM Table 3 — Phân tích RC saturation ngay step đầu

## 1. Mục tiêu

Khảo sát RC policy collapse khi reproduce student Table 3 trên FB15K-237.

Xác định update optimizer đầu có gây saturation ngay không, và normalize input
hay policy optimizer riêng có ngăn được không.

## 2. Thiết lập thí nghiệm

Dataset: FB15K-237

Teacher:
Pretrained multimodal ComplExMDR checkpoint.

Student:
Unimodal ComplExMDR, rank 2000.

RC policy:
- MLP 1024 hidden units
- Bảy multimodal teacher strategies
- Binary reward
- Reward = +1
- Penalty = -10
- Policy temperature = 1

Cấu hình student chung:
- Adagrad, learning rate 0.1
- Batch size 1024
- N3 regularization, weight 0.01
- KD temperature 4
- KD weight 2
- NDKD alpha = 1, beta = 1

Các thí nghiệm so sánh:

| Run | Policy optimizer | Policy LR | Input normalization |
|---|---|---:|---|
| R1 | Shared Adagrad | 0.1 | No |
| R2 | Separate Adam | 0.00001 | No |
| R4 | Shared Adagrad | 0.1 | Yes |

R1/R2 có cấu hình policy khởi tạo giống nhau, diagnostics trước update ở batch đầu giống nhau.

R2 đổi cả optimizer algorithm và learning rate; so sánh này chưa cô lập từng đóng góp.

## 3. Phương pháp chẩn đoán

Ghi policy statistics trên cùng minibatch ngay trước/sau optimizer update.

Mỗi probe:
1. Tính lại teacher predictions.
2. Tính policy probabilities khi tắt dropout.
3. Ghi entropy, hidden RMS, output logit range.
4. So trước/sau optimizer.step().

Probe dùng cùng batch, không thêm policy optimization step.

Đo entropy trên bảy strategies.

Entropy tối đa:

H_max = ln(7) = xấp xỉ 1.94591

Entropy gần zero cho thấy policy gần deterministic.

Nguồn files liên quan:
- src/kd.py
- src/kd_optimizers.py
- src/kd_learn.py

## 4. Kết quả step đầu

### 4.1 Entropy policy

| Run | Trước update | Sau update |
|---|---:|---:|
| R1 | 1.832511 | 0.004147 |
| R2 | 1.832511 | 1.737293 |
| R4 | 1.919530 | 0.002966 |

R1/R4 gần saturation hoàn toàn sau update đầu.

R2 giữ entropy cao hơn đáng kể.

### 4.2 RMS hidden activations

| Run | Trước update | Sau update |
|---|---:|---:|
| R1 | 0.857350 | 2032.694092 |
| R2 | 0.857350 | 0.895919 |
| R4 | 0.405769 | 997.501831 |

Hidden RMS tăng mạnh ở R1/R4.

R2 giữ hidden activation gần giá trị trước update.

### 4.3 Khoảng logits đầu ra policy

| Run | Trước update | Sau update |
|---|---:|---:|
| R1 | 1.446046 | 246115.843750 |
| R2 | 1.446046 | 3.250485 |
| R4 | 0.689721 | 120498.843750 |

Logit ranges rất lớn sau update R1/R4 khiến softmax policy tập trung mạnh.

### 4.4 Policy gradient và parameter update

| Run | Gradient norm | Maximum FC2 weight change | Nonzero advantages |
|---|---:|---:|---:|
| R1 | 1.977970 | 0.10000002 | 11/1024 |
| R2 | 1.977970 | 0.00001000 | 11/1024 |
| R4 | 1.251625 | 0.10000002 | 14/1024 |

R1/R2 có cùng gradient norms và nonzero-advantage counts ở batch đầu.

Tuy nhiên optimizer updates khác nhau nhiều.

Đo trực tiếp maximum FC2 weight change; logs chưa đo toàn bộ độ lớn FC1 update.

## 5. Kiểm tra input normalization

R4 dùng:

    policy_input_norm=True

MLP input statistics được ghi đầu tiên:

    teacher_mean   = -0.0000
    teacher_std    = 1.0000
    teacher_absmax = 13.5943

Normalize input hoạt động, tạo mean/std đúng kỳ vọng.

Nhưng sau optimizer step đầu:

    hidden_rms  = 997.5018
    logit_range = 120498.8438
    entropy     = 0.002966

Kết luận:

Chỉ normalize input không ngăn first-step saturation khi dùng chung Adagrad LR 0.1.

## 6. Diễn giải cơ chế

Ở step đầu, Adagrad gần áp dụng:

    delta_w = -lr * gradient / (abs(gradient) + epsilon)

Với gradients lớn hơn epsilon, update magnitude đầu có thể gần learning rate.

Policy LR = 0.1 cho observed maximum FC2 change xấp xỉ 0.1.

MLP policy nhận input nhiều chiều, ghép scores ba modalities trên mọi candidate entities.

Với FB15K-237, tương ứng khoảng:

14951 entities * 3 modalities = 44853 input features

FC layer đầu ánh xạ sang 1024 hidden units.

Update lớn trên mạng nhiều chiều có thể khuếch đại mạnh hidden activations/output logits.

Cơ chế phù hợp quan sát saturation bước đầu ở R1/R4.

Probe trước/sau cho thấy optimizer update là nguyên nhân trực tiếp thay đổi policy outputs được quan sát.

Chưa cô lập đóng góp từng layer hoặc tách optimizer algorithm khỏi learning-rate effect.

## 7. Saturation tức thời và collapse dài hạn

Thí nghiệm hỗ trợ phân biệt hai hiện tượng.

### 7.1 Saturation tức thời

R1/R4 entropy gần zero ngay sau update đầu.

R2 tránh tác động tức thời đó.

### 7.2 Collapse dài hạn

Audits R1-R7 trước đó ghi nhận policy cuối cùng tập trung Structural-only.

Cụ thể:
- R1/R4: khoảng 100% Structural
- R2: khoảng 99.41% Structural

Con số mô tả final training-log window đã audit, không phải mọi prediction toàn dataset.

Tránh immediate saturation chưa đảm bảo policy đa dạng suốt training.

Chưa giải quyết cơ chế long-term policy collapse.

## 8. Kết quả student evaluation

Full filtered test evaluation:

| Chỉ số | R1 | R2 | R4 |
|---|---:|---:|---:|
| MR | 162.53 | 159.58 | 160.15 |
| MRR | 0.35958 | 0.35708 | 0.35718 |
| Hits@1 (%) | 26.57 | 26.24 | 26.26 |
| Hits@3 (%) | 39.45 | 39.35 | 39.31 |
| Hits@10 (%) | 54.85 | 54.59 | 54.70 |

Student reference DSoM/Table 3:

    MR       = 116
    Hits@1   = 27.5%
    Hits@3   = 41.1%
    Hits@10  = 56.4%

R2 tránh saturation nặng như R1 nhưng không hơn R1 về MRR/Hits@1/Hits@10 trong recorded runs.

Chưa chứng minh loại immediate saturation sẽ thu hẹp Table 3 gap.

## 9. Vị trí bằng chứng

R1 log:

    runs/student/FB15K-237/table3_batch1024/full/train.log

R2 log:

    runs/student/FB15K-237/table3_separate_adam/full/train.log

R4 log:

    runs/student/FB15K-237/table3_r4_input_norm/full/train.log

Diagnostic markers chính:

    ===== FIXED-BATCH POLICY PROBE =====
    RC batch=0
    MLP batch=0

Mọi kết quả dựa trên logs đã có, không training mới.

## 10. Phát hiện và độ chắc chắn

Đã xác nhận:

1. R1/R4 policy gần deterministic ngay sau update đầu.

2. Normalize input R4 hoạt động đúng.

3. R1/R2 bắt đầu với diagnostics và first-batch gradient norms giống nhau.

4. Adam policy optimizer riêng LR 1e-5 ngăn saturation nặng bước đầu ở R2.

5. Maximum FC2 update khoảng 0.1 cho R1/R4 và 1e-5 cho R2.

Được hỗ trợ mạnh bởi bằng chứng:

6. Shared Adagrad LR 0.1 làm policy update đầu quá lớn đối với kiến trúc được thử.

Chưa xác lập:

7. Adagrad hay độ lớn LR chịu trách nhiệm chính.

8. Cùng cơ chế có giải thích mọi collapse về sau không.

9. Ổn định RC có đủ tái lập published student Table 3 metrics không.

## 11. Hàm ý đối với reproduction

First-step saturation R1/R4 là optimization instability cụ thể, không phải normalize input thất bại.

Đổi optimizer config vẫn là experimental variation nếu chưa độc lập xác minh paper protocol.

Đề xuất:

1. Giữ R1 làm source-oriented baseline hiện tại.
2. Giữ R2/R4 làm diagnostic ablations.
3. Tránh tiếp tục tìm optimizer hyperparameters thiếu định hướng.
4. Ghi immediate/long-term collapse riêng.
5. Tiếp tục căn chỉnh paper/source, nhất là features và teacher config.

## 12. Kết luận

R1/R4 saturation ngay sau một shared-Adagrad step.

R2 cho thấy Adam riêng với LR nhỏ hơn nhiều tránh immediate saturation nhưng không ngăn concentration cuối.

Phát hiện cung cấp giải thích có thể tái lập cho một RC training instability lớn đã quan sát.

Tự chúng chưa xác lập nguyên nhân phần Table 3 reproduction gap còn lại.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
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
