# DSoM Reproduction — Research Progress Note

**Cập nhật:** 22/10/2026  
**Paper:** DSoM — *Dark Side of Modalities: Reinforced Multimodal Distillation for Multimodal Knowledge Graph Reasoning* (ACM MM 2025)  
**Repository:** https://github.com/keiranoluv/Reproduce_DSoM  
**Workspace:** `/workspace/workspace/DSoM`  
**Python:** `./dsom/bin/python`  
**GPU:** NVIDIA RTX 5090 32GB  
**W&B project:** `DSoM-Reproduction`

## 1. Mục tiêu nghiên cứu

Ban đầu: Reproduce các kết quả thực nghiệm của DSoM.

Hiện tại: Ngoài reproduction, tập trung kiểm chứng hai đóng góp chính của paper:

1. **RC (Reinforced Configuration):** Reinforcement Learning có thực sự giúp lựa chọn Teacher strategies tốt hơn không?
2. **NDKD (Neighbor-Discriminative Knowledge Distillation):** NDKD có thực sự tốt hơn Vanilla KD và DKD không?

Các thí nghiệm DB15K sử dụng Teacher đã train, không cần train lại Teacher cho từng ablation.

## 2. Reproduction đã thực hiện trước đó

- **Table 3 — FB15K-237:** Đã thực hiện Student reproduction, best Test MRR khoảng 0.35958.
- **WN18:** Đã thực hiện reproduction.
- **Table 1 — MKG-Y:** Test MRR 0.406919, so với paper 0.4002.
- **Table 1 — MKG-W:** Test MRR 0.405840, so với paper 0.4090.
- **Table 1 — DB15K:** Exp01 Source-frozen RC, Test MRR 0.413046, so với paper 0.4268.

Các thí nghiệm đã được triển khai trong các giai đoạn trước, với checkpoints và logs lưu trong `runs/`.

## 3. DB15K — Kết quả hoàn thành

Tất cả MRR bên dưới đều lấy từ `final_result` của **best-validation checkpoint**, không phải metric của epoch cuối trong W&B.

| Experiment | Teacher selection | Distillation | Best epoch | Test MRR |
|---|---|---|---:|---:|
| Exp01 | Source-frozen RC | NDKD | 63 | 0.413046 |
| A1 | Structural-only | NDKD | 48 | 0.412724 |
| A2 | Description-only | NDKD | 18 | 0.381768 |
| A3 | Structural+Description | NDKD | 39 | 0.413734 |
| A4 | Average | NDKD | 39 | 0.413150 |
| A5 | Learned RC, Binary Reward | NDKD | 63 | 0.414221 |
| A6 | Learned RC, Continuous CE | NDKD | 45 | 0.412854 |
| **A7** | **Structural+Description** | **Vanilla KD** | **39** | **0.414357** |
| A8 | Structural+Description | DKD | 39 | 0.413976 |

**Hiện đã có 9 kết quả DB15K:** Exp01 và A1–A8.

A7 đang có MRR cao nhất trong nhóm đã chạy.

### So với DSoM paper

Paper Table 1 trên DB15K:

- MRR: **0.4268**
- Hits@1: **0.3445**
- Hits@3: **0.4662**
- Hits@10: **0.5812**

Kết quả tốt nhất theo MRR của chúng ta là A7: **0.414357**, thấp hơn paper khoảng 1.244 điểm phần trăm MRR.

A7 không phải mô hình DSoM đầy đủ, vì không sử dụng RC.

## 4. Các phát hiện khoa học quan trọng

### 4.1. RC policy không được học trong released-code optimizer

Audit trước đây xác nhận:

- RC có policy gradients.
- Tuy nhiên, optimizer theo released-code configuration không chứa RC parameters.
- Vì vậy, policy weights không cập nhật.
- Student vẫn được cập nhật bình thường.

Chế độ tái hiện này được đặt tên `source_frozen`.

Do đó, cần phân biệt **Source-frozen RC** và **Learned RC** khi trình bày kết quả.

### 4.2. Learned RC có hiện tượng policy collapse

Chúng ta đã kiểm tra vấn đề này trên FB15K-237 và DB15K.

Trên DB15K, A5 dùng:

- Student: Adagrad, LR 0.1.
- RC policy: Adam riêng, LR 1e-5.
- Binary Reward: +1/−10.
- NDKD.

Diagnostics cho thấy policy weights thực sự cập nhật, nhưng phân phối chọn strategy gần như suy biến về Structural-only.

Ví dụ:

`entropy=0.002152`  
`actions=[444,0,0,0,0,0,0]`

Đây là bằng chứng về policy collapse ở các batch được quan sát, nhưng cần thêm đánh giá theo epoch và validation queries trước khi khẳng định hành vi trên toàn bộ quá trình training.

### 4.3. Fixed strategies gần bằng Learned RC

A3 Structural+Description đạt MRR 0.413734, gần bằng A5 Learned RC 0.414221.

A1 Structural-only cũng đạt 0.412724.

Điều này đặt câu hỏi liệu dynamic teacher selection có tạo ra lợi thế thực chất trong pipeline DB15K hiện tại hay không.

Chưa thể kết luận chắc chắn vì mới có single-seed runs.

### 4.4. Continuous CE Reward chưa cải thiện RC

A5 Binary Reward: MRR 0.414221.

A6 Continuous CE: MRR 0.412854.

Chênh lệch A6 − A5: **−0.001368**, tức −0.1368 điểm phần trăm.

A6 kém A5 trong lần chạy hiện tại. Chưa xác nhận Continuous CE có giảm policy collapse hay không.

### 4.5. NDKD chưa vượt KL/DKD với fixed Teacher

Giữ nguyên Structural+Description:

- A3 NDKD: 0.413734.
- A7 Vanilla KD: **0.414357**.
- A8 DKD: 0.413976.

Kết quả hiện tại: KL > DKD > NDKD.

Điều này khác với xu hướng Table 4 trong paper. Tuy nhiên, paper đánh giá các losses trong cấu hình có RC, nên A3/A7/A8 **không phải phép tái lập trực tiếp các hàng RC + loss**.

Chênh lệch cực nhỏ; chưa đủ bằng chứng về ý nghĩa thống kê.

## 5. DB15K — Các vấn đề dữ liệu đã phát hiện

Dataset sử dụng: `MMKB-DB15K`.

- 12.842 entities.
- 279 original relations.
- 79.222 train triples.
- 9.904 validation triples.
- 9.902 test triples.

Audit phát hiện validation leakage nghiêm trọng:

- Validation có 2.423 dòng duplicate.
- Train/validation overlap 6.584 unique triples.
- Validation/test overlap 824 unique triples.
- Chỉ 73 unique triples thuộc riêng validation theo phép audit đã thực hiện.
- Train/test không có triples trùng trực tiếp.

Do đó, best validation MRR khoảng **0.94** không thể được diễn giải như bằng chứng generalization độc lập.

Các kết quả hiện tại tuân theo benchmark protocol đã sử dụng, nhưng cần clean-split sensitivity analysis.

Image features được dựng lại từ MMKB historical image sources kết hợp ViT; text features dùng BERT. Chưa xác minh toàn bộ quá trình preprocessing trùng hoàn toàn paper.

## 6. Đang triển khai — A9 và A10

**Trạng thái xác nhận gần nhất:** Đã chuẩn bị và gửi launcher, nhưng chưa nhận được log chứng minh A9/A10 đã khởi chạy hoặc hoàn thành.

| Experiment | RC | Reward | Loss | Trạng thái |
|---|---|---|---|---|
| A9 | Learned RC, Adam 1e-5 | Binary | Vanilla KD | Chuẩn bị/chờ log |
| A10 | Learned RC, Adam 1e-5 | Binary | DKD | Chuẩn bị/chờ log |

Mục tiêu là so sánh trực tiếp với A5 NDKD trong cùng một cấu hình Learned RC.

Script được gửi để tạo trực tiếp bằng heredoc:

`scripts/run_db15k_a9_a10_direct.sh`

Lệnh:

```bash
bash scripts/run_db15k_a9_a10_direct.sh --prepare

git diff --check
git add src/kd_learn.py src/kd_optimizers.py \
  scripts/run_db15k_a9_a10_direct.sh
git commit -m "exp(db15k): add A9 A10 learned RC loss ablation"

bash scripts/run_db15k_a9_a10_direct.sh --smoke

bash scripts/run_db15k_a9_a10_direct.sh --full
```

Full mode chạy hai jobs song song, với W&B và checkpoints riêng. Smoke mode không gửi W&B.

**Lưu ý:** Chỉ chạy lại bước chuẩn bị nếu patch chưa được áp dụng; trước khi chạy full, xác nhận smoke test PASS và không có job tương ứng đang chạy.

## 7. Đường dẫn kết quả quan trọng

**Exp01:**

`runs/student/MMKB-DB15K/full/T1_DB15K_Exp01_BERT_ViT_Student_RC_SourceFrozen_SchMin_BS1000_R2000_LR0p1_N3_KL2_T4_Seed2022/20261021_151535_2887785/checkpoints/0/result.json`

**A1–A4:**

`runs/student/MMKB-DB15K/fixed_teacher_ndkd_full/20261021_163151_2915928/summary.json`

**A5:**

`runs/student/MMKB-DB15K/learned_rc_adam_full/20261021_173746_2988787/checkpoints/0/result.json`

**A6:**

`runs/student/MMKB-DB15K/learned_rc_continuous_ce_full/20261021_175005_3002605/checkpoints/0/result.json`

**A7/A8:**

`runs/student/MMKB-DB15K/table4_fixed_loss_full/20261021_181938_3048584/`

Các experiments đã dùng nhiều Git commits khác nhau. Cần audit sự khác biệt training logic giữa các commits trước khi kết luận cuối.

## 8. Việc cần làm tiếp theo

- [x] Reproduce Table 1 cho MKG-Y, MKG-W và DB15K.
- [x] Audit optimizer membership và RC policy collapse.
- [x] Chạy DB15K fixed teacher selection A1–A4.
- [x] Chạy A5 Learned RC Binary.
- [x] Chạy A6 Learned RC Continuous CE.
- [x] Chạy A7/A8 fixed teacher KL vs DKD.
- [ ] Xác nhận chạy và lấy kết quả A9/A10 Learned RC KL vs DKD.
- [ ] Hoàn thiện các hàng còn thiếu của Table 4 theo protocol paper.
- [ ] Chạy Neighbor-only và Non-neighbor-only KD ablations.
- [ ] Chạy repeated seeds với các phương pháp quan trọng.
- [ ] Đánh giá ảnh hưởng của DB15K validation leakage.
- [ ] Audit Git commits, feature provenance và checkpoint selection.
- [ ] Tổng hợp kết quả thành research report.

## 9. Kết luận tạm thời

**Chúng ta đã reproduce được hiệu năng tương đối gần paper DSoM trên DB15K, nhưng chưa quan sát thấy lợi ích rõ ràng từ learned RC hoặc NDKD trong các ablations hiện tại.**

Các kết quả gợi ý rằng fixed teacher selection và Vanilla KD có thể là những đối chứng cạnh tranh mạnh.

Tuy nhiên, chưa thể bác bỏ đóng góp của paper khi chưa kiểm soát đầy đủ seed, optimizer behavior, feature pipeline và validation leakage.

**Next milestone:** Hoàn thành A9/A10, tạo bảng Learned RC + NDKD/KL/DKD, rồi đánh giá xem NDKD có thực sự tốt hơn khi RC được cập nhật hay không.
