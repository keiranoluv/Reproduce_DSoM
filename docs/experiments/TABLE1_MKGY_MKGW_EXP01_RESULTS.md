# DSoM Table 1 — Kết quả reproduction MKG-Y/MKG-W Exp01

**Trạng thái:** Cả hai full Teacher -> Student pipelines hoàn tất thành công.
**Mã thí nghiệm:** `T1_MKGY_Exp01`, `T1_MKGW_Exp01`
**Biến thể:** Source-Frozen RC baseline, không phải learned-policy REINFORCE.
**Quy tắc đánh giá:** Báo **test metrics từ `final_result` trong `result.json` của checkpoint có validation tốt nhất**, không dùng W&B `test/*` cuối.

## 1. So sánh paper trên student test set

Các giá trị là phần trăm. Delta = reproduction trừ paper, tính bằng **điểm phần trăm (pp)**.

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

**Diễn giải:** Hai student runs gần các giá trị Table 1 trong paper. Đây là kết quả một lần chạy; gần nhau về con số **chưa chứng minh tái lập chính xác phương pháp**.

## 2. Tóm tắt runs

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

## 3. Student test metrics chưa làm tròn tại best checkpoint

Dưới đây là outputs chưa làm tròn của mỗi full pipeline đã hoàn thành:

| Dataset | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|:--|--:|--:|--:|--:|--:|
| MKG-Y | 468.45867919921875 | 0.4069194197654724 | 0.36012017726898193 | 0.4241456985473633 | 0.48986104130744934 |
| MKG-W | 232.71842193603516 | 0.4058402553200722 | 0.3354000747203827 | 0.43191391229629517 | 0.5373186469078064 |

Các giá trị lấy từ output `STUDENT FULL: PASS`, đọc `result["final_result"]` trong student `result.json` ở thư mục checkpoint.

## 4. Thiết lập training và nguồn gốc

Cấu hình chung của Exp01 pipelines:

- Teacher: `ComplExMDR`, dùng structural, image và description modalities.
- Student: structural model được distill từ teacher checkpoint tương ứng.
- `rank=2000`, `batch_size=1000`, optimizer `Adagrad`, learning rate ban đầu `0.1`.
- Regularizer `N3` với `reg=0.01`; scheduler mode `min`.
- `max_epochs=200`, validation mỗi 3 epochs, early stopping patience 10.
- Student: `kd_loss_type=kdrl`, `teacher_strategy=rc`, `policy_optimizer=source_frozen`, KD temperature 4 và KL weight 2.
- Seed ghi trong tên thí nghiệm: `2022`.

**MKG-Y Git SHA đã xác nhận trong student `result.json`:** `39c7886f8bfd728e9c79afb11a0800ae0ac4fad7`.

**MKG-W Git SHA đã xác nhận trong student `result.json`:** `e148dbeb8330a36cf3f41d3e60fd103a942304b4`.

**MKG-Y W&B Student:** https://wandb.ai/keiranoluv-hcmus/DSoM-Reproduction/runs/2a9t5tjk

### Vị trí artifacts kết quả

- MKG-Y student run ID: `20261020_174808_2562445`
- MKG-W student run ID: `20261020_181221_2590139`
- Student `result.json` nằm dưới: `runs/student/<DATASET>/full/<EXPERIMENT_NAME>/<RUN_ID>/checkpoints/0/result.json`
- Đường dẫn teacher checkpoint tương ứng được lưu ở field `teacher_checkpoint` trong student `result.json`.

### Vì sao W&B `test/*` khác kết quả cuối

`test/*` trong W&B summary là **evaluation được log cuối**, không nhất thiết là metrics của **student checkpoint tốt nhất**. Với MKG-W, W&B báo `test/MRR=0.4047639220952988` tại epoch 63; `final_result.MRR=0.4058402553200722` thuộc best epoch 54. Run train tổng cộng 65 epochs. Vì vậy **dùng `final_result` trong JSON local cho Table 1**.

## 5. Các điểm chính của audit dataset/modalities

- **MKG-Y:** 15,000 entities; image vectors `(15000, 1000)`, description vectors `(15000, 768)`; đã kiểm tra structural data và features.
- **MKG-W:** Audit structural triples/mapping thành công; 15,000/15,000 mappings Wikidata-to-DBpedia, không thiếu links hoặc conflicts.
- **MKG-W text:** Khớp 14,150 descriptions; thiếu 850; audit alignment **PASS**.
- **MKG-W raw images:** 8,959 entities có *ảnh ứng viên* trong archive; 6,041 không có; audit alignment **PASS**. Số ứng viên khác số ảnh thực sự decode thành công.
- Features từ raw images có thể khác độ phủ image embeddings MMRNS công bố. Giữ manifests để ghi số ảnh thực tế được encode.

Xem hồ sơ chuẩn bị MKG-Y/MKG-W để truy nguyên preprocessing và SHA256 đầu vào; lệnh chạy hiện tại nằm trong DATA.md và RUNNING.md.

## 6. Giới hạn reproduction và bước tiếp theo

1. **RC policy đóng băng:** `source_frozen` sample actions nhưng không cập nhật weights; đây là baseline, không phải learned-policy RC trong paper. MKG-W logs có 69/69 batches với policy gradients khác không nhưng 0 batches cập nhật weights, đúng kỳ vọng.
2. **Khác biệt features:** Trích xuất BERT/ViT, chọn ảnh và khởi tạo missing modality là lựa chọn reproduction; chưa xác minh tương đương đầy đủ với tác giả.
3. **Một seed/một run:** Không coi delta dương/âm là cải thiện/suy giảm thống kê nếu chưa chạy nhiều seeds.
4. **Giá trị paper:** Được chép trong quá trình thảo luận reproduction; cần đối chiếu Table 1 bản xuất bản trước khi công bố chính thức.
5. **Hoàn thiện provenance:** Bổ sung MKG-W Git SHA và W&B run URL khi lấy được artifacts local, không đổi SHA đã dùng khi training.

**Kết luận:** `T1_MKGY_Exp01` và `T1_MKGW_Exp01` đều đạt `FULL PIPELINE: PASS`; test metrics của selected student gần Table 1 theo Source-Frozen RC protocol đã ghi nhận.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="1-paper-comparison-student-test-set"></a>
<a id="2-run-summary"></a>
<a id="3-exact-best-checkpoint-student-test-metrics"></a>
<a id="4-training-setup-and-provenance"></a>
<a id="5-dataset-and-modality-audit-highlights"></a>
<a id="6-reproduction-limitations-and-next-steps"></a>
<a id="dsom-table-1--mkg-y-and-mkg-w-exp01-reproduction-results"></a>
<a id="result-artifact-locations"></a>
<a id="why-wb-test-differs-from-the-final-values"></a>
