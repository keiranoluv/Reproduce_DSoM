# DSoM Table 1 — Reproduction MKG-Y Exp01

## 1. Thông tin thí nghiệm

- Experiment ID: T1_MKGY_Exp01
- Dataset: MKG-Y
- GPU: NVIDIA RTX 5090 32GB
- Seed: 2022
- Git commit: 39c7886f8bfd728e9c79afb11a0800ae0ac4fad7
- Teacher: ComplExMDR (Structure + Image + Description)
- Student: Structure-only ComplExMDR
- Distillation: RC, source_frozen

## 2. Dataset

Nguồn: MMRNS
https://github.com/quqxui/MMRNS

- Entities: 15,000
- Base relations: 28
- Train triples: 21,310
- Validation triples: 2,665
- Test triples: 2,663

Structural preprocessing và multimodal feature validation hoàn tất.

Feature matrices:
- Image: (15000, 1000), ViT B/16
- Text: (15000, 768), BERT-base-uncased

## 3. Cấu hình training

| Tham số | Giá trị |
|---|---|
| Rank | 2000 |
| Batch size | 1000 |
| Optimizer | Adagrad |
| Learning rate | 0.1 |
| Scheduler | min |
| Regularizer | N3, weight 0.01 |
| Max epochs | 200 |
| Validation interval | 3 |
| Early stopping | 10 |
| Student KD loss | kdrl |
| Teacher strategy | rc |
| Policy optimizer | source_frozen |
| KD temperature | 4 |
| CE weight | 1 |
| KL weight | 2 |
| RC hidden dimension | 1024 |
| RC reward / penalty | 1 / 10 |
| RC reward mode | binary |

## 4. Tóm tắt training

Teacher:
- Epoch tốt nhất: 36
- Tổng epochs: 47
- Chọn checkpoint: validation Hits@10 tốt nhất

Student:
- Epoch tốt nhất: 42
- Tổng epochs: 53
- Chọn checkpoint: validation MRR tốt nhất
- Best validation MRR: 0.38296

Kiểm tra student:
- Dùng full training data
- Full validation/test evaluation
- Load teacher thành công
- Source-frozen optimizer parameter check PASS
- Lưu student checkpoint
- Đồng bộ W&B hoàn tất
- Full pipeline: PASS

## 5. Kết quả test student

Kết quả chính thức đọc từ final_result của student result.json tại checkpoint
có validation tốt nhất.

| Chỉ số | DSoM Paper | Exp01 | Chênh lệch |
|---|---:|---:|---:|
| MR | Không ghi tại đây | 468.4587 | N/A |
| MRR (%) | 40.02 | 40.69 | +0.67 pp |
| Hits@1 (%) | 35.67 | 36.01 | +0.34 pp |
| Hits@3 (%) | 41.70 | 42.41 | +0.71 pp |
| Hits@10 (%) | 47.71 | 48.99 | +1.28 pp |

Metrics reproduction chưa làm tròn:
- MR: 468.45867919921875
- MRR: 0.4069194197654724
- Hits@1: 0.36012017726898193
- Hits@3: 0.4241456985473633
- Hits@10: 0.48986104130744934

## 6. Theo dõi thí nghiệm

W&B Student run:
https://wandb.ai/keiranoluv-hcmus/DSoM-Reproduction/runs/2a9t5tjk

Run ID:
20261020_174808_2562445

Kết quả teacher:
runs/teacher/MKG-Y/full/

Kết quả student:
runs/student/MKG-Y/full/

Nguồn số liệu cuối là student result.json, không W&B test metrics log cuối.

## 7. Giới hạn reproduction

- Text/image feature generation là lựa chọn reproduction, chưa xác minh hoàn toàn preprocessing tác giả.
- source_frozen sample strategies mà không update RC policy parameters.
- Không xác lập reproduction learned-policy REINFORCE mô tả trong paper.
- Teacher eval có oracle modality ranking, không so trực tiếp Table 1.
- Một run, không đo variance nhiều seeds.

## 8. Kết luận

MKG-Y Exp01 hoàn tất thành công.

Student test MRR/Hits@K gần và nhỉnh hơn published Table 1.

Dataset tiếp theo: MKG-W.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="1-experiment-information"></a>
<a id="3-training-configuration"></a>
<a id="4-training-summary"></a>
<a id="5-student-test-results"></a>
<a id="6-experiment-tracking"></a>
<a id="7-reproduction-limitations"></a>
<a id="8-conclusion"></a>
<a id="dsom-table-1--mkg-y-exp01-reproduction"></a>
