# L9: student học từ Structural+Description

Sau kết quả filtered teacher ở L8, L9 thử distill student từ trung bình structural và description logits, trọng số `(0.5, 0, 0.5)`. Hai teacher branch được đóng băng.

## Cấu hình

Dataset FB15K-237 có 544.230 reciprocal training query. Student là unimodal ComplExMDR, dùng `fixed_ndkd`, rank 2000, batch 16, năm epoch, Adagrad LR 0.1, N3 0.01, CE/KD weights 1/2, temperature 4 và scheduler ReduceLROnPlateau `min`.

Teacher checkpoint: `runs/teacher/FB15K-237/full/20261017_000249/checkpoints/0/m-20261017_0002-n-teacher_fb15k237_full_20261017_000249.pth`.

## Kết quả filtered đầy đủ

| Split | MR | MRR | Hits@1 | Hits@3 | Hits@10 |
|---|---:|---:|---:|---:|---:|
| Validation | 112.2216 | 0.362118 | 0.264271 | 0.399829 | 0.558968 |
| Test | 117.0304 | 0.357728 | 0.260945 | 0.393873 | 0.554481 |

So với hai cấu hình K trên test:

| Student | MR | MRR | Hits@1 | Hits@10 |
|---|---:|---:|---:|---:|
| K1 - Description | 119.17 | 0.35342 | 0.25731 | 0.54666 |
| K2 - Learned RC | 154.85 | 0.34586 | 0.25164 | 0.53765 |
| L9 - Structural+Description | 117.03 | 0.357728 | 0.260945 | 0.554481 |

So với Table 3 của paper:

| Chỉ số | Paper | L9 | Chênh lệch |
|---|---:|---:|---:|
| MR | 116 | 117.0304 | +1.0304 |
| Hits@1 | 0.275 | 0.260945 | -0.014055 |
| Hits@3 | 0.411 | 0.393873 | -0.017127 |
| Hits@10 | 0.564 | 0.554481 | -0.009519 |

L9 có test MRR, H@1 và H@10 cao hơn K1/K2 trong những lần chạy này. Cấu hình dùng strategy cố định và năm epoch, còn mỗi phương pháp chỉ có một lần chạy; chưa có khoảng tin cậy nhiều seed hoặc xác nhận cấu hình khớp toàn bộ paper.

L9 khép lại chuỗi khảo sát K/L. Kết quả tái lập Table 3 sau đó được lưu ở [trang FB15K-237/WN18](../../research/tables/table-3-fb15k237-wn18.md).

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="comparison-with-dsom-table-3---test"></a>
<a id="comparison-with-experiment-k---test"></a>
<a id="conclusion"></a>
<a id="configuration"></a>
<a id="experiment-l9---fixed-structuraldescription-student"></a>
<a id="findings"></a>
<a id="full-filtered-evaluation"></a>
<a id="limitations"></a>
<a id="objective"></a>
