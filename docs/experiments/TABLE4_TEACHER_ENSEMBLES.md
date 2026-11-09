# Table 4 — Teacher rows và giới hạn protocol

Trạng thái: người dùng đã hoàn tất full teacher-ensemble suite. Trợ lý chỉ
review CPU/files; báo cáo giữ các giới hạn adaptation và protocol.

## Audit student đã hoàn tất

Đã tái tạo `runs/table4_components/full/20261023_075850_052725/summary.json`
bằng kiểm tra CPU/files: PASS. Input hashes gốc khớp, configs khớp A5, selected
test metrics khớp curve entries, checkpoints tồn tại, cả hai runs có policy
updates khác không.

| Component | Epoch tốt nhất | Test MRR | H@1 | H@3 | H@10 | MRR paper | MRR delta (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RC + NEKD | 75 | .411020 | .333115 | .450010 | .560038 | .4100 | +.102 |
| RC + NNKD | 42 | .412105 | .332256 | .452030 | .563775 | .4233 | -1.120 |

Đủ sáu student rows gồm A4/A5/A9/A10. Policy học với Adam, không source-frozen.
NEKD dùng neighbor SUM của released code, không MEAN của paper. NNKD cao hơn
NEKD .1084 pp MRR, nhỏ hơn nhiều mức 1.33 pp của paper. Review không chạy lại
checkpoint inference trên GPU.

## Nguồn và diễn giải

[DSoM §5.3/Table 4](https://arxiv.org/html/2507.20738v1#S5.SS3) mô tả Teacher
Avg. là average performance của ba modal teachers. Cách hiểu ban đầu là
mean-logit ensemble inference chưa chắc chắn. Runner lưu CẢ trung bình metrics
từng modality (`teacher_mean_metrics`, cách hiểu theo nghĩa đen chính) VÀ rank
của mean raw logits (`teacher_mean_logits`, diagnostic), cùng metrics từng
modality. Cả hai khác oracle minimum rank của original teacher evaluation.

[Meta learner MoSE công bố](https://github.com/OreOZhao/MoSE4MKGC/blob/192d9ff2882b5be6ef8239a37cbd90c5dd102b5a/src/meta_learner.py)
áp dụng Linear(3,32), ReLU, Linear(32,1), ReLU độc lập lên ba raw modality scores
của mỗi entity. Dùng validation triples hai chiều, Adagrad LR .1, batch 256,
20 epochs. Giữ kiến trúc/schedule, sửa unrestricted squeeze để giữ singleton
batches. MoSE gốc chọn checkpoint bằng TEST Hits@10; ở đây cố định final epoch
trước khi chạy, chỉ đánh giá test sau fitting. Đây là adaptation kiến trúc MoSE,
không phải source run giống từng bit. Commit/SHA256 nguồn trong `table4_ensemble_sources.json`.

Release/paper DSoM được kiểm tra chưa đủ chi tiết standalone RC-on-validation.
Protocol **RC-on-val-adapted** được chọn tường minh như sau:

- Đóng băng Exp01 teacher, không tạo/train student.
- Raw teacher scores [batch, entities, S/V/D] đi qua policy kiểu DSoM: flatten, dropout .1, Linear(3*entities,1024), ReLU, Linear(1024,7).
- Bảy actions: S, V, D, SV, SD, VD, SVD. Actions nhiều modalities lấy mean raw logits.
- Sample action khi training. Nếu unfiltered target CE tốt hơn average-teacher CE, reward +1; nếu không -10. Reward reference là -10, do đó REINFORCE advantage bằng 11 khi tốt hơn, 0 nếu không. Không entropy term hoặc student comparison bổ sung; dùng Categorical log probabilities ổn định.
- Adam LR 1e-5, cố định 20 epochs trên validation RHS rồi LHS, seed 2022; giữ duplicates, shuffle mỗi chiều mỗi epoch.
- Inference chọn argmax xác định, tắt dropout, không dùng nhãn để chọn action. Chỉ filtered ranking SAU score fusion.

Reward reference và training horizon là lựa chọn reproduction, chưa xác nhận
cấu hình tác giả. Không reuse A5 student-policy checkpoints. Full training dùng
tất cả validation entries gốc, kể cả contamination đã biết. Test labels chỉ
dùng metrics/filtering. Validation metrics sau fitting là in-sample diagnostic,
không ước lượng generalization. Không tìm hyperparameters/chọn checkpoint theo test.

## Kết quả full đã ghi nhận

Suite `runs/table4_ensembles/full/20261023_103437_794958/` PASS. Test MRR:
Teacher Avg. mean metrics .242729, mean logits diagnostic .329994, ML-on-val
.408904, RC-on-val .400841. Đây là teacher ensemble scores, không kết quả train
structural student mới. Kết hợp sáu student rows thì đủ chín dòng có kết quả;
exact author protocol vẫn chưa xác minh.

## Hướng dẫn thực thi

Lệnh chạy hiện tại, chọn từng ô/toàn bảng và prerequisites: [RUNNING.md](../reproduction/RUNNING.md).

Mỗi lần gọi tạo thư mục mới trong `runs/table4_ensembles/{mode}/`. `--output`
chỉ chọn thư mục MỚI trong cây này, từ chối thư mục đã tồn tại. Lưu full source
snapshots, input hashes, config, per-head history, final head/optimizer state
và results. Chỉ checkpoint heads, không nhân bản teacher/features/data.
Hash artifacts gốc trước/sau. Default student suite là run đã hoàn thành
20261023_075850_052725; `--student-suite` thay lựa chọn đó.

Outputs: `result.json` từng method, ML/RC `final_head.pth`, `history.json`,
suite `summary.json` và `table4_results.md` cùng sáu student rows đã có.
Full ba rows đặt `all_nine_rows_have_results=true`, nhưng vẫn
`exact_author_protocol_reproduced=false`: checkpoint selection ML và standalone
RC là adaptations, Teacher Avg. chưa rõ cách hiểu, DB15K overlaps còn nguyên,
chỉ một seed và reconstructed features.

CPU tests bao phủ reciprocal queries/duplicates, pessimistic filtered ties,
kiểm tra rank độc lập, phân biệt mean-metric/mean-logit, kiến trúc MoSE, RC
advantage, chọn action không nhãn, chỉ cập nhật head trên validation, frozen
teacher, reload saved head, default plan và full guards.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="completed-student-audit"></a>
<a id="sources-and-interpretation"></a>
<a id="table-4--remaining-teacher-rows-and-protocol-limits"></a>
