# Bàn giao reproduction DSoM — 2026-10-16

Handoff lịch sử mô tả trạng thái ngày 2026-10-16. Setup/workflow hiện tại ở
[README](../../../README.md) và [hướng dẫn kết quả](../../reproduction/RESULTS.md).
Cập nhật 2026-10-24: full WN9/Table 7 PASS artifact review, filtered MRR .919634,
raw .618763, source-frozen RC; author protocol còn mở. Xem
[báo cáo Table 7](../../experiments/TABLE7_WN9_REPRODUCTION.md).

## Nguồn thông tin chính
- Repository: https://github.com/keiranoluv/Reproduce_DSoM
- Nhánh làm việc: `teacher-compat-fixes`
- HEAD được review: `594ce6d510fd920d134cdf5e5187282b1e8118f3`
- Nhánh mặc định `main`, chậm 3 commits so với `teacher-compat-fixes` tại thời điểm kiểm tra.
- Upstream: https://github.com/OreOZhao/DSoM
- Mục tiêu: reproduce DSoM ACM MM 2025, Table 3 FB15K-237 student MR 116 / H@1 27.5% / H@3 41.1% / H@10 56.4%.

## Đã hoàn thành
1. Chuyển FB15K-237 sang kbc pickle, align IDs/features. Có 14,541 entities, 237 raw relations, 272,115/17,535/20,466 train/valid/test triples; reciprocal teacher dùng 474 relations, 544,230 train triples.
2. Text `data/FB15K-237/dscp_vec.pickle` shape (14541,768), bert-base-uncased, mean pooling hidden state áp chót, thay 26 descriptions thiếu bằng random vectors.
3. Image `data/FB15K-237/img_vec.pickle` shape (14541,1000), pytorch_pretrained_vit B_16_imagenet1k, input 384x384; 244 missing random vectors. Khác vit-base-patch16-224 paper, cần ghi rõ.
4. Tương thích PyTorch 2.7 teacher: thêm `--modal_missing 0.0`, bỏ unsupported `ReduceLROnPlateau(verbose=True)` trong src/learn.py.
5. Thêm scripts/train_teacher_fb15k237.sh: `--smoke` (1 epoch, validation 1), `--full` (200 epochs, validation 3), `--no-wandb`; thư mục runs/teacher/FB15K-237/{smoke,full}/YYYYMMDD_HHMMSS; W&B qua DSOM_WANDB, ignore artifacts bằng .gitignore.
6. Teacher smoke PASS trên RTX 5090 32GB/PyTorch 2.7+cu128, env `/workspace/workspace/DSoM/dsom`: epoch ~34.08 giây; loss 19.19335; VALID MR 86.3159, MRR .369794, H@1 .277417, H@3 .401967, H@10 .556715; TEST MR 89.0186, MRR .366091, H@1 .273405, H@3 .399199, H@10 .555678. W&B logs: https://wandb.ai/keiranoluv-hcmus/DSoM-Reproduction/runs/i8hotqby. Checkpoint local runs/.../smoke/20261016_233349/checkpoints/0/m-20261016_2333-n-teacher_fb15k237_smoke_20261016_233349.pth.
7. Đã kiểm tra GitHub files/history. Nhánh hơn main 3 commits, chậm 0 tại thời điểm review: 833518f compatibility, 69191d2 teacher Bash, 594ce6d W&B/timestamp. Chưa merge lúc kiểm tra.

## Nhận xét code review tại thời điểm viết
- `src/learn.py` giữ `ReduceLROnPlateau(mode='min')` dù Hits@10 cần tăng; giữ để theo source baseline, là vấn đề scheduler tiềm tàng.
- Teacher chọn validation Hits@10 dù thông báo nói best Valid MRR.
- `src/learn.py` evaluate test mỗi validation interval; không tune/chọn model theo test.
- Teacher modality-split lấy oracle minimum rank, không so trực tiếp student Table 3.
- README lúc đó còn kiểu upstream, thiếu hướng dẫn chạy/config/data mới; cần cập nhật.
- `.gitignore` loại `runs/`, `ckpt/`, `data/`, không upload artifacts.
- `wandb.define_metric('train_eval/*', step_metric='epoch')` có trailing whitespace, có thể dọn.
- GitHub review tĩnh, không verify GPU runtime; smoke PASS dựa logs người dùng.

## Chưa hoàn thành tại thời điểm viết
- Chưa xác nhận full teacher `--full`; chưa có selected checkpoint/W&B record.
- Chưa train student RC + NDKD, chưa reproduce Table 3 cuối.
- Chưa ablations/datasets khác.
- Chưa giải quyết khác biệt image feature checkpoint.

## Blockers student phát hiện qua GitHub
- `src/kd_learn.py` còn teacher_model_path FB15K-237 là `'YOUR_PATH_TO_TEACHER_MODEL_ComplExMDR'`.
- Truy cập `dataset.train_to_skip` nhưng loader chỉ có `to_skip` chứa train+valid+test. Phải tạo training-only neighbor map với reciprocals; KHÔNG alias `train_to_skip=to_skip` vì leak evaluation.
- `ReduceLROnPlateau(..., verbose=True)` còn không tương thích PyTorch 2.7.
- Full teacher `torch.load` có thể cần trusted `weights_only=False` ở PyTorch mới.
- Sau kiểm tra implementation, full student dùng `--kd_loss_type kdrl --kd_kl_weight 2.0 --kd_temp 4.0 --dkd_alpha 1.0 --dkd_beta 1.0 --rl_hs 1024`. Smoke trước full.

## Hướng dẫn tiếp theo
1. Trước full teacher, kiểm tra đúng branch/không job GPU cũ; `git status -sb`, `git rev-parse --short HEAD`.
2. Chạy `cd /workspace/workspace/DSoM && bash scripts/train_teacher_fb15k237.sh --full`. W&B mặc định bật; khoảng ~23 giây training/epoch cộng validation, phụ thuộc early stopping.
3. Khi xong ghi timestamp, run folder, best epoch/VALID Hits@10, exact `.pth`, SHA/W&B URL/selected test; kiểm tra checkpoint tồn tại.
4. Audit/sửa tối thiểu student path, train-only map, scheduler/load compatibility; tạo `scripts/train_student_fb15k237.sh` theo quy ước timestamp/W&B; smoke rồi full.
5. So structural student filtered ranks với Table 3; sau main result mới cân nhắc ablations.

## Môi trường và đường dẫn
- Server repo `/workspace/workspace/DSoM`
- Python env `/workspace/workspace/DSoM/dsom/bin/python`
- Chạy Bash ở gốc repo; script đổi cwd sang src vì datasets.py dùng `../data`.
- Git remote: https://github.com/keiranoluv/Reproduce_DSoM
- Ghi chép lịch sử yêu cầu sửa file trực tiếp hoặc thay toàn bộ; không dùng Python để patch source.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="completed"></a>
<a id="current-code-review-observations"></a>
<a id="dsom-reproduction-handoff--2026-10-16"></a>
<a id="environment--paths"></a>
<a id="next-instructions"></a>
<a id="not-yet-completed"></a>
<a id="source-of-truth"></a>
<a id="student-code-blockers-discovered-by-github-inspection"></a>
