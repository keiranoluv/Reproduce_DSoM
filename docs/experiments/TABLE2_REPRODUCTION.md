# Table 2 — Workflow reproduction teacher selection

Trạng thái: **full workflow Table 2 đã hoàn thành local**; vẫn giữ giới hạn
source-frozen baseline và MulDE-adapted. File này ghi protocol nghiên cứu.
Mặc định launcher chỉ ghi plan. Suite đã hoàn thành được nêu bên dưới;
hướng dẫn thực thi và bảng số liệu nằm trong tài liệu này.

## Phạm vi và đối chiếu

Mục tiêu: [DSoM v1, Table 2 và Section 5.4](https://arxiv.org/html/2507.20738v1).
Chạy bốn student comparisons cho mỗi dataset MMKB-DB15K, MKG-W, MKG-Y:

| CLI method | Nhãn báo cáo | Chọn teacher / supervision |
|---|---|---|
| `best_conf` | ConfTeacher | Maximum softmax probability cao nhất trong S/V/D, sau đó NDKD |
| `best_teacher` | BestTeacher | Ground-truth CE thấp nhất trong S/V/D, sau đó NDKD |
| `best_strategy` | BestStrategy | CE thấp nhất giữa bảy subset means S/V/D, sau đó NDKD |
| `mulde` | MulDE-adapted | Top-K của student, Senior scaling và contrast attention có học; KL + Senior BCE |

Có 12 student runs mới. Exp01 source-frozen RC cung cấp ba ô reference,
tổng cộng 15 ô method/dataset. Test time chỉ đánh giá student; teacher selection
phụ thuộc nhãn chỉ dùng trong training batches. NDKD neighbors lấy từ training
triples và reciprocals qua `Dataset.train_to_skip`.

Ở legacy lookup, baseline result JSON được lấy từ `runs/student/DATASET/full/`
của từng dataset. Đúng teacher checkpoint, feature paths và training settings
được chép vào plan để review. Không cần retrain teacher. Launcher từ chối
reference không duy nhất; fresh clone dùng explicit registry như phần hướng dẫn thực thi bên dưới.

DB15K dùng benchmark split gốc với validation contamination đã biết.
**CleanV1 và mọi kết quả trước đó được giữ riêng.** DSoM reference reuse có
nhãn `DSoM-source-frozen`, không phải learned-RC. Paper targets của bốn metrics
lưu dưới dạng tỷ lệ trong [table2_paper_targets.json](table2_paper_targets.json).

## Các cấu hình được đối chiếu

Sao chép từ Exp01 student tương ứng: rank, batch size, optimizer, learning rate,
regularizer/strength, scheduler, temperature, CE/KD weights, alpha/beta,
initialization, scaling, optimizer betas, epoch limit, validation interval và
early stopping. Cấu hình hiện tại: rank 2000, batch 1000, Adagrad LR 0.1,
N3 0.01, temperature 4, CE/KD weights 1/2, tối đa 200 epochs, validation
mỗi 3 epochs, seed 2022.

Giữ scheduler `min` theo dõi validation Hits@10 để khớp baseline, kể cả
chiều theo dõi không phù hợp đã biết. Chọn checkpoint bằng validation MRR.
Bảng kết quả đọc `final_result`, không lấy `test` epoch cuối hoặc W&B summary.
Test metrics do training code log không được dùng để chọn checkpoint.

Ba heuristic classes giữ hành vi selection công bố: ConfTeacher softmax
raw scores ở temperature 1; BestTeacher lấy trung bình các teachers đồng hạng
CE nhỏ nhất; BestStrategy lấy minimum đầu tiên trong `S,V,D,SV,SD,VD,SVD`,
mỗi tổ hợp dùng trung bình logits. `BestStrategy.strategy` đã được đăng ký
làm buffer để đi theo `.to(device)`.

## Giới hạn MulDE adaptation

[DSoM source công khai](https://github.com/OreOZhao/DSoM), được kiểm tra ở commit
`171e59cb069d50abf039fa824b079e23b8956696`, có heuristic selectors nhưng không
cung cấp MulDE adaptation hoặc các thiết lập còn thiếu. Nguồn tree listing
đã lưu tại `runs/audit/table2_sources/`.

Comparison theo [MulDE gốc, Sections 3.2–3.4, Equations 7 và 10–14](https://arxiv.org/html/2010.07152v3),
với các lựa chọn tường minh cho multimodal DSoM:

- Ba pretrained ComplEx modality branches thay hyperbolic teachers gốc; luôn đóng băng.
- Junior tự chọn top 300 candidates (`min(300, entity_count)`), không ép positive target vào danh sách.
- Senior dùng sigmoid relation-specific scaling matrix, gồm reciprocal relation IDs, khởi tạo zero trước sigmoid nên scale đầu là 0.5.
- Contrast attention dùng unscaled teacher top-K distributions và detached Junior distributions. Equation 11 lấy trung bình KL trên K; temperature là `exp(floor(e/5))` với epoch `e` từ zero; Equation 12 nhân weights với số teachers, ba.
- Senior BCE labels cho biết true training target có nằm trong top-K không. Nếu không, cả vector nhãn là zero. Equation 13 cộng trên K rồi lấy trung bình queries.
- Student giữ full-entity CE và N3 của DSoM thay sampled hard-label BCE gốc. Soft supervision là temperature-4 KL trên top-K, weight 2, không NDKD. Senior BCE weight 1. KL và Senior BCE cùng cập nhật Senior; Junior input của attention được detach.
- Student dùng matched Adagrad; Senior dùng Adam riêng ở 0.001, không theo student scheduler. Kiểm tra optimizer memberships; epoch đầu phải có Senior weight update khác không.

Đây là **MulDE-adapted**, chưa xác minh tái lập chính xác DSoM/MulDE adaptation
không công bố của tác giả. Có thể đối chiếu con số với paper nhưng phải giữ
nhãn adaptation và các giả định này. Không tìm hyperparameters hoặc tune theo test.

## Hướng dẫn thực thi

Cần full teacher/student baselines của cả ba datasets và registry chứa các
baseline đã chọn. Chạy từ gốc repo trong môi trường đã cài dependencies:

```bash
python scripts/run_table2.py --mode plan --references references.local.json
python scripts/run_table2.py --mode smoke --workers 1 --references references.local.json
python scripts/run_table2.py --mode full --workers 1 --references references.local.json
```

Full mode yêu cầu source/config đã được review và commit. Để chạy một ô,
thêm `--datasets MKG-Y --methods best_conf` (hoặc `best_teacher`,
`best_strategy`, `mulde`). Full tạo run mới, không tiếp tục từ smoke.

Để tái tạo summary của suite có sẵn, không training lại:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/run_table2.py \
  --mode summarize --output runs/table2/full/20261023_055906_392721
```

Thư mục `runs/` là artifact local, không được đóng gói trong Git; fresh clone
cần tạo artifacts bằng workflow hoặc có bản sao của suite trước khi summarize.

## Kết quả và trạng thái

Sau training, `summary.json` và `table2_results.md` chứa MRR/Hits@1/3/10 của
selected student, paper targets, result paths và MRR differences. Smoke không
tính chênh lệch với paper. Kiểm tra lại mọi input/baseline hashes trước summarize.

Cách tái tạo summary được ghi trong phần hướng dẫn thực thi ở trên.

`status=PASS` nghĩa là các ô yêu cầu đã qua kiểm tra. Subset có thể PASS
nhưng chưa hoàn thành Table 2. Chỉ `full_benchmark_grid_completed=true`
xác nhận đủ 12 full-training cells mới trong suite. Kết quả vẫn có giới hạn
source-frozen/MulDE-adapted, reconstructed features, một seed và DB15K contamination.

Full suite đã ghi nhận: `runs/table2/full/20261023_055906_392721/`;
hoàn tất 12 students mới và 3 baseline rows reuse. Đây là reproduction profile
được ghi nhận, chưa xác nhận đúng cấu hình tác giả.

## Kết quả full đã xác minh — 2026-10-23

Suite báo `status=PASS`, `training_mode=full`, `completed_new_cells=12`,
`full_benchmark_grid_completed=true` và `failures=[]`. Ba dòng DSoM reuse
baseline, tổng cộng đủ 15 tổ hợp method/dataset của Table 2.

Snapshot có thể đọc từ fresh clone:
[table2_results_20261023.json](table2_results_20261023.json).
Snapshot lưu đủ metrics, paper targets, epoch, đường dẫn artifact tương đối,
SHA256 của từng result và provenance của suite. Source commit khi chạy:
`5c77bb2610ebd4e88443c2e33fde2ab8de6e0980`; seed `2022`.

Ngày 2026-11-08 đã đối chiếu cả 15 file kết quả gốc: `best_epoch`,
MRR và Hits@1/3/10 trong `final_result` đều khớp summary. Các paper targets
được đối chiếu với Table 2, trang 7 của DSoM v1. Đây là kiểm tra artifacts
đã lưu; không chạy lại training hoặc inference từ checkpoint.

Các metrics dưới đây dùng thang **0–1** (paper trình bày theo phần trăm).
`Delta MRR = reproduced − paper`; nhân 100 để đổi sang điểm phần trăm.
Epoch là checkpoint được chọn theo validation MRR.

| Dataset | Method | Epoch | Test MRR | H@1 | H@3 | H@10 | Paper MRR | Delta MRR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MMKB-DB15K | DSoM-source-frozen | 63 | 0.413046 | 0.331701 | 0.454656 | 0.565744 | 0.4268 | -0.013754 |
| MKG-W | DSoM-source-frozen | 54 | 0.405840 | 0.335400 | 0.431914 | 0.537319 | 0.4090 | -0.003160 |
| MKG-Y | DSoM-source-frozen | 42 | 0.406919 | 0.360120 | 0.424146 | 0.489861 | 0.4002 | +0.006719 |
| MMKB-DB15K | ConfTeacher | 18 | 0.408126 | 0.329630 | 0.446576 | 0.558221 | 0.4087 | -0.000574 |
| MMKB-DB15K | BestTeacher | 18 | 0.409742 | 0.329529 | 0.448950 | 0.563623 | 0.4143 | -0.004558 |
| MMKB-DB15K | BestStrategy | 57 | 0.415297 | 0.336599 | 0.454504 | 0.565391 | 0.4165 | -0.001203 |
| MMKB-DB15K | MulDE-adapted | 42 | 0.387925 | 0.316350 | 0.423854 | 0.523379 | 0.3994 | -0.011475 |
| MKG-W | ConfTeacher | 21 | 0.377240 | 0.318320 | 0.398690 | 0.491811 | 0.3749 | +0.002340 |
| MKG-W | BestTeacher | 30 | 0.377255 | 0.316799 | 0.402901 | 0.489120 | 0.3827 | -0.005445 |
| MKG-W | BestStrategy | 15 | 0.382127 | 0.320075 | 0.405241 | 0.497426 | 0.3914 | -0.009273 |
| MKG-W | MulDE-adapted | 9 | 0.373181 | 0.310716 | 0.394712 | 0.493098 | 0.3868 | -0.013619 |
| MKG-Y | ConfTeacher | 18 | 0.385972 | 0.350169 | 0.401615 | 0.452122 | 0.3776 | +0.008372 |
| MKG-Y | BestTeacher | 18 | 0.387228 | 0.350920 | 0.404431 | 0.450244 | 0.3851 | +0.002128 |
| MKG-Y | BestStrategy | 15 | 0.389047 | 0.347353 | 0.411190 | 0.459820 | 0.3934 | -0.004353 |
| MKG-Y | MulDE-adapted | 12 | 0.389991 | 0.345851 | 0.408937 | 0.463950 | 0.3962 | -0.006209 |

### Diễn giải và giới hạn

- Trên DB15K, BestStrategy đạt MRR 0.415297, cao hơn DSoM-source-frozen
  0.413046. Trên MKG-W và MKG-Y, DSoM-source-frozen có MRR cao nhất trong
  các dòng đã chạy. Vì vậy thứ hạng chưa tái hiện hoàn toàn kết luận của paper
  rằng DSoM đứng đầu cả ba datasets.
- DSoM baseline dùng RC policy được khởi tạo và sampling nhưng không được
  optimizer cập nhật. Các kết quả này chưa chứng minh lợi ích của learned RL.
  MulDE-adapted giữ các giả định được mô tả ở trên vì thiếu thiết lập adaptation
  của tác giả.
- Features BERT/ViT được tái dựng; DB15K giữ validation contamination của
  benchmark gốc. Chỉ có một seed, không suy ra ý nghĩa thống kê từ chênh lệch.
  **Đã hoàn thành đủ bảng theo protocol repo; chưa xác nhận exact reproduction
  của protocol hoặc kết luận paper.**

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="mulde-adaptation-boundary"></a>
<a id="results-and-status"></a>
<a id="scope-and-reference"></a>
<a id="table-2--teacher-selection-reproduction-workflow"></a>
<a id="what-is-matched"></a>
