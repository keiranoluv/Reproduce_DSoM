# Table 2: so sánh cách chọn teacher

Table 2 gồm 15 tổ hợp phương pháp/dataset: 12 student train mới và 3 baseline DSoM dùng lại. Suite `runs/table2/full/20261023_055906_392721/` đã hoàn tất toàn bộ bảng với seed 2022, commit `5c77bb2610ebd4e88443c2e33fde2ab8de6e0980`.

## Kết quả

[Bản tổng hợp JSON](reference/table2_results_20261023.json) lưu đủ chỉ số, giá trị paper, epoch, đường dẫn và hash của từng file kết quả. Ngày 2026-11-08, cả 15 file gốc đã được đối chiếu lại với summary; giá trị paper được kiểm tra với Table 2, trang 7 của DSoM v1.

Các chỉ số dùng thang 0–1. Δ MRR = tái lập − paper; nhân 100 để đổi sang điểm phần trăm. Epoch thuộc checkpoint được chọn theo validation MRR.

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

BestStrategy có MRR cao nhất trên DB15K trong các cấu hình đã chạy. DSoM-source-frozen đứng đầu trên MKG-W và MKG-Y. Thứ hạng này chưa khớp hoàn toàn với paper, nơi DSoM đứng đầu cả ba dataset.

## Các phương pháp được chạy

Đối chiếu [DSoM v1, Table 2 và §5.4](https://arxiv.org/html/2507.20738v1):

| CLI method | Nhãn báo cáo | Chọn teacher / supervision |
|---|---|---|
| `best_conf` | ConfTeacher | Maximum softmax probability cao nhất trong S/V/D, sau đó NDKD |
| `best_teacher` | BestTeacher | Ground-truth CE thấp nhất trong S/V/D, sau đó NDKD |
| `best_strategy` | BestStrategy | CE thấp nhất giữa bảy subset means S/V/D, sau đó NDKD |
| `mulde` | MulDE-adapted | Top-K của student, Senior scaling và contrast attention có học; KL + Senior BCE |

Teacher selection chỉ diễn ra trong training. Khi đánh giá, mỗi phương pháp dùng structural student. NDKD lấy neighbor từ training triples và reciprocals qua `Dataset.train_to_skip`.

ConfTeacher lấy maximum softmax probability ở temperature 1. BestTeacher lấy trung bình các teacher đồng hạng CE nhỏ nhất. BestStrategy chọn minimum đầu tiên trong thứ tự `S,V,D,SV,SD,VD,SVD`; mỗi tổ hợp lấy trung bình logits. Strategy tensor được đăng ký buffer để đi theo thiết bị của model.

## Cấu hình chung

Mỗi student dùng cùng teacher, features và các tham số của baseline Exp01 tương ứng: rank 2000, batch 1000, Adagrad LR 0.1, N3 0.01, temperature 4, CE/KD weights 1/2, tối đa 200 epoch, validation mỗi 3 epoch và seed 2022. Các tham số còn lại như alpha/beta, init, scale, optimizer betas và early stopping cũng được chép từ baseline.

Scheduler giữ `min` trên validation Hits@10 để khớp lần chạy gốc, dù chỉ số này cần tăng. Checkpoint student được chọn theo validation MRR; số liệu lấy từ `final_result`. DB15K giữ nguyên validation có overlap của bộ công khai. Dòng DSoM dùng `source_frozen`, còn CleanV1 được ghi riêng.

## Cách cài đặt MulDE adaptation

Source DSoM ở commit `171e59cb069d50abf039fa824b079e23b8956696` có các heuristic selector nhưng thiếu cấu hình MulDE adaptation. Bản kiểm tra nguồn nằm ở `runs/audit/table2_sources/`. Repo triển khai theo [MulDE, §3.2–3.4, Eq.7 và Eq.10–14](https://arxiv.org/html/2010.07152v3), với các lựa chọn sau:

- Dùng ba nhánh ComplEx đã train và đóng băng thay cho hyperbolic teacher của MulDE.
- Junior chọn top `min(300, entity_count)` candidate, không ép target vào danh sách.
- Senior dùng sigmoid relation-specific scaling, gồm reciprocal relation IDs. Trọng số trước sigmoid khởi tạo bằng zero, nên scale ban đầu là 0.5.
- Contrast attention dùng teacher top-K distributions chưa scale và Junior distributions đã detach. Eq.11 lấy trung bình KL trên K, temperature `exp(floor(e/5))` với epoch bắt đầu từ zero; Eq.12 nhân weights với số teacher là 3.
- Senior BCE đánh dấu target nếu nó có trong top-K; nếu không, cả vector nhãn bằng zero. Eq.13 cộng trên K rồi lấy trung bình query.
- Student giữ full-entity CE và N3 của DSoM. Soft loss là KL top-K ở temperature 4, weight 2; Senior BCE weight 1. Cả KL và BCE cập nhật Senior, còn Junior input của attention được detach. Cấu hình này không dùng NDKD.
- Student dùng Adagrad như baseline. Senior dùng Adam LR 0.001 riêng, không theo student scheduler; lần chạy kiểm tra Senior có cập nhật trọng số từ epoch đầu.

Vì thiếu thiết lập của tác giả, bảng giữ nhãn **MulDE-adapted**. Các lựa chọn trên được cố định trước khi đánh giá test.

## Chạy lại và tổng hợp

Xem [lệnh Table 2](../../reproduction/running.md#table-2) để chọn một ô hoặc toàn bảng. Chỉ cần baseline full của những dataset được chọn. Với toàn bảng:

```bash
python scripts/run_table2.py --mode plan --references references.local.json
python scripts/run_table2.py --mode smoke --workers 1 --references references.local.json
python scripts/run_table2.py --mode full --workers 1 --references references.local.json
```

Để tạo lại summary của suite đã có:

```bash
CUDA_VISIBLE_DEVICES='' python scripts/run_table2.py \
  --mode summarize --output runs/table2/full/20261023_055906_392721
```

Lệnh summarize cần các file của suite trên máy. Kết quả gồm `summary.json` và `table2_results.md`; suite đã lưu có `completed_new_cells=12`, `full_benchmark_grid_completed=true`, `failures=[]`. Một subset vẫn có thể PASS mà chưa đủ toàn bảng.

Kết quả dùng một seed, đặc trưng dựng lại và DB15K split gốc. Xem [các khác biệt với paper](../methodology/protocol.md) trước khi diễn giải chênh lệch giữa các phương pháp.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="mulde-adaptation-boundary"></a>
<a id="results-and-status"></a>
<a id="scope-and-reference"></a>
<a id="table-2--teacher-selection-reproduction-workflow"></a>
<a id="what-is-matched"></a>
