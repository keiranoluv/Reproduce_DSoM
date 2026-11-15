# E-I: theo dõi strategy, reward và policy update

Chuỗi kiểm tra này dùng pretrained MIN teacher FB15K-237. Baseline ở nhánh `student-pipeline-fixes`; nhánh thử nghiệm là `experiment/rc-policy-stabilization`. Mục đích là hiểu hành vi policy trong các cấu hình nhỏ trước khi train đầy đủ.

## E: train 20.000 example

Student dùng Adagrad LR 0.1; policy dùng Adam LR `1e-5`. Chạy ba epoch với batch 16, đánh giá 1.024 query:

| Epoch | Valid MRR | Test MRR | Test Hits@10 |
|---|---:|---:|---:|
| 1 | 0.238054 | 0.226723 | 0.367188 |
| 2 | 0.225990 | 0.241456 | 0.380859 |
| 3 | 0.257907 | 0.263478 | 0.409668 |

Entropy ở minibatch cuối mỗi epoch:

| Epoch | Entropy | Near-deterministic |
|---|---:|---:|
| 1 | 0.081332 | 13/16 |
| 2 | 0.000929 | 15/16 |
| 3 | 0.000003 | 16/16 |

Student có chỉ số tốt hơn trong khi policy ngày càng gần deterministic.

## F: policy chọn teacher tốt tới đâu?

Lấy best distiller của E, tắt dropout và chọn policy argmax trên 1.024 validation query gồm reciprocals. Policy chọn structural ở 100% query, trùng CE oracle 2.6367%. Mean CE của student là 6.519696, của policy 6.272866 và của oracle 4.651416; regret là 1.621450.

| Chiến lược | Frequency |
|---|---:|
| Structural | 2.64% |
| Image | 5.47% |
| Description | 50.78% |
| Structural+Image | 3.71% |
| Structural+Description | 12.11% |
| Image+Description | 20.61% |
| All three | 4.69% |

Oracle dùng ground-truth label để chọn strategy có CE thấp nhất, chỉ dùng để phân tích.

## G: reward có phân biệt strategy không?

Trên cùng checkpoint/query của F, strategy nhận +1 nếu CE thấp hơn student, ngược lại -10. Baseline reward tính từ trung bình ba modality:

| Chiến lược | Mean CE | Positive reward |
|---|---:|---:|
| Structural | 6.2729 | 55.76% |
| Image | 6.3664 | 55.47% |
| Description | 4.9021 | 88.87% |
| Structural+Image | 5.6744 | 74.41% |
| Structural+Description | 5.2009 | 85.64% |
| Image+Description | 5.2135 | 85.16% |
| All three | 5.2242 | 86.52% |

Có 43.55% query mà mọi strategy cùng reward; 91.89% có ít nhất hai strategy nhận +1. Mean CE gap giữa các strategy nhận reward dương là 1.411396. Reward của policy kém reward tốt nhất có thể chọn ở 38.09% query.

Binary reward phân biệt khá ít trong mẫu này, nhưng chưa đủ để xác định nguyên nhân policy collapse.

## H: xác suất đổi theo hướng nào sau update?

Với 2.048 training example, một epoch, student Adagrad 0.1 và policy Adam `1e-5`, xác suất được đo trước/sau trên cùng minibatch, tắt dropout:

- Batch 3 có một advantage âm: P(description) giảm 0.091313, P(structural) tăng 0.002867.
- Batch 8 có hai advantage âm: P(image) giảm 0.108906, P(structural) tăng 0.039095.

## I: advantage ở 20 batch đầu

Trong 320 example, có 0 advantage dương, 7 âm và 313 bằng zero. Bảy action bị phạt gồm image 5 lần, description 1 lần, image+description 1 lần. Description ở batch 3 nhận -11; hai image selection ở batch 8 cũng nhận -11 mỗi selection.

Hướng đổi xác suất phù hợp REINFORCE phạt action đã sample. Chưa xác định các update đầu có đủ gây ra policy structural-only cuối cùng hay không.

## Đọc tiếp

E chỉ chạy ba epoch trên tập nhỏ; F/G dùng sampled validation, H/I chỉ đo 20 batch đầu. [J](j-fixed-strategies.md) tiếp tục so các strategy cố định với learned RC, cố gắng giữ cùng initialization, subset, optimizer, NDKD và cách đánh giá.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="dsom-rc-policy-diagnostics--fb15k-237"></a>
<a id="experiment-e--medium-training"></a>
<a id="experiment-f--strategy-quality-audit"></a>
<a id="experiment-g--reward-signal-audit"></a>
<a id="experiment-h--update-direction-audit"></a>
<a id="experiment-i--advantage-attribution"></a>
<a id="limitations"></a>
<a id="next-experiment-j"></a>
<a id="scope"></a>
