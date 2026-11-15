# L0/L1: bỏ momentum có giúp policy bớt tập trung?

L0 và L1 dùng 2.048 training example FB15K-237, batch 16, một epoch tương ứng 128 optimizer step. Policy dùng Adam LR `1e-5`, binary reward +1/-10. L0 giữ beta1=0.9; L1 đặt beta1=0.0.

## Kết quả

| Chỉ số | L0 | L1 |
|---|---:|---:|
| Entropy ban đầu | 1.864328 | 1.864328 |
| Entropy cuối | 0.169272 | 0.063810 |
| Xác suất cực đại cuối | 0.944511 | 0.976755 |
| Batches advantage zero | 97/128 | 90/128 |
| Weight updates khi gradient zero | 94 | 0 |

L0 có 1.8066% sample nhận advantage khác không. Khi gradient bằng zero, momentum vẫn làm weights thay đổi ở 94 batch của L0; L1 không có hiện tượng đó. Tuy vậy, entropy cuối L1 còn thấp hơn L0.

## Theo dõi hidden activation và logits

| Chỉ số | L0 batch 0 | L0 batch 127 | L1 batch 127 |
|---|---:|---:|---:|
| Teacher std | 2.1279 | 2.1016 | 2.1016 |
| Hidden RMS | 0.8562 | 1.8736 | 1.8483 |
| Logit range policy | 1.2292 | 17.1699 | 18.0931 |

Teacher logit std khá ổn định, còn hidden RMS và khoảng logits policy tăng mạnh. Bỏ first-moment momentum chưa ngăn được policy tập trung trong 128 step này.

Hai lần chạy ngắn có quỹ đạo training khác nhau và chưa xác định nguồn khuếch đại logits. [L2](l2-normalized-input.md) tiếp tục thử normalize input, giữ logits dùng cho reward/KD như cũ.

<!-- Liên kết từ các phiên bản tài liệu trước. -->
<a id="experiment-l0l1---early-rc-policy-concentration"></a>
<a id="interpretation"></a>
<a id="limitations"></a>
<a id="mlp-diagnostics"></a>
<a id="next"></a>
<a id="results"></a>
<a id="setup"></a>
