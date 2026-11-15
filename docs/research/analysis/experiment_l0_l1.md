# Thí nghiệm L0/L1 — RC policy tập trung sớm

## Thiết lập

Dataset: FB15K-237
Mẫu training: 2048
Batch size: 16
Số optimizer steps: 128
Số epochs: 1
Learning rate policy: 1e-5
Reward: +1 / -10

L0: Adam beta1=0.9
L1: Adam beta1=0.0

## Kết quả

| Chỉ số | L0 | L1 |
|---|---:|---:|
| Entropy ban đầu | 1.864328 | 1.864328 |
| Entropy cuối | 0.169272 | 0.063810 |
| Xác suất cực đại cuối | 0.944511 | 0.976755 |
| Batches advantage zero | 97/128 | 90/128 |
| Weight updates khi gradient zero | 94 | 0 |

Tỷ lệ samples có advantage khác không của L0: 1.8066%.

## Chẩn đoán MLP

| Chỉ số | L0 batch 0 | L0 batch 127 | L1 batch 127 |
|---|---:|---:|---:|
| Teacher std | 2.1279 | 2.1016 | 2.1016 |
| Hidden RMS | 0.8562 | 1.8736 | 1.8483 |
| Logit range policy | 1.2292 | 17.1699 | 18.0931 |

## Diễn giải

- Policy tập trung trong 128 training steps.
- Bỏ first-moment momentum Adam không ngăn được concentration.
- Teacher logit std tương đối ổn định.
- RC policy logit range tăng mạnh.
- Chưa xác nhận nguồn gây khuếch đại logits.

## Giới hạn

Đây là diagnostic runs ngắn, không full-data experiments. Đổi beta1 cũng
đổi training trajectory. Chưa thiết lập giải thích nhân quả.

## Tiếp theo

L2: normalize policy input, giữ teacher logits dùng trong reward và KD không đổi.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="experiment-l0l1---early-rc-policy-concentration"></a>
<a id="interpretation"></a>
<a id="limitations"></a>
<a id="mlp-diagnostics"></a>
<a id="next"></a>
<a id="results"></a>
<a id="setup"></a>
