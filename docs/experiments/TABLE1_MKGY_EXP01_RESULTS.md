# DSoM Table 1 — MKG-Y Exp01 Reproduction

## 1. Experiment Information

- Experiment ID: T1_MKGY_Exp01
- Dataset: MKG-Y
- GPU: NVIDIA RTX 5090 32GB
- Seed: 2022
- Git commit: 39c7886f8bfd728e9c79afb11a0800ae0ac4fad7
- Teacher: ComplExMDR (Structure + Image + Description)
- Student: Structure-only ComplExMDR
- Distillation: RC, source_frozen

## 2. Dataset

Source: MMRNS
https://github.com/quqxui/MMRNS

- Entities: 15,000
- Original relations: 28
- Train triples: 21,310
- Validation triples: 2,665
- Test triples: 2,663

Structural preprocessing and multimodal feature validation
completed successfully.

Feature matrices:
- Image: (15000, 1000), ViT B/16
- Text: (15000, 768), BERT-base-uncased

## 3. Training Configuration

| Parameter | Value |
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

## 4. Training Summary

Teacher:
- Best epoch: 36
- Total epochs: 47
- Checkpoint selection: Best Validation Hits@10

Student:
- Best epoch: 42
- Total epochs: 53
- Checkpoint selection: Best Validation MRR
- Best Validation MRR: 0.38296

Student verification:
- Full training dataset used
- Full validation and test evaluation
- Teacher checkpoint loaded successfully
- Source-frozen optimizer parameter check passed
- Student checkpoint saved
- W&B synchronization completed
- Full pipeline: PASS

## 5. Student Test Results

Official reproduction results are read from final_result
in the Student result.json at the best validation checkpoint.

| Metric | DSoM Paper | Exp01 | Difference |
|---|---:|---:|---:|
| MR | Not recorded here | 468.4587 | N/A |
| MRR (%) | 40.02 | 40.69 | +0.67 pp |
| Hits@1 (%) | 35.67 | 36.01 | +0.34 pp |
| Hits@3 (%) | 41.70 | 42.41 | +0.71 pp |
| Hits@10 (%) | 47.71 | 48.99 | +1.28 pp |

Exact reproduced metrics:
- MR: 468.45867919921875
- MRR: 0.4069194197654724
- Hits@1: 0.36012017726898193
- Hits@3: 0.4241456985473633
- Hits@10: 0.48986104130744934

## 6. Experiment Tracking

W&B Student run:
https://wandb.ai/keiranoluv-hcmus/DSoM-Reproduction/runs/2a9t5tjk

Run ID:
20261020_174808_2562445

Teacher results:
runs/teacher/MKG-Y/full/

Student results:
runs/student/MKG-Y/full/

Student result.json is the source of the final numbers,
not the last logged W&B test metrics.

## 7. Reproduction Limitations

- Text and image feature generation are reproduction
  choices; exact author preprocessing is not fully verified.
- The source_frozen RC implementation samples strategies
  without updating RC policy network parameters.
- This does not establish reproduction of learned-policy
  REINFORCE behavior described in the paper.
- Teacher evaluation includes oracle-style modality
  ranking and must not be compared directly to Table 1.
- This report documents one experiment run, not
  statistical variance across multiple seeds.

## 8. Conclusion

MKG-Y Exp01 completed successfully.

The Student test results are close to and slightly above
the reported DSoM Table 1 metrics across MRR and Hits@K.

Next dataset: MKG-W.
