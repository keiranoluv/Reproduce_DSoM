# DSoM Table 1 — Thiết lập MKG-W Exp01

## Mục đích và nguồn gốc

- Nguồn [MMRNS](https://github.com/quqxui/MMRNS), giữ splits công bố gốc.
- MKG-W kỳ vọng: 15,000 entities, 169 relations, train 34,196, valid 4,276, test 4,274.
- Tên thí nghiệm `T1_MKGW_Exp01`; tiền tố `L` dành cho nghiên cứu RC.
- Teacher ComplExMDR ba modalities; student structural ComplExMDR, RC source-frozen.
- Reuse optimizer/scheduler/KD hyperparameters/early stopping/encoders baseline MKG-Y.
- Chuẩn bị features là **giả định reproduction**, chưa chắc khớp assets tác giả.

## Alignment riêng cho MKG-W

Structural triples chứa Wikidata URIs; descriptions/ảnh dùng DBpedia keys.
`src_data/MKG-W/ent_links` nối structural entity sang DBpedia URI.

1. Đọc `data/MKG-W/ent_id` xác lập thứ tự entity.
2. Map Wikidata URI sang DBpedia resource key qua `ent_links`.
3. Dùng key đọc description hoặc image folder.
4. Missing modality giữ Gaussian random vectors có seed.

Giới hạn quan trọng: MMRNS có HDF5 image features cho nhiều entities **không có
image folder** trong raw archive riêng. Không tự thay HDF5 vectors cho ViT
output `(15000,1000)` DSoM. Raw ZIP khoảng 9,301 entity folders, không 15,000.

## Lệnh chạy (repository root)

```bash
cd /workspace/workspace/DSoM

# Dataset stage/preprocessing: runs only when data/MKG-W is absent.
./dsom/bin/python -m scripts.prepare_mkgw_dataset

# CPU-only integrity and alignment audits. Stop and investigate if either fails.
./dsom/bin/python -m scripts.mkgw_alignment
./dsom/bin/python -m scripts.build_mkgw_text_features --audit-only
./dsom/bin/python -m scripts.build_mkgw_image_features --audit-only

# Independent GPU feature extractions; run one at a time.
./dsom/bin/python -m scripts.build_mkgw_text_features --device cuda --batch-size 16 --save-every 100
./dsom/bin/python -m scripts.build_mkgw_image_features --batch-size 8 --save-every 300

./dsom/bin/python -m scripts.validate_mkgw_features

# Enable MKG-W in the Teacher CLI, commit the entire setup, then smoke.
./dsom/bin/python -m scripts.enable_mkgw_teacher
bash scripts/smoke_test_mkgw.sh

# After smoke PASS, run full Teacher -> Student (logs to W&B).
bash scripts/run_mkgw_exp01_full.sh
```

## Hồ sơ và metadata

- Giữ raw files local, Git ignore `src_data/MKG-W/`.
- `data/MKG-W/structural_manifest.json` ghi split checksums.
- Text/image manifests ghi coverage và representation.
- Teacher/student giữ riêng dưới `runs/teacher/MKG-W/`, `runs/student/MKG-W/`.
- Full cần source commit, ghi SHA vào `result.json`/W&B config, group hai runs theo W&B group ID.
- Final metrics lấy student **best-checkpoint** `result.json/final_result`, không W&B test cuối.
- Paper student targets: MRR 40.90%, Hits@1 33.56%, Hits@3 43.58%, Hits@10 55.28%.
- Sau run, ghi checkpoint/metrics/SHA/W&B URLs và khác biệt paper.

<!-- Giữ anchors trước khi Việt hóa để tương thích liên kết cũ. -->
<a id="commands-repository-root"></a>
<a id="dsom-table-1--mkg-w-exp01-setup"></a>
<a id="mkg-w-specific-alignment"></a>
<a id="purpose-and-provenance"></a>
<a id="records-and-metadata"></a>
