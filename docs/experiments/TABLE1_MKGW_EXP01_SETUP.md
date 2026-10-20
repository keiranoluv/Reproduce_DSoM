# DSoM Table 1 — MKG-W Exp01 Setup

## Purpose and provenance

- Source: [MMRNS](https://github.com/quqxui/MMRNS); use the original released splits.
- Expected MKG-W: 15,000 entities; 169 relations; train 34,196; valid 4,276; test 4,274.
- Experiment name: `T1_MKGW_Exp01` (the `L` prefix is reserved for RC investigations).
- Teacher: ComplExMDR, three modalities. Student: structural ComplExMDR, RC source-frozen.
- Reuses MKG-Y baseline optimizer, scheduler, KD hyperparameters, early stopping and feature encoders.
- Feature preparation is an **assumption of this reproduction** and may not exactly match author assets.

## MKG-W-specific alignment

Structural triples contain Wikidata URIs; raw descriptions and raw-image entity folders are DBpedia-keyed.
`src_data/MKG-W/ent_links` maps the structural entity to DBpedia URI.

1. Read `data/MKG-W/ent_id` to establish numerical entity order.
2. Map Wikidata URI to DBpedia resource key via `ent_links`.
3. Use that key to read a raw description or image folder.
4. Keep missing modalities as seeded Gaussian random vectors.

Crucial limitation: MMRNS has image HDF5 features for many entities which have **no image folder** in the separately released raw archive. HDF5 image vectors are not DSoM's required `(15000,1000)` ViT output and must not be substituted silently. The raw ZIP has approximately 9,301 entity folders, not 15,000.

## Commands (repository root)

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

## Records and metadata

- Raw dataset files are kept local and Git-ignored (`src_data/MKG-W/`).
- `data/MKG-W/structural_manifest.json` records split checksums.
- Text and image feature manifests record coverage and representation details.
- Separate directories under `runs/teacher/MKG-W/` and `runs/student/MKG-W/`.
- Full training requires committed source, stores Git commit SHA in both `result.json` and W&B config, groups two runs by W&B group ID.
- Final Table 1 metrics must come from Student **best-checkpoint** `result.json/final_result`, not final logged W&B `test/*` values.
- Expected paper MKG-W Student targets: MRR 40.90%; Hits@1 33.56%; Hits@3 43.58%; Hits@10 55.28%.
- After the run, record exact checkpoint, final metrics, Git SHA, W&B run URLs and deviations from paper.
