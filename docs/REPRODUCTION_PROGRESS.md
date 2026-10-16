# DSoM reproduction handoff — 2026-10-16

## Source of truth
- Repository: https://github.com/keiranoluv/Reproduce_DSoM
- Working branch: `teacher-compat-fixes`
- Reviewed HEAD: `594ce6d510fd920d134cdf5e5187282b1e8118f3`
- Default branch: `main`, currently 3 commits behind `teacher-compat-fixes` (as inspected).
- Upstream: https://github.com/OreOZhao/DSoM
- Goal: reproduce DSoM ACM MM 2025, main result Table 3, FB15K-237 student MR 116 / H@1 27.5% / H@3 41.1% / H@10 56.4%.

## Completed
1. Dataset FB15K-237 transformed to kbc pickle format; entity IDs aligned with modality vectors. 14,541 entities, 237 raw relations, 272,115/17,535/20,466 train/valid/test triples. Teacher uses reciprocal relations: 474 and 544,230 training triples.
2. Text features `data/FB15K-237/dscp_vec.pickle` shape (14541,768), bert-base-uncased, mean-pooling penultimate hidden states, 26 missing descriptions replaced with random vectors.
3. Image features `data/FB15K-237/img_vec.pickle` shape (14541,1000), pytorch_pretrained_vit B_16_imagenet1k, 384x384 preprocessing, 244 missing replaced with random vectors. IMPORTANT: differs from paper's described vit-base-patch16-224 feature extraction; document discrepancy.
4. Teacher PyTorch 2.7 compatibility: added `--modal_missing 0.0`; removed unsupported `ReduceLROnPlateau(verbose=True)` in src/learn.py.
5. Added scripts/train_teacher_fb15k237.sh with `--smoke` (1 epoch, validation 1), `--full` (200 epochs, validation 3) and `--no-wandb`, per-timestamp dirs under runs/teacher/FB15K-237/{smoke,full}/YYYYMMDD_HHMMSS; W&B flags via DSOM_WANDB, artifact exclusion via .gitignore.
6. Teacher smoke test successful on RTX 5090 32GB with PyTorch 2.7+cu128, env `/workspace/workspace/DSoM/dsom`: one epoch ~34.08 sec; train loss 19.19335; VALID MR 86.3159, MRR .369794, H@1 .277417, H@3 .401967, H@10 .556715; TEST MR 89.0186, MRR .366091, H@1 .273405, H@3 .399199, H@10 .555678. W&B received logs: https://wandb.ai/keiranoluv-hcmus/DSoM-Reproduction/runs/i8hotqby. Best checkpoint saved under local runs/.../smoke/20261016_233349/checkpoints/0/m-20261016_2333-n-teacher_fb15k237_smoke_20261016_233349.pth.
7. Checked GitHub branch files and commit history. Branch is 3 commits ahead and 0 behind main at review time. Commits: 833518f compatibility; 69191d2 teacher Bash; 594ce6d W&B timestamp run management. Not merged into main at inspection time.

## Current code review observations
- `src/learn.py` keeps original source `ReduceLROnPlateau(mode='min')` even though monitored Hits@10 should increase. This is a potential scheduler issue; kept intentionally for source-faithful baseline.
- Teacher checkpoint chosen by validation Hits@10, despite result print saying 'best Valid MRR'.
- `src/learn.py` also evaluates test each validation interval; do not use test metrics for model selection/hyperparameter tuning.
- Teacher modality-split evaluation produces oracle-like/minimum-rank across modalities; NOT comparable directly to final student Table 3.
- README on branch is still upstream-style and does not document new run commands/configuration or dataset recipe. Update later.
- `.gitignore` excludes `runs/`, `ckpt/`, `data/`; these experiment artifacts are not uploaded to GitHub.
- Minor: `wandb.define_metric('train_eval/*', step_metric='epoch')` has trailing whitespace on branch; optional cleanup.
- GitHub review is static, not GPU runtime verification; latest successful smoke result comes from user's supplied logs.

## Not yet completed
- Full teacher training (`--full`) has NOT been confirmed completed; no full-run best checkpoint / W&B record supplied as of this handoff.
- Student RC + NDKD training NOT done, final Table 3 NOT reproduced.
- Ablations / other datasets NOT done.
- Visual feature checkpoint discrepancy unresolved.

## Student code blockers discovered by GitHub inspection
- In `src/kd_learn.py`, `teacher_model_path` for FB15K-237 is still `'YOUR_PATH_TO_TEACHER_MODEL_ComplExMDR'`.
- `src/kd_learn.py` accesses `dataset.train_to_skip`, but `src/datasets.py` only defines `to_skip` containing train+valid+test. Must compute training-only neighbor map including reciprocal relations for NDKD; NEVER alias `train_to_skip=to_skip` (would leak evaluation triples).
- `src/kd_learn.py` still creates `ReduceLROnPlateau(..., verbose=True)` which is incompatible with PyTorch 2.7.
- `torch.load` full teacher object may require trusted `weights_only=False` on newer PyTorch.
- For full paper student command use `--kd_loss_type kdrl --kd_kl_weight 2.0 --kd_temp 4.0 --dkd_alpha 1.0 --dkd_beta 1.0 --rl_hs 1024`, after checking implementation. Student training must be smoke tested first.

## Next instructions
1. Before full teacher: ensure correct branch and no old GPU job; `git status -sb`; `git rev-parse --short HEAD`.
2. Run: `cd /workspace/workspace/DSoM && bash scripts/train_teacher_fb15k237.sh --full`. W&B enabled by default. Time estimate ~23 sec training per epoch plus validation, actual runtime depends on early stopping.
3. On completion, record timestamp, run folder, best epoch and best VALID Hits@10, exact checkpoint `.pth`, Git SHA, W&B run URL and selected test result. Check checkpoint exists.
4. Then audit and minimally fix student path, train-only neighbor map, scheduler compatibility, checkpoint load; create `scripts/train_student_fb15k237.sh` with same timestamp/W&B conventions; run student smoke then full.
5. Compare final structural-only student filtered ranks to paper Table 3. Only after main result, consider ablation experiments.

## Environment / paths
- Server repo `/workspace/workspace/DSoM`
- Python env `/workspace/workspace/DSoM/dsom/bin/python`
- Run Bash from repo root (Bash sets cwd to src because datasets.py uses `../data`).
- Git remote repository https://github.com/keiranoluv/Reproduce_DSoM
- Use exact file edits or full replacement files; do not use Python scripts to patch source.
