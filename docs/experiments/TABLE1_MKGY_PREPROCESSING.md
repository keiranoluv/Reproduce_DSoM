# DSoM Table 1 — MKG-Y Dataset Preprocessing

## Objective

Reproduce DSoM Table 1 results on the MKG-Y dataset.

## Data provenance

Dataset: MKG-Y

Source: MMRNS
https://github.com/quqxui/MMRNS

Raw dataset files:
- train.txt
- valid.txt
- test.txt

Expected statistics:
- Entities: 15,000
- Relations: 28
- Train: 21,310
- Validation: 2,665
- Test: 2,663

Integrity audit:
- No duplicate triples within splits
- No train/validation overlap
- No train/test overlap
- No validation/test overlap

## Step 1: Prepare raw triples

From the repository root:

```bash
mkdir -p src_data/MKG-Y

cp /workspace/workspace/dataset_audit/MMRNS/MKG-Y/train.txt \
   src_data/MKG-Y/train

cp /workspace/workspace/dataset_audit/MMRNS/MKG-Y/valid.txt \
   src_data/MKG-Y/valid

cp /workspace/workspace/dataset_audit/MMRNS/MKG-Y/test.txt \
   src_data/MKG-Y/test
```

The source path above refers to the local staging directory.
Adjust it if the MMRNS dataset is stored elsewhere.

Do not replace existing dataset files without auditing them.

## Step 2: Preprocess

```bash
cd /workspace/workspace/DSoM
./dsom/bin/python -m scripts.prepare_mkgy_dataset
```

This calls the original DSoM
`src.process_datasets.prepare_dataset()` function.

The script refuses to overwrite an existing output directory.

## Expected output

```text
data/MKG-Y/
├── ent_id
├── rel_id
├── train.pickle
├── valid.pickle
├── test.pickle
├── to_skip.pickle
└── probas.pickle
```

Entity and relation IDs are assigned by sorting the
vocabulary across train, validation, and test.

## Notes

- Preserve the original MMRNS split contents.
- Do not deduplicate or resplit the data.
- Keep structural preprocessing separate from multimodal
  feature extraction.
- Teacher and Student training are not part of this step.
- Validate generated artifacts before feature extraction.

## Reproduction status

Structural preprocessing: completed locally.
Artifact validation: pending confirmation.
