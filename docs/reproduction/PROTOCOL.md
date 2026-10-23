# Supported reproduction protocol

## Supported scope

The documented training model is `ComplExMDR`. Its teacher separates structural,
visual, and description relations; its student is structure-only. The portable
profiles, Table 2/4/6 launchers, and WN9 workflow construct validated combinations
of that model. This is not a promise that every option in the inherited CLI or
every legacy model class works.

In particular, legacy RSME construction references undefined global options,
and legacy `soft_kl`/assistant distillers are incomplete. They are not used in
these workflows. TMR, DySarl and other competitors are not trained by the DSoM
launchers. Historical experiment scripts remain research records, not alternate
unqualified installation entrypoints.

## Matched configuration

The fresh baseline profiles retain rank 2000 (ComplEx's actual vectors have
`2 * rank` components), batch 1000, Adagrad .1, N3 .01, init .001, scale 20,
max 200 epochs, validation every 3, early stopping 10 and seed 2022. Teacher
selection uses validation Hits@10; student uses validation MRR. The existing
`min` scheduler monitors validation Hits@10 in the wrong direction for a maximum
metric; it is retained to match source/baseline runs, not presented as a corrected
best practice. Changes require a new matched experiment.

Student CE/KD weights are 1/2, temperature 4, NDKD alpha/beta 1/1. Training
neighbors use training triples and their reciprocals, not evaluation filters.
Teacher parameters are frozen during student training. Evaluation filters use
all original splits as in the benchmark.

## Source-frozen versus learned RC

`source_frozen` excludes the RC policy from optimizer parameter groups while
still constructing and sampling it. It is the default main baseline protocol
in this reproduction, including default Table 7. It does not establish the
effectiveness of learned reinforcement learning.

The DB15K Table 4 learned variants use a separate policy Adam at 1e-5 and binary
+1/-10 rewards. Optional Table 7 `--policy-optimizer Adam` is a separately labelled
adaptation. Do not merge it with a source-frozen result under one unnamed row.
Fixed S/V/D/average teacher selection does not train a policy. Teacher oracle
minimum ranks cannot substitute for student results or direct teacher ensembles.

## Method and data differences

- Released NDKD groups **sums** of neighbor probabilities; paper Eq.12 describes
  their **mean**. All historical and supported default profiles preserve SUM.
  NEKD is grouped neighbor/non-neighbor mass KL, not KL within individual neighbors.
- BERT/ViT features are reconstructed. Image input is 384 and output 1000 classifier
  logits in the implemented pipeline, whereas the paper describes ViT224. Raw-image
  coverage, historic URL availability and missing-vector RNG order can differ.
- DB15K public validation overlaps train/test and contains duplicates. Main tables
  preserve those files. CleanV1 is an isolated sensitivity dataset with different
  training size/filters/validation distribution, not an unconfounded causal estimate.
- WN9 source has many inverse facts across splits. It also contains a preserved
  source self-loop absent from the current WN18 graph union. Generic encoder features
  can be reused by synset ID; WN18 graph checkpoints cannot be reused for WN9.
- MulDE in Table 2 is an adaptation: the authors' DSoM adaptation settings are not
  released. Its details and optimizer ownership are documented in the Table 2 record.
- Table 4 Teacher Avg. is ambiguous. Both mean metrics and mean logits are saved.
  ML-on-val follows MoSE architecture but fixes the final epoch instead of selecting
  on test. Standalone RC-on-val uses explicitly documented reward/training choices.
- Table 7 compares to methods reporting raw metrics, while the DSoM evaluator is
  filtered. Both are reported on one filtered-validation-selected checkpoint;
  exact author Table 7 settings remain unverified. The completed source-frozen
  full run achieved filtered MRR .919634 versus raw .618763; filtered closeness
  to paper .9213 does not establish protocol equivalence. See the
  [recorded result](RESULTS.md#recorded-table-7-full-result).

## Reporting

Report source commit, dataset/feature provenance, seed, policy optimizer,
selected epoch and raw/filtered setting. The published table grid being complete
under a documented profile is different from exact numerical or author-protocol
reproduction. No single-seed gain is treated as statistically established.
Keep smoke, full and sensitivity results separate. Read student `final_result`,
not the last test log entry. Existing experiment/audit documents retain detailed
evidence rather than silently revising old benchmark results.
