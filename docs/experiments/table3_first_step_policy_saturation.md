# DSoM Table 3 — First-Step RC Policy Saturation Analysis

## 1. Objective

Investigate the reinforced curriculum (RC) policy collapse observed
during Student reproduction of DSoM Table 3 on FB15K-237.

Specifically, determine whether the first optimizer update causes
immediate policy saturation and whether input normalization or
a separate policy optimizer prevents this behavior.

## 2. Experimental Setup

Dataset: FB15K-237

Teacher:
Pretrained multimodal ComplExMDR checkpoint.

Student:
Unimodal ComplExMDR, rank 2000.

RC policy:
- MLP with 1024 hidden units
- Seven multimodal teacher strategies
- Binary reward
- Reward = +1
- Penalty = -10
- Policy temperature = 1

Common Student configuration:
- Adagrad, learning rate 0.1
- Batch size 1024
- N3 regularization, weight 0.01
- KD temperature 4
- KD weight 2
- NDKD alpha = 1, beta = 1

Compared experiments:

| Run | Policy optimizer | Policy LR | Input normalization |
|---|---|---:|---|
| R1 | Shared Adagrad | 0.1 | No |
| R2 | Separate Adam | 0.00001 | No |
| R4 | Shared Adagrad | 0.1 | Yes |

R1 and R2 use identical initial policy configurations and
produce identical first-batch pre-update diagnostic values.

R2 changes both the policy optimizer algorithm and its learning rate.
Therefore, their individual contributions cannot be isolated
from this comparison alone.

## 3. Diagnostic Method

The implementation records policy statistics on the same training
minibatch immediately before and after the optimizer update.

For each probe:
1. Recompute teacher predictions.
2. Evaluate policy probabilities with policy dropout disabled.
3. Record entropy, hidden RMS, and output logit range.
4. Compare values before and after optimizer.step().

The probe uses the same batch and does not introduce a separate
policy optimization step.

Entropy is measured over seven strategies.

Maximum possible entropy:

    H_max = ln(7) = approximately 1.94591

A near-zero entropy indicates an almost deterministic policy.

Relevant source files:
- src/kd.py
- src/kd_optimizers.py
- src/kd_learn.py

## 4. First-Step Results

### 4.1 Policy entropy

| Run | Before update | After update |
|---|---:|---:|
| R1 | 1.832511 | 0.004147 |
| R2 | 1.832511 | 1.737293 |
| R4 | 1.919530 | 0.002966 |

R1 and R4 exhibit near-complete policy saturation after
the first optimizer update.

R2 preserves substantially higher policy entropy.

### 4.2 Hidden activation RMS

| Run | Before update | After update |
|---|---:|---:|
| R1 | 0.857350 | 2032.694092 |
| R2 | 0.857350 | 0.895919 |
| R4 | 0.405769 | 997.501831 |

The hidden activation RMS increases dramatically in R1 and R4.

In R2, hidden activation magnitude remains close to its
pre-update value.

### 4.3 Policy output logit range

| Run | Before update | After update |
|---|---:|---:|
| R1 | 1.446046 | 246115.843750 |
| R2 | 1.446046 | 3.250485 |
| R4 | 0.689721 | 120498.843750 |

The very large post-update logit ranges in R1 and R4
cause the softmax policy distribution to become highly
concentrated.

### 4.4 Policy gradient and parameter update

| Run | Gradient norm | Maximum FC2 weight change | Nonzero advantages |
|---|---:|---:|---:|
| R1 | 1.977970 | 0.10000002 | 11/1024 |
| R2 | 1.977970 | 0.00001000 | 11/1024 |
| R4 | 1.251625 | 0.10000002 | 14/1024 |

R1 and R2 have identical first-batch policy gradient norms
and nonzero-advantage counts.

However, their optimizer updates differ substantially.

The maximum FC2 weight change is measured directly.
The full FC1 parameter-update magnitude was not measured
in these logs.

## 5. Input Normalization Verification

R4 uses:

    policy_input_norm=True

The first recorded MLP input statistics are:

    teacher_mean   = -0.0000
    teacher_std    = 1.0000
    teacher_absmax = 13.5943

Therefore, input normalization is active and produces
the expected mean and standard deviation.

Nevertheless, after the first optimizer step:

    hidden_rms  = 997.5018
    logit_range = 120498.8438
    entropy     = 0.002966

Conclusion:

Input normalization alone does not prevent first-step
policy saturation under shared Adagrad with LR 0.1.

## 6. Mechanistic Interpretation

At the first optimization step, Adagrad approximately applies:

    delta_w = -lr * gradient / (abs(gradient) + epsilon)

For parameters whose gradients dominate epsilon, the first
update magnitude can approach the learning rate.

With policy LR = 0.1, the observed maximum FC2 parameter
change is approximately 0.1.

The RC policy MLP receives a high-dimensional input formed
by concatenating prediction scores from three modalities
over all candidate entities.

For FB15K-237, this corresponds to approximately:

    14951 entities * 3 modalities = 44853 input features

The first fully connected layer maps this representation
to 1024 hidden units.

A large optimizer update in such a high-dimensional network
can strongly amplify hidden activations and output logits.

This mechanism is consistent with the observed first-step
saturation in R1 and R4.

The before/after probes establish the optimizer update as
the immediate cause of the observed change in policy outputs.

However, the evidence does not isolate the contribution
of every policy layer or distinguish optimizer algorithm
from learning-rate effects.

## 7. Immediate Saturation vs Long-Term Collapse

The experiments support distinguishing two phenomena.

### 7.1 Immediate saturation

R1 and R4 exhibit near-zero entropy immediately after the
first optimizer update.

R2 avoids this immediate effect.

### 7.2 Long-term collapse

Earlier R1-R7 policy audits reported that the policy
eventually concentrated heavily on Structural-only.

In particular:
- R1 and R4: approximately 100% Structural strategy
- R2: approximately 99.41% Structural strategy

These figures describe the previously audited final
training-log window, not every prediction over the
full dataset.

Therefore, avoiding immediate saturation is not sufficient
to guarantee that the policy remains diverse throughout
training.

The mechanism of long-term policy collapse remains unresolved.

## 8. Student Evaluation Results

Full filtered test evaluation:

| Metric | R1 | R2 | R4 |
|---|---:|---:|---:|
| MR | 162.53 | 159.58 | 160.15 |
| MRR | 0.35958 | 0.35708 | 0.35718 |
| Hits@1 (%) | 26.57 | 26.24 | 26.26 |
| Hits@3 (%) | 39.45 | 39.35 | 39.31 |
| Hits@10 (%) | 54.85 | 54.59 | 54.70 |

DSoM Table 3 Student reference:

    MR       = 116
    Hits@1   = 27.5%
    Hits@3   = 41.1%
    Hits@10  = 56.4%

R2 avoids the severe first-step saturation observed
in R1, but does not outperform R1 in MRR, Hits@1,
or Hits@10 in the recorded runs.

Hence, eliminating immediate policy saturation is
not yet demonstrated to close the Table 3 performance gap.

## 9. Evidence Locations

R1 log:

    runs/student/FB15K-237/table3_batch1024/full/train.log

R2 log:

    runs/student/FB15K-237/table3_separate_adam/full/train.log

R4 log:

    runs/student/FB15K-237/table3_r4_input_norm/full/train.log

Key diagnostic markers:

    ===== FIXED-BATCH POLICY PROBE =====
    RC batch=0
    MLP batch=0

All results in this document are based on existing
experiment logs; no new training was performed.

## 10. Findings and Confidence

Confirmed:

1. R1 and R4 show near-deterministic policy output immediately
   after their first optimizer update.

2. R4 input normalization is active and functioning.

3. R1 and R2 begin with identical policy diagnostic values
   and first-batch gradient norms.

4. A separate Adam policy optimizer at LR 1e-5 prevents
   severe first-step saturation in R2.

5. The recorded maximum FC2 update is approximately 0.1
   for R1/R4 and 1e-5 for R2.

Strongly supported:

6. Shared Adagrad with LR 0.1 makes the first policy update
   excessively large for the tested RC architecture.

Not established:

7. Whether Adagrad itself or the learning-rate magnitude
   is primarily responsible.

8. Whether the same mechanism explains all later
   policy-collapse behavior.

9. Whether stabilizing RC is sufficient to reproduce
   the published Table 3 Student metrics.

## 11. Implications for Reproduction

The first-step saturation observed in R1/R4 is a concrete
optimization instability rather than an input-normalization
failure.

Nevertheless, changing optimizer configuration constitutes
an experimental variation unless the original paper's
training protocol is independently verified.

Recommendations:

1. Preserve R1 as the current source-oriented baseline.
2. Keep R2 and R4 as diagnostic ablations.
3. Avoid further blind optimizer hyperparameter searches.
4. Document immediate and long-term collapse separately.
5. Continue paper/source alignment analysis, especially
   multimodal feature extraction and Teacher configuration.

## 12. Conclusion

R1 and R4 demonstrate immediate RC policy saturation
after a single shared-Adagrad optimization step.

R2 demonstrates that a separate Adam policy optimizer
with a much smaller learning rate avoids this immediate
saturation but does not prevent eventual policy concentration.

The findings provide a reproducible explanation for a major
observed RC training instability.

They do not, by themselves, establish the cause of the
remaining DSoM Table 3 reproduction gap.
