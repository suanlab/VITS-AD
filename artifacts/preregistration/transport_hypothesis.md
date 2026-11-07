# Pre-registration: TR, does the toggle effect transport to YAHOO series TSB-AD left out?

Written and pushed before any detector, score or metric was computed on the target series. Results
will be reported whatever they are. Changes after this commit, if any, will be listed at the end
with reason and date.

## What was known when this was written
- All results of the paper, including the toggle on all 870 TSB-AD-U series and its split by
  source. On the 472 held-out series YAHOO (224 series) decides most signs.
- Exploratory, computed for this registration from those known results: matched to their TSB-UAD
  files, TSB-AD-U's 259 YAHOO series fall into the Yahoo S5 subsets A1 (real, 28), A2 (synthetic,
  92) and A3/A4 (synthetic benchmark series, 139). The mean toggle gain on them is positive on A1 for
  all six detectors (+0.070 to +0.203) and mostly negative on A2 and A3/A4 (table below).
- Target series: TSB-UAD-Public (SHA-256 ff4aa83a...49a1, as for H11) holds 367 YAHOO files.
  `scripts/match_tsbuad_shared.py` matched every TSB-AD-U YAHOO series to a TSB-UAD file by raw values
  only (no labels, scores or detectors; `artifacts/official_protocol/tsbuad_shared_match.json`); 102
  distinct TSB-UAD YAHOO series match none. `scripts/convert_tsbuad_shared.py` writes them in
  TSB-AD's format with TSB-AD's YAHOO training part (first 500 points; every TSB-AD-U YAHOO file has
  tr_500). Inspected: file names, lengths, whether a series has any label and the position of its
  first label. 99 have labels (A1 34, A2 8, A3/A4 57); the 3 without labels are skipped, as
  TSB-AD's protocol does.
- KDD21 (TSB-AD's UCR) was considered and dropped before any run: TSB-AD cut its UCR series and set
  training parts by a rule we could not reproduce (4 of 228 training parts equal the archive's).

## Procedure (fixed)
- `run_detector_normalization --split uad_shared`: the six Sub- detectors, TSB-AD 1.5 classes,
  TSB-AD's tuned hyperparameters, per-window vs global replacement exactly as for H9 and H11; VUS-PR
  from get_metrics with TSB-AD's sliding-window parameter.
- Delta = global - per-window per series; strata A1, A2, A3A4 (A3 and A4 share series).
- **Stratified prediction** (registered values below): the stratum means of Delta over TSB-AD-U's 259
  YAHOO series, weighted by the target's stratum counts (34, 8, 57). Its 95% interval: 10,000
  bootstrap resamples of series within each TSB-AD stratum (seed 0).
- **Competing predictions:** the unstratified mean over the 259 TSB-AD-U YAHOO series, and the
  evaluation-split mean (no composition at all).
- Observed: the mean Delta over the target series where both variants ran; for the interval of
  observed minus predicted, target series are resampled within strata (10,000, seed 0) jointly with
  the prediction's resamples.
- Direction convention: a Holm-significant result is a gain only if the mean and the signed ranks are
  both positive, a loss only if both are negative.

## Registered predictions (computed by `transport()` in scripts/paper_numbers.py before any target run)

| detector | A1 | A2 | A3A4 (TSB-AD means) | stratified prediction [95%] | unstratified YAHOO | evaluation split |
|---|---|---|---|---|---|---|
| Sub-KNN | +0.096 | -0.005 | -0.108 | -0.030 [-0.077, +0.017] | -0.050 | +0.192 |
| Sub-MCD | +0.135 | -0.015 | -0.014 | +0.037 [+0.010, +0.066] | +0.002 | +0.206 |
| Sub-OCSVM | +0.100 | -0.246 | -0.098 | -0.042 [-0.081, -0.001] | -0.130 | +0.162 |
| Sub-HBOS | +0.186 | -0.052 | +0.002 | +0.061 [+0.017, +0.103] | +0.003 | +0.128 |
| Sub-IForest | +0.203 | -0.040 | -0.037 | +0.045 [+0.001, +0.091] | -0.012 | +0.130 |
| Sub-LOF | +0.070 | +0.038 | -0.091 | -0.025 [-0.062, +0.012] | -0.027 | +0.089 |

## Hypotheses and decision rules
**TR1 (composition below the source level transports).** If the toggle's effect on YAHOO is a
property of its subsets, the stratified prediction matches the target better than the unstratified
YAHOO mean.
- MAE = mean over the six detectors of |observed - prediction|.
- **Supported** if MAE(stratified) <= MAE(unstratified) - 0.02 **and** the interval of observed minus
  stratified prediction contains 0 for at least 5 of 6 detectors.
- **Refuted** if MAE(stratified) >= MAE(unstratified).
- Otherwise **inconclusive**.
- Behaviour: the two predictions differ by 0.002 to 0.088 (mean 0.043). If the unstratified mean
  were the truth, MAE(stratified) would exceed MAE(unstratified) by about that amount, plus noise from
  99 target series (bootstrap half-widths of a detector's mean near 0.02 to 0.04).

**TR2 (real YAHOO series behave like the other sources).** On the 34 labelled A1 target series,
global beats per-window.
- **Supported** if mean Delta > 0 for at least 5 of 6 detectors **and** Holm-significant gains for at
  least 3 of 6 (as H6 and H11; 34 series limit power).
- **Refuted** if Holm-significant losses for 2 or more detectors.
- Otherwise **inconclusive**.

## Secondary (reported, no rule)
- Observed means by stratum; MAE of the evaluation-split prediction.
- The same under AUC-PR.
- Failed runs per detector and variant.

## Compute cutoff
Series not finished by 2026-11-15 00:00 KST are left out and listed.

## Changes after registration
(none)
