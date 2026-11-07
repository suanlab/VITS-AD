# Pre-registration: RT, re-tuning both normalization variants of TSB-AD's Sub- detectors

Written and pushed before the registered grid search or any re-tuned run. Results will be
reported whatever they are. Changes after this commit, if any, will be listed at the end with
reason and date.

## What was known when this was written
- All results of the paper. With TSB-AD's tuned hyperparameters, which TSB-AD selected under
  per-window normalization, for both variants:
  - evaluation split (350): global beats per-window for all six Sub- detectors (+0.089 to +0.206);
  - tuning split (48, H6): supported at its threshold;
  - 472 series in neither split (H9): refuted (means Sub-KNN +0.059, Sub-MCD +0.088, Sub-OCSVM
    -0.037, Sub-HBOS -0.002, Sub-IForest -0.010, Sub-LOF +0.051; Holm-significant: Sub-MCD gain,
    Sub-OCSVM loss).
- Motivation: all five simulated reviews of round 17 noted that the global variant runs with
  configurations tuned for the per-window variant, so the direction of this bias is unknown. Our
  internal advice note (ADVISE.md, 2026-11-07) predicted that re-tuning changes the
  evaluation-split gains by less than 0.03 and at most one sign on the 472.
- Inspected before writing:
  - TSB-AD's search grid (`Uni_algo_HP_dict` in TSB-AD 1.5) for the six detectors. It is
    identical at commits 35ba9c3b, 6beac72e and 303dffbf: periodicity {1, 2, 3} crossed with
    n_neighbors {10..50} (Sub-KNN, Sub-LOF), n_estimators {25..200} (Sub-IForest), n_bins
    {5..40} (Sub-HBOS), kernel {linear, poly, rbf, sigmoid} (Sub-OCSVM) and support_fraction
    {0.2..0.8, None} (Sub-MCD): 87 configurations.
  - A timing run of the full grid, both variants, on one tuning series
    (003_NAB_id_3_WebService_tr_1362_1st_1462.csv; 174 runs, 13.5 min). Only the number of entries
    (174) and of failures (0) were printed; no value was looked at. Its output (SHA-256
    abd37eec592919c4e75bf4957130f1388d61e81b0698860f67f5ee24649f80d1) is not used: the series is
    run again in the registered grid.

## Procedure (fixed)
- `scripts/run_retune.py` (this commit), TSB-AD 1.5 classes with the same per-window / global
  replacement as the toggle, univariate, detectors Sub_KNN, Sub_MCD, Sub_OCSVM, Sub_HBOS,
  Sub_IForest, Sub_LOF.
- **Grid** on the 48 series of TSB-AD-U-Tuning: every configuration, both variants; VUS-PR from
  TSB-AD's get_metrics with its sliding-window parameter.
- **Selection**, per detector and variant separately: the configuration with the highest mean
  VUS-PR over the tuning series; a failed run scores 0 (as in TSB-AD's HP_Tuning_U.py); ties go to
  the first configuration in grid order. If the per-window selection does not reproduce TSB-AD's
  tuned configuration, that is reported as a finding; the rule does not change.
- **Apply**: the kept configurations on the evaluation split (350) and on the 472 series in
  neither split. Where a kept configuration equals TSB-AD's tuned one, the existing toggle result
  is its value (repeated runs were identical, `repeat_determinism`); the others are run.
- **Statistic**, per detector: Delta_RT = global (re-tuned) - per-window (re-tuned), paired over
  series where both ran; two-sided Wilcoxon signed-rank; Holm over the six detectors; 95% bootstrap
  interval of the mean (10,000 resamples, seed 0).
- **Direction convention (fixed here, unlike earlier registrations):** a Holm-significant result
  counts as a gain only if both the mean Delta and the signed ranks (W+ > W-) are positive, and as
  a loss only if both are negative; a significant result whose mean and signed ranks disagree
  counts as neither.

## Hypotheses and decision rules
**RT1 (evaluation split).** If the toggle's gain is not an artefact of per-window tuning, then with
each variant tuned on the tuning split, global beats per-window on the evaluation split.
- **Supported** if mean Delta_RT > 0 for at least 5 of 6 detectors **and** Holm-significant gains
  for at least 4 of 6.
- **Refuted** if Holm-significant losses for 2 or more detectors.
- Otherwise **inconclusive**.
- Null behaviour: with no difference between the variants, Holm keeps the probability of any
  significant result at or below 0.05, so P(refuted) <= 0.05 and P(supported) <= 0.05.

**RT2 (472 held-out series).** If H9's outcome is not an artefact of per-window tuning, re-tuning
leaves the held-out comparison where it was.
- Per detector, the change c = Delta_RT(472) - Delta_H9(472) (H9 values above).
- **Unchanged** if |c| <= 0.03 for at least 5 of 6 detectors **and** Sub-MCD remains a
  Holm-significant gain and Sub-OCSVM a Holm-significant loss.
- **Changed** if |c| > 0.03 for 3 or more detectors, **or** Sub-MCD or Sub-OCSVM becomes
  Holm-significant in the opposite direction.
- Otherwise **partly changed**.
- The margin 0.03 is above every 95% interval half-width of Delta_H9 (0.015 to 0.026). The
  comparison is on the same series, so c has no series-sampling noise; it is the effect of the
  configuration change itself. With no tuning effect (the same configurations kept), c = 0 and
  the rule returns "unchanged".

**RT3 (ranking).** Sub-KNN with its re-tuned global configuration ranks first among the 32
univariate detectors, built as Table 3 (published values for the other 26; the six Sub- detectors
at their re-tuned global values; a failed run keeps its published value). Supported if rank 1,
refuted otherwise.

## Secondary (reported, no rule)
- Whether the per-window selection reproduces TSB-AD's tuned configuration, per detector.
- Each variant's mean on the evaluation split and the 472, re-tuned against TSB-AD's tuning.
- Re-tuned global against TSB-AD's published per-window configuration (the Table 3 comparison).
- The 64-setting rank range with re-tuned values.
- Failed runs per detector, variant and configuration.

## Compute cutoff (fixed now)
- Grid: if some series has not finished by 2026-11-15 00:00 KST, the selection for that detector
  and variant uses the tuning series where every one of its configurations finished, and the
  unfinished series are listed with the result.
- Apply: series not finished by 2026-11-19 00:00 KST are left out of the paired comparison and
  listed; the decision is taken on the rest.
- Run order and the split of work across processes (detector and shard, `--part`) change no
  computation.

## Changes after registration
(none)
