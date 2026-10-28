# Pre-registration: T1-T2, calibrated thresholds under drift for TSB-AD's Sub- detectors

Written and committed before the experiment below was run on any benchmark series; the script was
only linted and run once on one synthetic series to check that it executes. Results will be reported
whatever they are. Changes after this commit, if any, will be listed at the end with reason and date.

## What was known when this was written
- The Raw-1NN threshold study (paper Sec. VIII, run_drift_threshold2.py, H5): at a threshold calibrated
  on the training part, the global variant's false-alarm rate (FAR) rises to 0.62-0.90 under injected
  benign drift while the per-window variant stays at 0.11-0.14; without drift 0.23 vs 0.11.
- No threshold or FAR result for any TSB-AD Sub- detector had been computed.
- Motivation: a mock review (paper/icde2027/REVIEW_SIMULATION.md, round 8) noted that the
  recommendation "evaluate at a threshold" rested on the authors' own baseline only.

## Procedure (fixed)
- `scripts/run_threshold_sub.py`: Sub-KNN, Sub-IForest, Sub-HBOS through TSB-AD 1.5's wrappers with
  the tuned hyperparameters used throughout the paper, per-window vs global normalization as in
  run_detector_normalization.py. These detectors fit and score the whole series.
- Series: TSB-AD-U-Eva (350). Drifts injected into the test part: none, level +3 training standard
  deviations, linear trend to +3 (run_drift_injection.inject).
- Threshold: 99th percentile of point scores on the second half of the training part (nominal FAR
  1%). FAR = fraction of normal test points above it; event recall = fraction of anomalous test
  segments with an alarm. Diagnostic: equal-FAR operating points (1%, 5%) set with test labels.
- Means over series where the run succeeded; paired Wilcoxon and 95% bootstrap intervals (10,000,
  seed 0) for differences.

## Hypotheses and decision rules
**T1 (drift inflates global FAR, not per-window FAR).** For each detector, the increase in mean FAR from
no drift to the level +3 drift.
- Supported if, for at least two of the three detectors, the global variant's increase is >= 0.20 and
  the per-window variant's increase is <= 0.05.
- Refuted if, for at least two detectors, the global variant's increase is < 0.10.
- Otherwise inconclusive.

**T2 (ranking and threshold disagree).** Without drift, does the variant with the higher VUS-PR on the
evaluation split (global, for all three; paper Table III) also have the higher event recall at equal
FAR 1%?
- Supported (they disagree) if per-window has higher mean event recall at equal FAR 1% for at least two
  detectors.
- Refuted if global has higher mean event recall at equal FAR 1% for all three.
- Otherwise inconclusive.

## Secondary (reported, no rule)
- The same quantities under the trend drift; calibrated event recall; FAR without drift.
- Failed runs per detector, variant and drift.

## Changes after registration
(none)
