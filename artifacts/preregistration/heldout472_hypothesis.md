# Pre-registration: H9, confirmatory replication on 472 unused TSB-AD-U series

Written and committed before any analysis on these series. Results will be reported whatever they
are. Changes after this commit, if any, will be listed at the end with reason and date.

## What was known when this was written
- All results on TSB-AD-U-Eva (350) and TSB-AD-U-Tuning (48), including H6 (supported at its threshold).
- The list of TSB-AD-U series that are in neither split: 472 series, from four sources only
  (YAHOO 224, UCR 152, WSD 86, Stock 10). Only file names and source labels were inspected; no
  detector, score or label statistic on these series has been computed.
- On the evaluation split, per-window views did comparatively well on YAHOO and UCR. This replication
  set is therefore not a random sample of TSB-AD and is a harder test of the claim.

## Hypothesis
If removing per-window z-normalization helps TSB-AD's Sub- detectors in general, then on these 472
series global normalization will again beat per-window normalization for the six per-window Sub-
detectors.

## Procedure (fixed)
- `scripts/run_detector_normalization.py`, unchanged except for reading this series list
  (`--split rest`), univariate, detectors Sub_KNN, Sub_MCD, Sub_OCSVM, Sub_HBOS, Sub_IForest, Sub_LOF,
  tuned hyperparameters of TSB-AD's HP_list at commit 6beac72e (as in the paper), TSB-AD 1.5 classes
  with the same per-window / global replacement as in Section IV.
- Metric: VUS-PR from TSB-AD's get_metrics with TSB-AD's sliding-window parameter.
- Statistic: per detector, Delta = global - per-window, paired over series where both variants
  run; two-sided Wilcoxon; Holm over the six detectors; 95% bootstrap interval (10,000, seed 0).

## Decision rule
- **Supported** if mean Delta > 0 for at least 5 of 6 detectors **and** Holm-significant positive
  (p < 0.05) for at least 4 of 6.
- **Refuted** if mean Delta < 0 for 2 or more detectors.
- Otherwise **inconclusive**.

## Secondary (reported, not part of the decision)
- Per-source means of Delta (YAHOO, UCR, WSD, Stock).
- Sub-OCSVM with its fit-time per-window MinMax scaler removed in both variants (see Section IV
  caveat raised in review), reported alongside the registered Sub-OCSVM result.

## Changes after registration
(none)
