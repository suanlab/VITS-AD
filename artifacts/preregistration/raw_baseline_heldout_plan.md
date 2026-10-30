# Analysis plan: training-free baselines on the 472 held-out series (exploratory)

Written and pushed before our own run on these series. **Not a confirmatory test**: a simulated
reviewer (internal mock review, 2026-10-30) re-implemented Raw-1NN and reported, for 356 of the 472
series, a global minus per-window difference of about -0.04 (YAHOO about -0.19). We had seen that
estimate before writing this plan, and the evaluation split's YAHOO series already showed -0.212 in
our released files (per_series_knn). The plan fixes the analysis, not a prediction; no decision rule.

## What was known when this was written
- Evaluation split (350): Raw-1NN global minus per-window +0.177 (paper Sec. VI); tuning split (48):
  +0.165, registered as H6's secondary and computed on 2026-10-30 from released files.
- Sub- detectors on the same 472 series: H9 refuted (three negative means).

## Procedure (fixed)
- `run_official_protocol --subset TSB-AD-U --split rest --arms raw_knn,raw_knn_wz,raw_flatten
  --tag knn_rest`: Raw-1NN with global and with per-window normalization, and Raw-LW (global), with
  W, strides, scoring and metric exactly as on the evaluation split. All 472 series; failures kept.
- Reported: Raw-1NN global minus per-window (mean, 95% bootstrap interval, Wilcoxon), per source
  (YAHOO, UCR, WSD, Stock) and with sources weighted equally; Raw-1NN and Raw-LW against the six Sub-
  detectors' two variants on the same series (means only).
- The paper will state the result whatever its sign, next to the evaluation-split contrast in
  Sec. VI, and will qualify Sec. VI's claims accordingly.
