# Pre-registration: P, the normalization toggle on a new collection (TimeEval's GutenTAG)

Written and pushed before any detector was run on these series. Results will be reported whatever
they are. Changes after this commit, if any, will be listed at the end with reason and date.

## Why
All registered tests of the paper so far are on TSB-AD and its predecessor TSB-UAD, and the one
registered prediction tested on new sources (M) had a single informative target. Simulated reviews
(round 18, 2026-11-16) asked for a prospective test outside the TSB-AD family. We searched for a
real univariate collection with labels that TSB-AD and TSB-UAD do not contain and found none whose
labels and licence suit (the industrial set of TimeSeriesBench is NEK, already in TSB-AD; SKAB is
multivariate with system-level labels). P therefore uses a synthetic collection designed by another
group: the 168 univariate GutenTAG datasets that TimeEval publishes. It is new to all of our analyses
as data, but it is synthetic, so P tests the transport of covariate-based predictions and of the
mechanism (Sec. 6 of the paper), not real-world generality.

## What was known when this was written
- Every result of the paper, including the exploratory analyses after review round 18 (the toggle by
  anomaly type and by anomaly length over the detector window, `paper_numbers.anomaly_type`,
  `window_estimation`).
- TimeEval's published results table (`data/timeeval_results/`) includes other algorithms' ROC-AUC on
  these GutenTAG datasets (TimeEval's own implementations and preprocessing); it entered M's
  TimeEval target, and M's exploratory per-collection scores for GutenTAG were displayed on
  2026-11-17. No result of TSB-AD's Sub- detectors on GutenTAG, under either normalization, has been
  computed or displayed.
- The GutenTAG archive (`gutentag.zip`, SHA-256
  3f3c65772adc20b06eb3abb78a8346757bc3031cba2186188dd4e51197c94d51, downloaded 2026-11-21T17:22Z from
  https://my.hidrive.com/api/sharelink/download?id=j2imAkqU; MIT licence for TimeEval's preprocessed
  datasets) was inspected for structure, its index and its generation config (anomaly kinds and
  lengths, base oscillations). The converted series were read only to compute the covariates below
  (labels, TSB-AD's window estimate).
- `run_detector_normalization.run_series` with the new test-part metrics was smoke-tested on one
  TSB-AD tuning series (003_NAB...; keys only printed).

## Data and procedure (fixed)
- `scripts/convert_gutentag.py`: the 168 univariate semi-supervised datasets; each series is the
  anomaly-free training series (10,000 points, labels 0) followed by the test series (10,000
  points); training part = the first 10,000 points. Families (base oscillation): sinus 42, ecg 39,
  cbf 33, rw 28, poly 26. Kind classes from the generation config: level = {mean, platform, trend},
  scale = {variance, amplitude}, shape = {pattern, pattern-shift, frequency}, point = {extremum}.
- `scripts/run_detector_normalization.py --split gutentag`: TSB-AD's six univariate Sub- detectors
  with TSB-AD's tuned hyperparameters, per-window and global, exactly as for H9 and H11; VUS-PR on the
  whole series (primary) and on the test part only (secondary, `|test`), since the junction of the
  two generated series may itself look anomalous to both variants.
- Gain = global minus per-window VUS-PR, paired per series; 95% bootstrap interval (10,000, seed 0);
  Wilcoxon, Holm over the six detectors. A detector's failed runs are excluded for that detector.
- Predictions: `scripts/gutentag_prediction.py` -> `gutentag_predictions.json` (this commit), from
  released TSB-AD results only. `--check` recomputes it.

## Hypotheses and decision rules
**P1 (covariate transport).** For each detector, the prediction of its mean gain on GutenTAG is the
GutenTAG-weighted mean of TSB-AD's gains by stratum (anomaly type: point/seq/both, x r-bin: <=0.25,
0.25-1, 1-4, >4, with r = median anomaly length / the detector's window from TSB-AD's estimator at
its periodicity rank; strata with fewer than 5 TSB-AD series fall back to the type mean; TSB-AD data:
the evaluation split and the 472 series in neither split, 822 series). Baselines: the pooled mean of
the same 822 series, and the evaluation split's mean (350).
Predicted values (`P1.by_detector`): Sub-KNN 0.145 (pooled 0.116), Sub-MCD 0.164 (0.138), Sub-OCSVM
0.080 (0.048), Sub-HBOS 0.090 (0.053), Sub-IForest 0.092 (0.050), Sub-LOF 0.076 (0.067).
- **Supported** if MAE(stratified) <= MAE(pooled) - 0.01 **and** MAE(stratified) <= MAE(evaluation
  split) - 0.01 over the six detectors.
- **Refuted** if MAE(stratified) >= MAE(pooled). Otherwise **inconclusive**.
- The two predictions differ by 0.009 to 0.042, so the margin 0.01 is about a third of the largest
  possible improvement on average.

**P2 (mechanism by anomaly kind).** The paper's mechanism (per-window normalization hides the interior
of level and scale anomalies; shape anomalies are seen by both) predicts that on the GutenTAG series
whose anomalies are all of one class, the gain is larger for level and scale anomalies than for shape
anomalies. Groups: level 26, scale 65, shape 27 series (point 12, secondary).
- **Supported** if, for at least 5 of 6 detectors, the mean gain on the level-or-scale series is
  positive **and** larger than the mean gain on the shape series.
- **Refuted** if that holds for at most 2 of 6. Otherwise **inconclusive**.

**P3 (replication prediction on new sources, M's rule).** Conclusions: the 66 pairwise orders of the
twelve variants (six detectors x two normalizations, TSB-AD's tuned hyperparameters), discovered on
TSB-AD's evaluation split (350 series, 23 sources; random effects over the 19 with at least two
series). Held out: GutenTAG's five families as new sources (O1, 330 instances); the pooled collection
(O2, secondary). Predictions fixed in `P3` of the predictions file.
- On O1, p_new is **better** than p_iid if Brier(p_new) <= Brier(p_iid) - 0.02 **and**
  AUC(p_new) >= AUC(p_iid) + 0.05 -> **supported**; **no gain** if Brier(p_new) >= Brier(p_iid) - 0.005
  **and** AUC(p_new) < AUC(p_iid) + 0.02 -> **refuted**; otherwise **inconclusive**.

## Secondary (reported, no rule)
- Every gain with its interval and Holm p; by family and by kind class (including point); on the test
  part only; P1 per detector; P3's O2, p_sig, calibration slopes and a constant forecast at the held
  share (oracle reference).

## Cutoff
- Runs that have not finished by 2026-11-29 00:00 KST are reported as missing; the analysis uses the
  completed series.

## Changes after registration
(none)
