# Pre-registration: H11, the normalization toggle on TSB-UAD sources that TSB-AD does not contain

Written and pushed before any detector, score or metric was computed on these series. Results will
be reported whatever they are. Changes after this commit, if any, will be listed at the end with
reason and date.

## What was known when this was written
- All results of the paper: on TSB-AD-U-Eva the six Sub- detectors gain from global normalization
  (+0.089 to +0.206); H6 supported at its threshold on the 48 tuning series; H9 refuted on the 472
  series in neither split (three negative means, one significant); Raw-1NN's contrast also fails on
  those 472 (-0.021, exploratory).
- Motivation: mock reviews (round 9) asked for a benchmark other than TSB-AD. TSB-UAD (Paparrizos et
  al., PVLDB 2022) is TSB-AD's predecessor, curated by the same group, and most of its sources were
  carried into TSB-AD. We use only the sources TSB-AD does not contain. This is a different curation
  of the same benchmark family, run with TSB-AD's detectors and protocol, not an independent harness.
- Inspected before writing (no detector run): source folders, file counts, series lengths and the
  position of the first anomaly label (to fix the training-part rule). Every primary series has
  anomaly labels within its first 30%; 24 GHL series have all their labels there.

## Data (fixed)
- `scripts/convert_tsbuad.py` on TSB-UAD-Public.zip (SHA-256 ff4aa83a...49a1):
  - **primary set** (sources in no TSB-AD file): SensorScope 23, Occupancy 10, Dodgers 1 = 34 series;
  - **secondary set** (sources absent from TSB-AD-U, present in TSB-AD-M as multivariate series):
    GHL 126, Genesis 6 = 132 series.
  - ECG is excluded (cut from MIT-BIH Long-Term ECG record 14046, i.e. TSB-AD's LTDB).
  - Training part: the first 30% of each series (TSB-UAD has no split), whatever its labels.

## Procedure (fixed)
- `run_detector_normalization --split uad_primary` and `--split uad_secondary`: the six Sub-
  detectors, TSB-AD 1.5 classes, tuned hyperparameters, per-window vs global replacement exactly as
  for H9; VUS-PR from TSB-AD's get_metrics with its sliding-window parameter, over the whole series
  as in TSB-AD.
- Statistic as H9: Delta = global - per-window, paired over series where both run; two-sided
  Wilcoxon, Holm over six detectors; 95% bootstrap interval (10,000, seed 0).

## Hypothesis and decision rule (primary set)
**H11.** If removing per-window z-normalization helps the Sub- detectors beyond TSB-AD's curation,
global normalization beats per-window on the primary set.
- **Supported** if mean Delta > 0 for at least 5 of 6 detectors **and** Holm-significant positive
  for at least 3 of 6 (as H6; 34 series limit power).
- **Refuted** if mean Delta < 0 for 2 or more detectors.
- Otherwise **inconclusive**.

## Secondary (reported, no rule)
- The same on the secondary set; per-source means and source-equal means for both sets.
- Raw-1NN global vs per-window and Raw-LW (`run_official_protocol --split uad_* --arms
  raw_knn,raw_knn_wz,raw_flatten --tag knn_uad_*`).
- Failed runs per detector and variant.

## Changes after registration
(none)
