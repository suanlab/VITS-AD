# Pre-registration: H10, normalization toggle for Matrix Profile and KMeansAD

Written and committed before any run of the experiment below; the script was only linted. Results
will be reported whatever they are. Changes after this commit, if any, will be listed at the end with
reason and date.

## What was known when this was written
- All results of the paper: the six Sub- detectors gain from global normalization on TSB-AD-U-Eva
  (+0.089 to +0.206) but not uniformly on the 472 series in neither split (H9 refuted: three means
  negative); the source decomposition (exploratory) ties the difference to the sources of the 472.
- Published VUS-PR of MatrixProfile (0.350, rank 10 of 32) and KMeansAD (0.366, rank 7).
- No result of either detector with global normalization had been computed.
- Motivation: a mock review (paper/icde2027/REVIEW_SIMULATION.md, round 7) asked whether the effect
  extends beyond the six Sub- wrappers to TSB-AD's other per-window families.

## Procedure (fixed)
- `scripts/run_family_toggle.py`: TSB-AD 1.5's `run_MatrixProfile` and `run_KMeansAD_U` with
  Optimal_Uni_algo_HP_dict hyperparameters. Per-window variant: raw input, z-normalized distance
  (stumpy.stump normalize=True) / z-normalized windows (KMeansAD normalize=True). Global variant:
  series z-scored with training-part statistics, non-normalized Euclidean distance
  (normalize=False) / windows as they are. KMeans random_state = 0 in both variants.
- Series: TSB-AD-U-Eva (350) and the 472 univariate series in neither split.
- Metric and statistics as for the Sub- detectors: VUS-PR with TSB-AD's get_metrics and sliding
  window; Delta = global - per-window, paired; two-sided Wilcoxon; Holm over the two detectors within
  each series set; 95% bootstrap interval (10,000, seed 0); also a bootstrap over source datasets.
- Reproduction check (reported, not part of the decision): per-window variant vs published values,
  median absolute difference. Matrix Profile is deterministic; KMeansAD is not seeded in TSB-AD.

## Hypothesis and decision rule
**H10.** On TSB-AD-U-Eva, global normalization raises both detectors.
- Supported if mean Delta > 0 and Holm p < 0.05 for both detectors.
- Refuted if mean Delta < 0 for at least one detector.
- Otherwise inconclusive.

## Secondary (reported, no decision rule)
- The same Deltas on the 472 held-out series, overall, per source, and with sources weighted equally.
- Failed runs per detector and variant.

## Changes after registration
- 2026-10-25 (compute cutoff; no H10 statistic computed): series longer than about 150,000 points
  run for many hours, mainly because KMeansAD's reverse windowing in TSB-AD loops over all windows
  for every point. If a series has not finished in both variants by 2026-10-29 00:00 KST, the
  decision rule is applied to the series where both variants finished, as for H9, and the unfinished
  series are listed with the result. Remaining at this time: four series of 195,000 to 900,000
  points and 24 shorter ones. Disclosure: one series (001_NAB_id_1) was inspected once as a smoke
  check right after the runs started; no other value was looked at.
- 2026-10-25 (operational, values unchanged): `--only`/`--part` options were added to
  `run_family_toggle.py` (725ddaf) so that a second process could finish part of a shard; results are
  merged by series and the duplicated series agree exactly.
- 2026-10-25 (disclosure added after the result): before this registration, Matrix Profile's
  per-window variant had been run on a 100-series subset of TSB-AD-U (artifacts/tsb_ad_pool,
  471c062, 2026-10-04); no global-normalization run of either detector existed.
