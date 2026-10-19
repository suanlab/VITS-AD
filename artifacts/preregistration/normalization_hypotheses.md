# Pre-registration: window normalization on TSB-AD (H1, H2, H4, H5, H6, H8)

Written and committed **before** any of the analyses below was run. The git commit that adds this
file is the timestamp. Results will be reported whatever they are, including refutations. Changes
after this commit, if any, will be listed at the end with their reason and date.

Already known when this was written (exploratory results on TSB-AD-U-Eva / M-Eva, see
`artifacts/official_protocol/paper_numbers.json`): per-window vs. global normalization of TSB-AD's
six per-window Sub- detectors on the 350 univariate evaluation series; Raw-1NN / Raw-LW results on
both evaluation splits; amplitude-ratio (z_r) terciles. **Not yet seen by anyone:** any result on
TSB-AD's tuning split for these detectors; probe scores of H1; the augmented normalization of H2;
the source-balanced ranking of H4; false-alarm rates of H5; the z_r analysis of published
foundation-model scores in H8.

Common definitions: VUS-PR from TSB-AD's `get_metrics` with TSB-AD's sliding-window parameter;
TSB-AD's tuned hyperparameters at commit 6beac72e; "global" = per-channel z-score with
training-part statistics; "per-window" = each window z-scored by its own mean and std (ddof=1,
constant windows -> 0). Paired tests are two-sided Wilcoxon signed-rank; intervals are 95%
bootstrap intervals of the mean (10,000 resamples, seed 0). alpha = 0.05.

## H6 (confirmatory replication on held-out data)
**If** the normalization effect found on the evaluation split is real, **then** on TSB-AD's tuning
split (TSB-AD-U-Tuning, 48 series, disjoint from the evaluation split) global normalization will
again beat per-window normalization for TSB-AD's six per-window Sub- detectors.
- Procedure: `run_detector_normalization.py` unchanged except for reading the tuning list; detectors
  Sub-KNN, Sub-MCD, Sub-OCSVM, Sub-HBOS, Sub-IForest, Sub-LOF with the tuned hyperparameters.
- Statistic: per detector, Delta = global - per-window (VUS-PR), paired over series; Holm over the
  six detectors.
- **Supported** if mean Delta > 0 for at least 5 of 6 detectors **and** Holm-significant (positive)
  for at least 3 of 6. **Refuted** if mean Delta < 0 for 2 or more detectors. Otherwise
  inconclusive (n=48 limits power).
- Secondary (exploratory, reported but not part of the decision): the same on TSB-AD-M-Tuning (20
  series, multivariate detectors), and Raw-1NN global vs per-window on both tuning splits.

## H1 (information decomposition predicts the per-series cost)
A window is w = mu*1 + sigma*u. Per-window normalization keeps only u. **If** the cost of
per-window normalization comes from discarding (mu, sigma), **then** across series it is predicted
by how much better (mu, sigma) alone detects the anomalies than u alone.
- Probes per series (W = 100, same windows, strides and centre mapping as run_official_protocol):
  s_amp = 1-NN distance in the 2-D space (m, l), m = (mu_w - mu_0)/sigma_0, l = log(sigma_w/sigma_0),
  per channel and concatenated for multivariate series; s_shape = 1-NN distance on per-window
  z-normalized windows (the existing Raw-1NN per-window arm).
- Predictor P = VUS-PR(s_amp) - VUS-PR(s_shape).
- Targets: Delta_series = global - per-window for (a) Raw-1NN, (b) Sub-KNN, (c) the factorial's
  Ledoit-Wolf W=100 cells; univariate evaluation split.
- **Supported** if Spearman(P, Delta) >= 0.40 for at least 2 of the 3 targets **and** exceeds
  |Spearman(z_r, Delta)| for at least 2 of the 3. **Refuted** if Spearman(P, Delta) < 0.20 for
  2 or more targets.

## H2 (augmented normalization keeps both)
phi(w) = [u ; lambda*m ; lambda*l]. **If** per-window and global views carry complementary
information, **then** Raw-1NN on phi is at least as good as the better of the two everywhere.
- lambda chosen from {1, 3, 10, 30} by mean VUS-PR on TSB-AD-U-Tuning (and separately on
  TSB-AD-M-Tuning for multivariate), then frozen and applied to the evaluation splits.
- **Supported** if on both evaluation splits mean VUS-PR(phi) >= mean VUS-PR(global) - 0.005,
  **and** in no z_r tercile is phi significantly (paired Wilcoxon p < 0.05) below the better of
  global and per-window in that tercile. **Refuted** if phi is significantly below the better one
  in any tercile on either split.

## H4 (not an artefact of source composition)
**If** the re-ranking is not driven by a few large sources, **then** weighting every source equally
(each series weight 1/n_source) preserves it.
- On the univariate evaluation split, recompute the mean VUS-PR of the 32 published detectors with
  and without replacing the six per-window Sub- detectors by their global variants, under source
  weights.
- **Supported** if Kendall tau between the two weighted rankings < 0.80 **and** global Sub-KNN is in
  the weighted top 3. **Refuted** if tau >= 0.90.

## H5 (the normalization choice interacts with the metric)
**If** ranking metrics hide the drift sensitivity of global normalization, **then** a calibrated
threshold exposes it.
- Raw-1NN on the univariate evaluation split with the drifts of run_drift_injection.py; training
  windows scored out of sample (2-fold); threshold = 99th percentile of those training scores;
  FAR = fraction of normal (label 0) test points above the threshold.
- **Supported** if under the +3 sigma level shift the mean FAR increase (vs. no drift) is >= 0.20
  for global and <= 0.02 for per-window normalization. **Refuted** if the global increase is < 0.05.

## H8 (foundation models behave like per-window detectors)
**If** a foundation-model detector normalizes each input instance by its own statistics, **then**
its accuracy will be nearly insensitive to z_r, like per-window detectors and unlike globally
normalized ones.
- Uses TSB-AD's published per-series VUS-PR (no re-run) of MOMENT (ZS/FT), TimesFM, Chronos,
  Lag-Llama and TSPulse (ZS/FT), classified as instance-normalizing or not **from verified primary
  sources only** (paper text or code), before looking at their z_r relation.
- Statistic: rho_X = Spearman(z_r, VUS-PR_X) over the univariate evaluation series.
- **Supported** if every instance-normalizing model has rho_X <= rho(Raw-1NN global) - 0.15.
  **Refuted** if any instance-normalizing model has rho_X >= rho(Raw-1NN global).
  Models whose normalization cannot be verified are reported but excluded from the decision.

## Changes after registration
- 2026-10-19, H8 classification (fixed before any z_r analysis of these models, as H8 requires).
  From the papers' text and code (sources and quotes in paper/icde2027/CITATION_AUDIT_ICDE.md):
  * instance-normalizing (enter the H8 decision): MOMENT ZS and FT (RevIN, paper Sec. 3.2),
    TimesFM (scaled by the mean and std of the context's first patch, paper Sec. 5),
    TSPulse ZS and FT (learnable RevIN on the input, paper p. 4);
  * partial, reported but excluded from the decision: Chronos (mean scaling with m = 0: divides by
    the mean absolute value, does not remove the level, paper Sec. 3.1) and Lag-Llama (median/IQR
    scaling per window, but location and scale are fed back as covariates, paper Sec. 4.4).
