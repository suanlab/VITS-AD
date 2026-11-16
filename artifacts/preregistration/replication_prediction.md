# Pre-registration: M, predicting which benchmark conclusions replicate on new sources

Written and pushed before any held-out value used below was looked at. Results will be reported
whatever they are. Changes after this commit, if any, will be listed at the end with reason and date.

## What was known when this was written
- All results of the paper, including TR (a prediction from Yahoo subsets transported to unseen
  YAHOO series with half the error of the plain YAHOO mean) and an exploratory census of TSB-AD's
  published per-series table (2026-11-10, `research/breakthrough/probes/p1_*`): 99.6% of the
  statistically significant method pairs have at least one data source that reverses them.
- Motivation: simulated reviews summarize the paper's general finding as "it depends"; the
  research note `research/breakthrough/2026-11-10-benchmark-conclusion-certification.md` proposes
  turning it into a prediction tested prospectively. Closest prior work (verified there):
  Voorhees & Buckley (SIGIR 2002, swap rates from score differences and topic-set size), Hu, Liu &
  Fu (arXiv 2606.29159, margin against subsystem spread, descriptive), Srinivasan (arXiv
  2608.11323, generalizability-theory reliability that failed to predict held-out reliability).
- RT: its grid on the 48 tuning series and the selected configurations (`retune_selection.json`,
  with their tuning means) were known. Its runs on the evaluation split and the 472 held-out series
  finished on 2026-11-11; only their completeness (every changed configuration has a record for every
  series) was checked. No value of them has been looked at.
- TimeEval: the published result table (`data/timeeval_results/timeeval_results.csv`, SHA-256
  49961990bda2450b6b1ec9a5632d3a29cdd43ce55c02192b67d2fe8c5f5d8288, downloaded 2026-11-16 from the
  TimeEval authors' public share link) was inspected for structure only: 60 algorithms, 16
  collections, 976 datasets (902 univariate in 9 collections: WebscopeS5 360, KDD-TSAD 249,
  GutenTAG 163, NAB 56, NASA-SMAP 35, NASA-MSL 16, MGAB 10, NormA 9, IOPS 4), statuses (39,784 OK,
  2,992 error, 2,355 timeout), one hyperparameter setting and one repetition per cell, and 29
  algorithms with a result on at least 90% of the univariate datasets. No metric value was printed.
- The code was run once on a synthetic table with TimeEval's columns and once on the RT tuning table
  (discovery data) to check that it runs; only shapes and counts were printed.

## Procedure (fixed): `scripts/replication_prediction.py` (this commit)
- A conclusion is a pairwise order "a above b" read from a discovery set as the sign of the mean
  per-series difference d.
- Predicted probability that the order holds on new data:
  - **p_new** (the rule under test): Phi(s mu / sqrt(tau^2 B + se_mu^2 + sigma_w^2 / n_new)), from a
    DerSimonian-Laird random-effects fit over discovery sources with at least two series (tau^2 = 0
    with fewer than two), s = sign(d), B = 1 for one new source and the sum of squared source weights
    for a pooled held-out set;
  - **p_iid** (primary baseline, same-population replication, no between-source term):
    Phi(|d| / sqrt(se_iid^2 + sigma^2 / n_new));
  - **p_sig** (secondary baseline, the significance view): Phi(|d| / se_iid).
- Outcomes: **O1** (primary), per conclusion and held-out source, whether the source's mean
  difference has sign s; **O2** (secondary), per conclusion, whether the held-out set's series-weighted
  mean difference has sign s.
- Targets:
  - **rt_eva**: discovery = RT's 48 tuning series, conclusions = the 66 pairs of the twelve re-tuned
    variants (six Sub- detectors x per-window/global, each at its selected configuration; a
    configuration equal to TSB-AD's tuned one takes the toggle's existing result); held out = the
    evaluation split (350 series, 23 sources);
  - **rt_rest**: the same conclusions; held out = the 472 series in neither split (4 sources);
  - **timeeval**: univariate datasets, ROC_AUC, the algorithms with a result on at least 90% of them;
    every split of the 9 collections into 5 discovery and 4 held-out collections (126 splits),
    conclusions = all pairs; per pair, datasets where both have a result.
- Scores per target and prediction: Brier score; ROC-AUC over instances; calibration slope.

## Hypothesis and decision rule
**M.** If between-source heterogeneity is what makes benchmark conclusions fail to replicate, a
prediction that includes it (p_new) foresees replication on new sources better than one that
treats new data as the same population (p_iid).
- On O1, for each of the three targets: "better" if Brier(p_new) <= Brier(p_iid) - 0.02 **and**
  AUC(p_new) >= AUC(p_iid) + 0.05; "no gain" if Brier(p_new) >= Brier(p_iid) - 0.005 **and**
  AUC(p_new) < AUC(p_iid) + 0.02.
- **Supported** if "better" on at least 2 of the 3 targets; **refuted** if "no gain" on all 3;
  otherwise **inconclusive**.
- Behaviour with no heterogeneity: tau^2 = 0 makes p_new differ from p_iid only through se_mu and
  the pooled within-source variance, so the rule cannot pass by construction; with few discovery
  sources tau^2 is noisy, which works against p_new.
- Instances share conclusions and sources, so they are not independent; the decision uses the
  scores themselves, not tests on them.

## Secondary (reported, no rule)
- O2 and p_sig for every target; calibration slopes; the same on TimeEval with PR_AUC.
- RT's own registered decisions (RT1-RT3, `retune_hypothesis.md`) are computed after this
  analysis's predictions are fixed and are reported separately.

## Changes after registration
(none)
