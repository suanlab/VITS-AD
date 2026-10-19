# Pre-registered hypotheses: outcomes

Decision rules: `normalization_hypotheses.md` (commit 1906af3, 2026-10-19 22:00 KST; H8 classification
fixed in 55b2543 before its analysis). Every outcome is reported, including refutations.

| Hypothesis | Outcome | Key numbers |
|---|---|---|
| H4 source composition | **Supported** | source-balanced weights (23 sources): Kendall tau = 0.597 (< 0.80); global Sub-KNN ranks 1st (0.542); Sub-OCSVM 28->4, Sub-MCD 29->5 |
| H8 foundation models behave like per-window detectors | **Inconclusive** (neither criterion met) | rho(z_r, VUS-PR): Raw-1NN global 0.408 (threshold 0.258). MOMENT ZS 0.344, MOMENT FT 0.341, TimesFM 0.368, TSPulse ZS 0.285, TSPulse FT 0.213. Only TSPulse FT <= threshold; none >= 0.408. Context (not in decision): per-window detectors Sub-KNN -0.078, Sub-MCD 0.097, per-window 1-NN 0.086; global/standardized Sub-PCA 0.572, CNN 0.557; partial normalizers Chronos 0.421, Lag-Llama 0.435 |
| H6 held-out replication | pending | |
| H1 information decomposition | pending | |
| H2 augmented normalization | pending | |
| H5 calibrated-threshold FAR under drift | **Supported** | Raw-1NN, 350 univariate series, threshold = 99th pct of out-of-sample training scores. Mean FAR, no drift -> +3 sigma level shift: global 0.148 -> 0.837 (+0.689; criterion >= 0.20), per-window 0.033 -> 0.041 (+0.008; criterion <= 0.02). Other drifts, global / per-window increase: +1 sigma +0.348 / +0.006; scale x2 +0.529 / +0.002; trend +0.412 / +0.023. Note: without drift the global FAR (0.148) already exceeds the nominal 1%, the per-window FAR (0.033) much less |

## Verifiability of the registration time
Commit timestamps are set locally. The first server-side record is GitHub's push event for
`suanlab/VITS-AD` at 2026-10-19T14:30:53Z (repository activity API), which contains the registration
(1906af3) and the H8 classification (55b2543). H4, H8 and H5 were analysed before that push, so for
them the order "registration before analysis" rests on the commit history only. H6, H1 and H2 had
not been analysed at the time of the push.
