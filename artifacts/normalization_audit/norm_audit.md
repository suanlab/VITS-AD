# How TSB-AD detectors normalize their input (code audit)

The per-detector, per-state entries are in `norm_audit.csv`: 135 rows, covering 45 detector/subset entries in each of 3 states. The 14 detectors that appear in both tables are split into separate U and M rows where their code paths differ (IForest, LOF, KMeansAD, AutoEncoder). The 10 deep models share one code path and are stored as a single "both" row each.

## States audited
| State | Commit | Date |
|---|---|---|
| A | 35ba9c3b | 2024-11-29 |
| B | 6beac72e | 2026-09-07 |
| C | 303dffbf (HEAD of main) | 2026-10-20 |

- **State A** is the commit that uploaded the published `*mergedTable*`. No detector code changed in that commit; the newest code change before it is f2bf6b3 "fix norm" (2024-11-06), which added `zscore(axis=1)` and the dataset `normalize=True` flags.
- **State C** includes PR #79 (9179dee). It also includes 061bcb5 and 303dffb, which fix FFT and SR.

**The benchmark scripts do not normalize anything.** In all three states, `benchmark_exp/Run_Detector_U.py` and `Run_Detector_M.py` (lines 55-71) read the CSV as raw floats and set `data_train = data[:train_index]`. They pass `(data_train, data)` or `data` straight to the wrapper. The only differences between the states are the license header and the `get_metrics` call.

At state A the wrapper also min-max scales every output score to [0,1]. Commit e298726 (2024-12-20) removed this. It is score post-processing, not input normalization.

## Counts at state A (normalization_class)
**Univariate (32 detectors):**

| Class | n | Detectors |
|---|---|---|
| per-window (TSB-AD code) | 8 | Sub-IForest, Sub-LOF, Sub-HBOS, Sub-OCSVM, Sub-MCD, Sub-KNN, KMeansAD, AutoEncoder |
| per-window (library-internal) | 3 | MatrixProfile, KShapeAD, SAND (stumpy z-normalized distance) |
| per-channel / column-wise over the whole series | 3 | POLY and SR (global min-max), Sub-PCA (StandardScaler over windows) |
| per-time-step, degenerate (D=1) | 2 | IForest, LOF: every feature becomes 0 |
| model-internal instance normalization | 6 | TimesNet, FITS, OFA, MOMENT (ZS), MOMENT (FT) (RevIN), Chronos (library mean scaling) |
| none | 7 | CNN, LSTMAD, TranAD, AnomalyTransformer, OmniAnomaly, USAD, Donut |
| unclear | 3 | Series2Graph, Lag-Llama, TimesFM |

AutoEncoder applies a per-window min-max, then a column z-score using train statistics.

**Multivariate (23 detectors):**

| Class | n | Detectors |
|---|---|---|
| per-time-step across channels | 10 | IForest, LOF, HBOS, OCSVM, MCD, KNN, COPOD, CBLOF, EIF, AutoEncoder (per-time-step min-max) |
| per-channel over the whole series | 1 | PCA (StandardScaler) |
| per-window | 1 | KMeansAD (40×D flattened window) |
| model-internal instance normalization | 3 | TimesNet, FITS, OFA |
| none | 8 | RobustPCA, CNN, LSTMAD, TranAD, AnomalyTransformer, OmniAnomaly, USAD, Donut |

## What changed between A and C
**Classical detectors:**
- **Univariate Sub-IForest, Sub-LOF, Sub-HBOS, Sub-MCD, Sub-KNN:** moved from per-window `zscore(axis=1)` to column-wise across windows `zscore(axis=0)`, which is roughly a global z-score of the series. This came from a8fc6c0 (IForest/LOF, 2024-12-08), 684d383 (windowing always applied) and a79f315 ("update norm", 2025-10-01).
- **Sub-PCA:** a79f315 added a `zscore(axis=0)` before StandardScaler. The class did not change.
- **Univariate IForest:** went from a single time step to a fixed 100-length window (684d383).
- **Univariate LOF:** now uses Window(1) with a global z-score.
- **Multivariate IForest, KNN, PCA:** B introduced fixed 100×D flattened windows z-scored per window (684d383). C switched these to column-wise (9179dee).
- **Multivariate LOF, HBOS, MCD, COPOD, CBLOF, EIF:** per-time-step across channels in A and B. C changed them to per-channel over the series (9179dee).
- **OCSVM (U and M):** A has a train/score mismatch. B crashes. C is per-channel or column-wise (9179dee).

**Deep and dataset-based detectors** (CNN, LSTMAD, TranAD, AnomalyTransformer, OmniAnomaly, USAD, Donut, TimesNet, FITS, OFA, MOMENT ZS/FT):
- In A the dataset normalization is dead code, so the input is raw.
- From dce9e5e (2025-03-16) onward it is a per-channel z-score. The statistics come from the train part in `fit` and from the whole series in `decision_function`.
- TranAD also gained a train-fitted MinMaxScaler (7e2dfd8).

**AutoEncoder:** crashes in B and C (dfd8fdb).

**Unchanged normalization:** POLY, SR, MatrixProfile, KShapeAD, SAND, KMeansAD, RobustPCA, Lag-Llama, Chronos, TimesFM, Series2Graph (code absent).
- SR's padding fix (303dffb) changes its scores but not its normalization.
- cd66bc1 changed the `find_length_rank` window rule for every period-based detector: periods 3-5 are now accepted, and the cutoff became 300 on the period rather than on the lag.

## Surprising findings
1. **In A, the dataset normalization for all deep models is dead code.** `ReconstructDataset`, `ForecastDataset` and `ReconstructDataset_Moment` store a z-scored copy in `self.data`, but build the samples from the raw `data` argument (`utils/dataset.py` A:35-64, 87-121, 138-164). I instantiated the A classes to confirm: window means equal the raw mean (about 100), compared with about 0 at B. So the published A-era deep-model numbers were run on raw-scale inputs, apart from TimesNet, FITS and OFA, which normalize inside the model.
2. **Univariate IForest and LOF at A see all-zero features.** With `sub=False`, X has shape (T,1), and `zscore(axis=1, ddof=1)` on a single-value row gives NaN, which `nan_to_num` turns into 0. Running the A wrapper on a synthetic series gives a constant score (std 0) for both detectors.
   - The published uni table is not constant for these columns. IForest has mean VUS-PR 0.298 and max 1.0, LOF has mean 0.169, and the mean anomaly ratio is 0.045.
   - So those two columns cannot have come from the A code as written. They were probably run before f2bf6b3 (no normalization) or with other code, but that is not settled.
3. **Train/score mismatch at A for OCSVM and Sub-OCSVM.** `fit` applies a z-score followed by a per-row MinMaxScaler to [0,1]. `decision_function` applies only the z-score (OCSVM.py A:163,167 vs 215).
4. **Detectors that crash in B and/or C,** returning an error string so no score is saved:
   - AutoEncoder, U and M, in B and C. The MinMaxScaler is fitted on X.T, so its feature count equals the number of train windows (dfd8fdb). Verified by running B and C code.
   - OCSVM and Sub-OCSVM in B. Same cause; fixed by 9179dee. Verified.
   - Donut in B and C. With `validation_size=0`, the validation `ReconstructDataset` calls `torch.stack([])`. Verified on C.
   - MOMENT (FT) in B and C. Same empty validation dataset; verified on the dataset class only, not the full model.
5. **Multivariate per-time-step z-scoring in A erases channel scale** (as PR #79 says). For D=2 every row becomes (±0.707, ∓0.707). In B, multivariate IForest, KNN and PCA instead z-score each 100×D flattened window jointly across channels.
6. **The window rule is mostly not data-dependent.** `find_length_rank` returns 0 for multivariate input (slidingWindows.py:9), so multivariate classical detectors in A work on single time steps. `SR(X, window_size)` ignores `window_size`.
7. **Statistics in B and C are computed over the whole series, anomalies included.** For semi-supervised detectors, `fit` standardizes with train-part statistics and scoring standardizes with whole-series statistics. This affects MCD, OCSVM and all ReconstructDataset/ForecastDataset models; the deep models also split train and validation with separate statistics.

## Entries marked unclear or low confidence
- **Series2Graph (U, all states):** `models/Series2Graph.py` is not in the repo. There is only `Series2Graph.txt`, a patent/password notice. The wrapper imports a module that does not exist in the clone.
- **Lag-Llama (U, all states):** `scaling=estimator_args["scaling"]` is read from the checkpoint (Lag_Llama.py:84). Its value is not in the code, and lag-llama is not installed here to check.
- **TimesFM (U, all states):** TSB-AD applies no normalization. The library internals were not checked (package not installed, version not pinned).
- **Medium confidence:**
  - MatrixProfile, KShapeAD, SAND and Chronos rely on library defaults (stumpy `normalize=True`, tslearn NCC, Chronos mean-abs scaling). I checked these in locally installed versions, but TSB-AD does not pin them.
  - MOMENT relies on RevIN in momentfm 0.1.5.
  - Mapping the uni table column "KMeansAD" to `run_KMeansAD_U` is inferred from the keys of the HP dict.

## Provenance
Produced by reading TSB-AD's source at 35ba9c3b (2024-11-29, state A), 6beac72e (state B) and 303dffbf
(2026-10-20, state C), with behaviour checked by running TSB-AD code on synthetic data where noted.
Independently re-checked by hand (2026-10-21): at state A, `TSB_AD/utils/dataset.py`
`ReconstructDataset`/`ForecastDataset` compute a z-scored `self.data` but build windows from the
raw `data`, so the normalization does not reach the models.
