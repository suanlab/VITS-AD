import csv

OUT = "/tmp/claude-1000/-home-suan-Projects-AI4TS-VITS/0a3ef7a6-80f7-4eac-aac4-3aaa2503b7cc/scratchpad/norm_audit.csv"
A, B, C = "A(35ba9c3b)", "B(6beac72)", "C(303dffb)"
ZS = "TSB_AD/utils/utility.py:24-35 (zscore; nan_to_num)"
FLR_B = "find_length_rank changed by cd66bc1 (periods 3-5 no longer discarded; slidingWindows.py B:46-52)"
rows = []


def add(det, sub, st, unit, cls, detail, mism, ev, chg, conf):
    rows.append(dict(detector=det, subset=sub, state=st, input_unit=unit, normalization_class=cls,
                     detail=detail, train_score_mismatch=mism, evidence_file_lines=ev,
                     changed_by_commit=chg, confidence=conf))


# ---------- classic sub-sequence detectors (univariate) ----------
def sub_uni(det, model, key, rank, wrapA, wrapB, linesA, linesB, linesC, semi=False, chgB="", chgC="none (9179dee rewrote the lines; univariate behaviour identical)"):
    if semi:
        mA = "no (same per-window zscore in fit and decision_function)"
        mBC = "yes - statistics source differs: fit standardizes train-part windows with train stats, decision_function standardizes whole-series windows with whole-series stats"
    else:
        mA = "n/a (unsupervised; scores from fit on whole series)"
        mBC = "n/a (unsupervised; stats over whole series)"
    add(det, "U", A, f"sliding windows, length find_length_rank(rank={rank}) on the series, stride 1 (Window, pandas shift)",
        "per-window", "zscore(X, axis=1, ddof=1): each window row standardized by its own mean/std", mA,
        f"{A}:TSB_AD/model_wrapper.py:{wrapA}; {A}:TSB_AD/models/{model}:{linesA}; {ZS}", "n/a (baseline)", "high")
    add(det, "U", B, f"sliding windows, length find_length_rank(rank={rank}), stride 1 (sliding_window_view); {FLR_B}",
        "column-wise across windows (whole series)" if not semi else "column-wise across windows (fit: train part; score: whole series)",
        "zscore(X, axis=0, ddof=0) on the window matrix: each lag column standardized over all windows (approx. a global z-score of the series)",
        mBC, f"{B}:TSB_AD/model_wrapper.py:{wrapB}; {B}:TSB_AD/models/{model}:{linesB}; {B}:TSB_AD/models/feature.py:26-29",
        chgB, "high")
    add(det, "U", C, f"same as B (window find_length_rank(rank={rank}))",
        "column-wise across windows (whole series)" if not semi else "column-wise across windows (fit: train part; score: whole series)",
        "zscore(X, axis=0, ddof=0) unconditionally", mBC,
        f"{C}:TSB_AD/model_wrapper.py:{wrapB}; {C}:TSB_AD/models/{model}:{linesC}", chgC, "high")


sub_uni("Sub-IForest", "IForest.py", "Sub_IForest", 1, "34-41", "51-57", "183-186", "183-189", "184-185",
        chgB="a8fc6c0 (2024-12-08: fit axis=1->axis=0 for n_features==1); 684d383 (window always applied)")
sub_uni("Sub-LOF", "LOF.py", "Sub_LOF", 2, "52-59", "66-72", "177-180", "177-184", "178-179",
        chgB="a8fc6c0 (fit axis=0 for univariate); 684d383 (window always applied)")
sub_uni("Sub-HBOS", "HBOS.py", "Sub_HBOS", 1, "152-159", "161-167", "106-109 (fit), 187-190 (decision)", "107-112", "107-108",
        chgB="a79f315 (2025-10-01 'update norm': axis=0 for univariate); 684d383")
sub_uni("Sub-MCD", "MCD.py", "Sub_MCD", 3, "188-195 (rank on data_test)", "191-197", "145-148 (fit), 199-202 (decision)",
        "146-151 (fit), 202-209 (decision)", "146-147, 200-201", semi=True,
        chgB="a79f315 (axis=0 univariate); 684d383")
sub_uni("Sub-KNN", "KNN.py", "Sub_KNN", 2, "206-213", "206-212", "190-193", "191-196", "191-192",
        chgB="a79f315 (axis=0 univariate); 684d383; 6342f12")

# Sub-OCSVM separately (mismatch + crash)
add("Sub-OCSVM", "U", A, "sliding windows, length find_length_rank(rank=2) computed on data_test (whole series), stride 1",
    "per-window", "fit: zscore(axis=1,ddof=1) per window, then MinMaxScaler().fit_transform(X.T).T = per-window min-max to [0,1]; decision_function: per-window zscore only (no min-max)",
    "yes - fit features are per-window z-score followed by per-window [0,1] min-max; scored features are per-window z-score only",
    f"{A}:TSB_AD/model_wrapper.py:170-177; {A}:TSB_AD/models/OCSVM.py:160-167 (fit), 212-215 (decision)", "n/a (baseline)", "high")
add("Sub-OCSVM", "U", B, f"sliding windows find_length_rank(rank=2); {FLR_B}",
    "n/a - detector raises before scoring",
    "intended: zscore(axis=0) (train windows in fit, whole-series windows in decision) + scaler_=MinMaxScaler().fit(X.T); scaler_.transform(X_test.T) raises ValueError (feature count = number of windows differs). Verified by running B code: wrapper returns error string, no score saved",
    "yes (crash)", f"{B}:TSB_AD/models/OCSVM.py:161-171 (fit), 217-225 (decision); {B}:TSB_AD/model_wrapper.py:176-182",
    "a79f315 (axis=0 univariate); dfd8fdb (2026-05-30: scaler_ fitted in fit and reused in decision -> crash)", "high")
add("Sub-OCSVM", "U", C, "same as B",
    "column-wise across windows (fit: train part; score: whole series)",
    "zscore(axis=0, ddof=0); MinMaxScaler removed",
    "yes - statistics source differs (train windows vs whole-series windows)",
    f"{C}:TSB_AD/models/OCSVM.py:160-161 (fit), 210-211 (decision)", "9179dee (removed MinMaxScaler; detector runs again; verified)", "high")

# Sub-PCA
add("Sub-PCA", "U", A, "sliding windows, length find_length_rank(rank=1), stride 1",
    "column-wise across windows (whole series)", "no zscore; standardizer() = sklearn StandardScaler fit on the window matrix (each lag column over all windows)",
    "n/a (unsupervised)", f"{A}:TSB_AD/model_wrapper.py:121-128; {A}:TSB_AD/models/PCA.py:226-237; {A}:TSB_AD/utils/utility.py:253-291",
    "n/a (baseline)", "high")
add("Sub-PCA", "U", B, f"sliding windows find_length_rank(rank=1); {FLR_B}",
    "column-wise across windows (whole series)", "zscore(axis=0,ddof=0) added before StandardScaler (redundant; both column-wise)",
    "n/a (unsupervised)", f"{B}:TSB_AD/model_wrapper.py:134-140; {B}:TSB_AD/models/PCA.py:229-243",
    "684d383; a79f315 (added zscore; class unchanged)", "high")
add("Sub-PCA", "U", C, "same as B", "column-wise across windows (whole series)", "zscore(axis=0) + StandardScaler",
    "n/a (unsupervised)", f"{C}:TSB_AD/models/PCA.py:229-239", "none effective (9179dee rewrote lines)", "high")

# Non-sub univariate IForest / LOF
for det, model, wA, wB, lA, lB, lC, chg in [
        ("IForest", "IForest.py", "43-50", "59-64", "181-186", "181-189", "184-185",
         "a8fc6c0 (univariate axis=0); 684d383 (run_IForest: sub=False -> slidingWindow=100 default; windows always)"),
        ("LOF", "LOF.py", "61-68", "74-79", "173-180", "177-184", "178-179",
         "a8fc6c0 (univariate axis=0); 684d383 (sub=False removed; run_LOF default slidingWindow=1)")]:
    add(det, "U", A, "single time steps (sub=False -> X kept as (T,1); find_length_rank only used for padding)",
        "per-time-step across channels (degenerate: D=1)",
        "zscore(X, axis=1, ddof=1) on 1-element rows -> std NaN -> nan_to_num -> every feature 0. Running A code gives a constant score (std 0). Published uni table values for this column are non-constant (see summary), so they cannot come from this code as written",
        "n/a (unsupervised)", f"{A}:TSB_AD/model_wrapper.py:{wA}; {A}:TSB_AD/models/{model}:{lA}; {ZS}", "n/a (baseline)", "high")
    unitB = ("sliding windows, fixed length 100 (run_IForest default slidingWindow=100; Optimal_Uni HP gives only n_estimators)"
             if det == "IForest" else "single time steps (Window(1); run_LOF default slidingWindow=1; HP gives only n_neighbors)")
    add(det, "U", B, unitB, "column-wise across windows (whole series)" if det == "IForest" else "per-channel over the series (whole series)",
        "zscore(axis=0, ddof=0)", "n/a (unsupervised)",
        f"{B}:TSB_AD/model_wrapper.py:{wB}; {B}:TSB_AD/models/{model}:{lB}; {B}:TSB_AD/HP_list.py:291-295", chg, "high")
    add(det, "U", C, unitB, "column-wise across windows (whole series)" if det == "IForest" else "per-channel over the series (whole series)",
        "zscore(axis=0, ddof=0)", "n/a (unsupervised)", f"{C}:TSB_AD/model_wrapper.py:{wB}; {C}:TSB_AD/models/{model}:{lC}",
        "none effective", "high")

# POLY, SR
for st in (A, B, C):
    add("POLY", "U", st, "whole series; polynomial fitted per segment of length find_length_rank(rank=1) over a neighborhood max(10*window,100)",
        "per-channel over the series (whole series)", "global min-max: (X - X.min())/(X.max() - X.min()) over the whole series", "n/a (unsupervised)",
        f"{st}:TSB_AD/models/POLY.py:82, 95-160; {st}:TSB_AD/model_wrapper.py:{'70-77' if st == A else '81-87'}",
        "n/a (baseline)" if st == A else "none", "high")
    add("SR", "U", st, "whole series (single FFT; window_size argument passed but unused in SR(); fixed 100-bin moving average of log-amplitude)",
        "per-channel over the series (whole series)", "global min-max over the whole series", "n/a (unsupervised)",
        f"{st}:TSB_AD/models/SR.py:6-7, 18-23; {st}:TSB_AD/model_wrapper.py:{'279-282' if st == A else '269-272'}",
        "n/a (baseline)" if st == A else ("none" if st == B else "none for normalization; 303dffb changed spectrum padding (was padding with min-max-scaled X[0]/X[-1], now freq[0]/freq[-1]) - changes scores"),
        "high")

# MatrixProfile / KShapeAD / SAND / Series2Graph
for st in (A, B, C):
    add("MatrixProfile", "U", st, "subsequences of length m = find_length_rank(rank=1) (stumpy.stump self-join)",
        "per-window (library-internal)", "no TSB-AD normalization; stumpy.stump called without normalize arg, stumpy default normalize=True = z-normalized Euclidean distance per subsequence (checked in stumpy 1.14.1 stump.py:513-551; requirements stumpy>=1.12.0)",
        "n/a (unsupervised)", f"{st}:TSB_AD/models/MatrixProfile.py:41; {st}:TSB_AD/model_wrapper.py:{'79-86' if st == A else '89-95'}",
        "n/a (baseline)" if st == A else "none", "medium")
    for det, wA, wB, semi in [("KShapeAD", "97-104", "112-118", False), ("SAND", "88-95", "104-110", True)]:
        add(det, "U", st, "subsequences: clustering on length 4*find_length_rank(rank=1) (step 1.5*w); scoring with pattern_length=find_length_rank(rank=1)" + ("; online batches, init_length=len(train), whole series passed" if semi else ""),
            "per-window (library-internal)",
            "no explicit input normalization; scoring via stumpy.stump(ts, w, centroid) = z-normalized distance (library default); k-Shape uses normalized cross-correlation (scale-invariant) and z-scored centroids (SAND.py:371); SAND.py:164-171 standardizes the score (score-level, not input)",
            "n/a (single pass over whole series)", f"{st}:TSB_AD/models/SAND.py:144, 164-171, 287-292, 371, 387-395; {st}:TSB_AD/model_wrapper.py:{wA if st == A else wB}",
            "n/a (baseline)" if st == A else "none", "medium")
    add("Series2Graph", "U", st, "whole series (squeezed); pattern_length=find_length_rank(rank=1), query_length=2*w",
        "unclear", "unclear: TSB_AD/models/Series2Graph.py is not in the repository at any state (only Series2Graph.txt, a password/licence notice); the wrapper imports .models.Series2Graph",
        "unclear", f"{st}:TSB_AD/models/Series2Graph.txt; {st}:TSB_AD/model_wrapper.py:{'106-119' if st == A else '120-132'}",
        "n/a (baseline)" if st == A else "unknown", "low")

# KMeansAD U and M
for st in (A, B, C):
    add("KMeansAD", "U", st, "sliding windows, length find_length_rank(rank=2) (run_KMeansAD_U; Optimal_Uni_algo_HP_dict has only 'KMeansAD_U'), stride 1",
        "per-window", "zscore(slides, axis=1, ddof=1) per window", "n/a (unsupervised)",
        f"{st}:TSB_AD/models/KMeansAD.py:21-27; {st}:TSB_AD/model_wrapper.py:{'231-237' if st == A else '227-232'}",
        "n/a (baseline)" if st == A else "none", "medium" )
    add("KMeansAD", "M", st, "sliding windows, fixed length 40 (HP window_size=40), flattened over channels (sliding_window_view reshape to 40*D), stride 1",
        "per-window", "zscore(axis=1, ddof=1) over all 40*D values of a window (mixes channels)", "n/a (unsupervised)",
        f"{st}:TSB_AD/models/KMeansAD.py:21-27; {st}:TSB_AD/model_wrapper.py:{'224-229' if st == A else '221-225'}",
        "n/a (baseline)" if st == A else "none", "high")

# ---------- multivariate classic ----------
def multi_pts(det, model, wA, wB, lA, lB, lC, unitB, clsB, detB, chgB, semi=False, extraA=""):
    mA = "no (same per-row zscore in fit and decision_function)" if semi else "n/a (unsupervised)"
    mC = "yes - statistics source differs: fit uses train-part per-channel stats, decision_function uses whole-series stats" if semi else "n/a (unsupervised)"
    add(det, "M", A, "single time steps (rows of D channels; sub=False and find_length_rank returns 0 for multivariate input)",
        "per-time-step across channels", "zscore(X, axis=1, ddof=1): each time step standardized across its D channels" + extraA, mA,
        f"{A}:TSB_AD/model_wrapper.py:{wA}; {A}:TSB_AD/models/{model}:{lA}; {A}:TSB_AD/utils/slidingWindows.py:9; {ZS}", "n/a (baseline)", "high")
    add(det, "M", B, unitB, clsB, detB, mA if clsB.startswith("per-time-step") else ("n/a (unsupervised)" if not semi else mA),
        f"{B}:TSB_AD/model_wrapper.py:{wB}; {B}:TSB_AD/models/{model}:{lB}", chgB, "high")
    add(det, "M", C, unitB, "per-channel over the series" + (" (fit: train part; score: whole series)" if semi else " (whole series)") if "100" not in unitB else "column-wise across windows (whole series)",
        "zscore(X, axis=0, ddof=0)" + ("; then StandardScaler (column-wise)" if det == "PCA" else ""), mC,
        f"{C}:TSB_AD/model_wrapper.py:{wB}; {C}:TSB_AD/models/{model}:{lC}", "9179dee (PR #79 'Fix multivariate normalization': axis=1 -> axis=0)", "high")

W100 = "sliding windows, fixed length 100 (wrapper default slidingWindow=100; multi HP gives none), flattened to 100*D features, stride 1"
W1 = "single time steps (Window(1); wrapper default slidingWindow=1)"
multi_pts("IForest", "IForest.py", "43-50", "59-64", "181-186", "183-189", "184-185", W100, "per-window",
          "zscore(axis=1, ddof=1) over the 100*D values of each flattened window (mixes channels)", "684d383 (windows of 100 for multivariate)")
multi_pts("LOF", "LOF.py", "61-68", "74-79", "177-180", "178-183", "178-179", W1, "per-time-step across channels",
          "zscore(axis=1, ddof=1) per time step across channels", "none effective (684d383/a79f315 touched code; multivariate behaviour same)")
multi_pts("KNN", "KNN.py", "215-222", "214-219", "190-193", "191-196", "191-192", W100, "per-window",
          "zscore(axis=1, ddof=1) over each flattened 100*D window", "684d383 (windows of 100)")
multi_pts("HBOS", "HBOS.py", "161-168", "169-174", "106-109", "107-112", "107-108", W1, "per-time-step across channels",
          "zscore(axis=1, ddof=1) per time step", "none effective")
multi_pts("MCD", "MCD.py", "197-204", "199-204", "145-148 (fit), 199-202 (decision)", "146-151 (fit), 202-209 (decision)",
          "146-147 (fit), 200-201 (decision)", W1, "per-time-step across channels", "zscore(axis=1, ddof=1) per time step in fit and decision",
          "none effective", semi=True)

# PCA multi
add("PCA", "M", A, "single time steps (sub=False)", "per-channel over the series (whole series)",
    "no zscore; StandardScaler (standardizer) fit on the (T,D) matrix = per-channel z-score over the whole series", "n/a (unsupervised)",
    f"{A}:TSB_AD/model_wrapper.py:130-137; {A}:TSB_AD/models/PCA.py:226-237", "n/a (baseline)", "high")
add("PCA", "M", B, W100, "per-window, then column-wise across windows",
    "zscore(axis=1, ddof=1) per flattened 100*D window, then StandardScaler column-wise over all windows", "n/a (unsupervised)",
    f"{B}:TSB_AD/model_wrapper.py:142-147; {B}:TSB_AD/models/PCA.py:229-243", "684d383 (windows of 100); a79f315 (zscore added)", "high")
add("PCA", "M", C, W100, "column-wise across windows (whole series)", "zscore(axis=0) then StandardScaler (both column-wise)",
    "n/a (unsupervised)", f"{C}:TSB_AD/model_wrapper.py:142-147; {C}:TSB_AD/models/PCA.py:229-239", "9179dee", "high")

# OCSVM multi
add("OCSVM", "M", A, "single time steps (find_length_rank(data_test) returns 0 for multivariate; sub=False)",
    "per-time-step across channels", "fit: zscore(axis=1,ddof=1) then per-row MinMaxScaler (min-max across channels); decision_function: zscore(axis=1) only",
    "yes - fit rows are z-scored then min-max scaled to [0,1]; scored rows are z-scored only",
    f"{A}:TSB_AD/model_wrapper.py:179-186; {A}:TSB_AD/models/OCSVM.py:160-167 (fit), 212-215 (decision)", "n/a (baseline)", "high")
add("OCSVM", "M", B, W1, "n/a - detector raises before scoring",
    "intended: zscore(axis=1) per time step + scaler_=MinMaxScaler().fit(X_train.T); scaler_.transform(X_test.T) raises ValueError (T_train != T). Verified by running B code",
    "yes (crash)", f"{B}:TSB_AD/model_wrapper.py:184-189; {B}:TSB_AD/models/OCSVM.py:161-171, 217-225", "684d383; dfd8fdb (scaler_ -> crash)", "high")
add("OCSVM", "M", C, W1, "per-channel over the series (fit: train part; score: whole series)", "zscore(axis=0, ddof=0); MinMaxScaler removed",
    "yes - statistics source differs (train-part stats in fit, whole-series stats in decision_function)",
    f"{C}:TSB_AD/model_wrapper.py:184-189; {C}:TSB_AD/models/OCSVM.py:160-161, 210-211", "9179dee", "high")

# COPOD, CBLOF, EIF
for det, model, wA, wB, ln in [("COPOD", "COPOD.py", "239-245", "234-239", "107-108"), ("CBLOF", "CBLOF.py", "247-253", "241-246", "166-169"),
                               ("EIF", "EIF.py", "263-269", "255-260", "422")]:
    for st in (A, B):
        add(det, "M", st, "single time steps (rows of D channels)", "per-time-step across channels", "zscore(X, axis=1, ddof=1)",
            "n/a (unsupervised)", f"{st}:TSB_AD/models/{model}:{ln}; {st}:TSB_AD/model_wrapper.py:{wA if st == A else wB}",
            "n/a (baseline)" if st == A else "none", "high")
    add(det, "M", C, "single time steps", "per-channel over the series (whole series)", "zscore(X, axis=0, ddof=0)", "n/a (unsupervised)",
        f"{C}:TSB_AD/models/{model}:{ln}; {C}:TSB_AD/model_wrapper.py:{wB}", "9179dee", "high")

for st in (A, B, C):
    add("RobustPCA", "M", st, "single time steps (T x D matrix)", "none", "no scaling; only all-zero columns pruned before Robust_PCA",
        "n/a (unsupervised)", f"{st}:TSB_AD/models/RobustPCA.py:84-95; {st}:TSB_AD/model_wrapper.py:{'271-277' if st == A else '262-267'}",
        "n/a (baseline)" if st == A else "none", "high")

# ---------- AutoEncoder ----------
add("AutoEncoder", "U", A, "sliding windows, fixed length 100 (HP window_size=100), stride 1",
    "per-window (+ column-wise with train stats)",
    "MinMaxScaler().fit_transform(X.T).T = each window min-max to [0,1] by its own range; then TSDataset z-scores each lag column with mean/std computed on the train windows (preprocessing=True)",
    "no (row min-max uses each window's own stats in both phases; column z-score reuses train mean/std)",
    f"{A}:TSB_AD/model_wrapper.py:284-290; {A}:TSB_AD/models/AE.py:220, 275-292 (fit), 375-386 (decision); {A}:TSB_AD/utils/dataset.py:22-23", "n/a (baseline)", "high")
add("AutoEncoder", "M", A, "single time steps (no windowing when n_features>1)",
    "per-time-step across channels (+ per-channel z-score with train stats)",
    "each time step min-max scaled across its D channels; then per-channel z-score with train-part mean/std",
    "no (as for U)", f"{A}:TSB_AD/models/AE.py:275-292, 375-386; {A}:TSB_AD/utils/dataset.py:22-23", "n/a (baseline)", "high")
for sub, unit in [("U", "sliding windows, length 100"), ("M", "single time steps")]:
    for st, chg in [(B, "dfd8fdb (2026-05-30 'fix score alignment': MinMaxScaler fitted on train, reused in decision_function)"), (C, "none (still crashes)")]:
        add("AutoEncoder", sub, st, unit, "n/a - detector raises before scoring",
            "intended: self.scaler = MinMaxScaler().fit(X_train_windowed.T) (one feature per train window/time step) then self.scaler.transform(X_test.T) -> ValueError whenever train length != series length (always in Run_Detector). Verified by running B and C code for U and M",
            "yes (crash)", f"{st}:TSB_AD/models/AE.py:285-286 (fit), 382 (decision)", chg, "high")

# ---------- deep models on ReconstructDataset / ForecastDataset ----------
DS_A = f"{A}:TSB_AD/utils/dataset.py:29-79 (ReconstructDataset), 82-129 (ForecastDataset): normalized array stored in self.data but samples built from the raw `data` argument"
DS_B = "TSB_AD/utils/dataset.py:6-20 (ReconstructDataset), 40-58 (ForecastDataset)"
deep = [
    ("CNN", "CNN.py", "ForecastDataset", "U 50 / M 50 (HP window_size), forecast next step", "292-298", "281-286", "130,135,206", "124,129,200", None, ""),
    ("LSTMAD", "LSTMAD.py", "ForecastDataset", "U 100 / M 150 (HP window_size), forecast next step", "300-306", "288-293", "94,99,171", "94,99,171", None, ""),
    ("TranAD", "TranAD.py", "ReconstructDataset", "U 10 / M 10 (HP win_size)", "308-314", "295-300", "165,171,247", "161-162,168,174,249,252", None, ""),
    ("AnomalyTransformer", "AnomalyTransformer.py", "ReconstructDataset", "U 50 / M 50 (HP win_size)", "316-322", "302-307", "393,399,471", "392,398,470", None, ""),
    ("OmniAnomaly", "OmniAnomaly.py", "ReconstructDataset", "U 5 / M 100 (HP win_size)", "324-330", "316-321", "108,114,180", "144,150,223", None, ""),
    ("USAD", "USAD.py", "ReconstructDataset", "U 100 / M 100 (HP win_size)", "332-338", "323-328", "102,108,182", "102,108,182", None, ""),
    ("Donut", "Donut.py", "ReconstructDataset", "U 60 / M 60 (HP win_size)", "340-346", "330-335", "336,342,365", "268,333,336,342,365", None, ""),
    ("TimesNet", "TimesNet.py", "ReconstructDataset", "U 32 / M 96 (HP win_size)", "348-354", "337-342", "222,228,287", "222,228,287",
     "Non-stationary instance norm: each window, per channel, mean/std over time (x_enc.mean(1), var dim=1) with de-normalization of the output", "149-154, 164-170"),
    ("FITS", "FITS.py", "ReconstructDataset", "U 100 / M 100 (HP win_size)", "356-362", "344-349", "124,130,184", "124,130,184",
     "RIN: each window, per channel, mean/var over time, reversed on output", "49-54, 75"),
    ("OFA", "OFA.py", "ReconstructDataset", "U 50 / M 50 (HP win_size), split into 25-step segments", "364-370", "351-356", "239,245,305", "241,247,306",
     "instance norm per 25-step segment per channel inside each window (rearrange b (n s) m, mean/var over s), de-normalized on output", "152-160, 178-185"),
]
for det, model, dsname, unit, wA, wB, lA, lB, internal, il in deep:
    unit_full = f"sliding windows, stride 1, length {unit}; {dsname}"
    if internal:
        clsA = "model-internal instance normalization"
        detA = f"{internal}. TSB-AD dataset normalization is computed but not applied (samples built from raw data)"
        evA = f"{A}:TSB_AD/models/{model}:{il} (internal), {lA} (datasets); {DS_A}; {A}:TSB_AD/model_wrapper.py:{wA}"
    else:
        clsA = "none"
        detA = f"{dsname}(normalize=True) computes a z-scored copy (self.data) but builds samples from the raw `data` argument, so windows are raw values; no other input scaling. Verified by instantiating the A dataset class (window mean equals raw mean)"
        evA = f"{A}:TSB_AD/models/{model}:{lA}; {DS_A}; {A}:TSB_AD/model_wrapper.py:{wA}"
    add(det, "both", A, unit_full, clsA, detA, "no (raw data in both phases" + ("; instance norm uses each window's own stats)" if internal else ")"), evA, "n/a (baseline)", "high")
    detB = (f"{dsname} z-scores each channel using mean/std of the array passed in: fit -> tsTrain (first 80% of train part) and tsValid separately; decision_function -> whole series")
    if det == "TranAD":
        detB += "; additionally MinMaxScaler fit on the full train part and applied to the whole series before the dataset (TranAD.py:161-162, 249), which the subsequent per-channel z-score largely cancels (affine)"
    if internal:
        detB += f"; plus {internal}"
    clsB = "column-wise per channel (fit: train part; score: whole series)" + (" + model-internal instance normalization" if internal else "")
    chgB = "dce9e5e (2025-03-16 'update datasets class supporting multi stride': normalization now actually applied)" + ("; 7e2dfd8 (2026-04-15 added MinMaxScaler)" if det == "TranAD" else "")
    mB = "yes - statistics source differs: fit standardizes with train-part stats (tsTrain and tsValid each with their own stats), decision_function with whole-series stats (including anomalies)"
    if det == "Donut":
        for st, chg in [(B, "dce9e5e (dataset now raises on an empty array)"), (C, "none (still crashes)")]:
            add(det, "both", st, unit_full, "n/a - detector raises before scoring",
                "validation_size=0 -> tsValid is empty -> ReconstructDataset(tsValid) calls torch.stack on an empty list -> RuntimeError. Verified by running C code. Intended normalization: per-channel z-score (train part in fit, whole series in decision)",
                "yes (crash)", f"{st}:TSB_AD/models/Donut.py:268, 332-345; {st}:{DS_B}, 22-30", chg, "high")
        continue
    for st in (B, C):
        add(det, "both", st, unit_full, clsB, detB, mB,
            f"{st}:TSB_AD/models/{model}:{lB}" + (f", {il} (internal)" if internal else "") + f"; {st}:{DS_B}; {st}:TSB_AD/model_wrapper.py:{wB}",
            chgB if st == B else "none", "high")

# ---------- foundation models (U only) ----------
for st in (A, B, C):
    w = {"Lag-Llama": ("372-378", "358-363"), "Chronos": ("380-386", "365-370"), "TimesFM": ("388-394", "372-377")}
    add("Lag-Llama", "U", st, "sliding windows of length 96 (HP win_size), stride 1, each window passed as its own series; predict next step",
        "unclear", "no TSB-AD normalization (MinMax line commented out). Library scaling = estimator_args['scaling'] read from the checkpoint; its value is not in the code, and lag-llama is not installed here to check",
        "n/a (zero-shot)", f"{st}:TSB_AD/models/Lag_Llama.py:44-64, 84, 128-139; {st}:TSB_AD/model_wrapper.py:{w['Lag-Llama'][0] if st == A else w['Lag-Llama'][1]}",
        "n/a (baseline)" if st == A else "none", "low")
    add("Chronos", "U", st, "sliding windows of length 100 (HP win_size), stride 1, each window an item_id in AutoGluon TimeSeriesPredictor; predict next step",
        "model-internal instance normalization (library)", "no TSB-AD normalization (MinMax line commented out); Chronos tokenizer divides each context by its mean absolute value (chronos-forecasting 2.2.2 chronos.py:176-180; reached through AutoGluon, version not pinned by TSB-AD)",
        "n/a (zero-shot)", f"{st}:TSB_AD/models/Chronos.py:31-60, 83-94; {st}:TSB_AD/model_wrapper.py:{w['Chronos'][0] if st == A else w['Chronos'][1]}",
        "n/a (baseline)" if st == A else "none", "medium")
    add("TimesFM", "U", st, "sliding windows of length 96 (HP win_size), stride 1, list of contexts to tfm.forecast; predict next step",
        "unclear", "no TSB-AD normalization (MinMax line commented out); timesfm library internals not checked (package not installed here, version not pinned)",
        "n/a (zero-shot)", f"{st}:TSB_AD/models/TimesFM.py:21-58, 76-87; {st}:TSB_AD/model_wrapper.py:{w['TimesFM'][0] if st == A else w['TimesFM'][1]}",
        "n/a (baseline)" if st == A else "none", "low")

REVIN = "momentfm RevIN(num_features=1, affine=False): each window, per channel, normalized and output de-normalized (momentfm 0.1.5 moment.py:105-107, 310, 337)"
add("MOMENT (ZS)", "U", A, "sliding windows of length 64 (HP win_size), stride 1 (ReconstructDataset_Moment)", "model-internal instance normalization",
    f"TSB-AD dataset normalization computed but not applied (raw windows); {REVIN}", "n/a (zero-shot)",
    f"{A}:TSB_AD/models/MOMENT.py:57-84; {A}:TSB_AD/utils/dataset.py:132-166; {A}:TSB_AD/model_wrapper.py:396-404", "n/a (baseline)", "medium")
for st in (B, C):
    add("MOMENT (ZS)", "U", st, "sliding windows of length 64, stride 1", "column-wise per channel (whole series) + model-internal instance normalization",
        f"ReconstructDataset_Moment z-scores the series with whole-series mean/std; then {REVIN}", "n/a (zero-shot)",
        f"{st}:TSB_AD/models/MOMENT.py:57-84; {st}:TSB_AD/utils/dataset.py:149-165; {st}:TSB_AD/model_wrapper.py:379-386",
        "dce9e5e" if st == B else "none", "medium")
add("MOMENT (FT)", "U", A, "sliding windows of length 64, stride 1; fine-tuned on the train part (validation_size=0)", "model-internal instance normalization",
    f"TSB-AD dataset normalization computed but not applied (raw windows); {REVIN}", "no (raw in both phases; RevIN per window)",
    f"{A}:TSB_AD/models/MOMENT.py:87-101, 168-199; {A}:TSB_AD/utils/dataset.py:132-166; {A}:TSB_AD/model_wrapper.py:406-414", "n/a (baseline)", "medium")
for st, chg in [(B, "dce9e5e (dataset now raises on an empty array)"), (C, "none")]:
    add("MOMENT (FT)", "U", st, "sliding windows of length 64", "n/a - detector raises before scoring",
        "validation_size=0 -> tsValid empty -> ReconstructDataset_Moment(tsValid) calls torch.stack on an empty list -> RuntimeError (verified on the dataset class; the full model was not run). Intended: per-channel z-score (train in fit, whole series in decision) + RevIN",
        "yes (crash)", f"{st}:TSB_AD/models/MOMENT.py:27, 88-101; {st}:TSB_AD/utils/dataset.py:149-183; {st}:TSB_AD/model_wrapper.py:410-417", chg, "medium")

FIELDS = ["detector", "subset", "state", "input_unit", "normalization_class", "detail", "train_score_mismatch",
          "evidence_file_lines", "changed_by_commit", "confidence"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    order = {A: 0, B: 1, C: 2}
    for r in sorted(rows, key=lambda r: (r["detector"], r["subset"], order[r["state"]])):
        w.writerow(r)
print(len(rows))
