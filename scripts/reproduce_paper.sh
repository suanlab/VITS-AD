#!/usr/bin/env bash
# Reproduce the paper on window normalization in TSB-AD (PVLDB Vol. 20 EA&B submission).
#
#   bash scripts/reproduce_paper.sh          # regen: GPU-free, regenerate every number, table and
#                                            # figure from artifacts/official_protocol/
#   bash scripts/reproduce_paper.sh core     # re-run the CPU experiments the paper reports (needs
#                                            # the TSB-AD series in data/tsb_ad/, see README), then regen
#   bash scripts/reproduce_paper.sh legacy   # re-run the rendered-pipeline (vision) runs of earlier
#                                            # work, not reported in the paper (GPU), then regen
#   bash scripts/reproduce_paper.sh full     # core and legacy
#
# The experiment modes run each script unsharded; the per-series runners (run_detector_normalization,
# run_family_toggle, run_official_protocol, run_normalization_ablation, run_drift_threshold2,
# run_threshold_sub) accept --shard k --num-shards K, which is how the released results were
# produced. Since 2026-10-29 they stop if a listed series is missing and if an existing output was
# started with a different configuration (see scripts/run_contract.py); outputs written before then
# have no .fingerprint file. Families that a mode does not re-run keep
# their released files; collect_official_artifacts lists them in collect_report.json.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

regenerate() {
  $PY -m scripts.fetch_tsbad_official
  $PY -m scripts.summarize_official --variant default
  $PY -m scripts.summarize_official --variant crossfit
  $PY -m scripts.paper_numbers
  $PY -m scripts.make_paper_tables --out paper/pvldb2027/tables
  $PY -m scripts.make_normalization_figures --out paper/pvldb2027/figures
  $PY -m scripts.make_extended_results
  $PY -m scripts.make_contract_records
  # regen checks the released manifest without rewriting it; experiment modes write a new one.
  if [[ "${1:-}" == write ]]; then $PY -m scripts.make_release_manifest; else
    $PY -m scripts.make_release_manifest --verify || { echo "differences from the released manifest (listed above)" >&2; return 1; }; fi
}

MODE=${1:-regen}
case "$MODE" in
  regen) regenerate; exit 0 ;;
  core|legacy|full) ;;
  *) echo "unknown mode $MODE (regen, core, legacy, full)" >&2; exit 1 ;;
esac

U=TSB-AD-U; M=TSB-AD-M
# Check everything the experiment modes need before hours of compute.
for v in TSBAD_SRC_UPLOAD TSBAD_SRC_PINNED TSBAD_SRC_HEAD; do
  [[ -d "${!v:-}" ]] || { echo "set $v to an unpacked TSB-AD checkout (see README)" >&2; exit 1; }
done
for d in data/tsb_ad/TSB-AD-U data/tsb_ad/TSB-AD-M; do
  [[ -d "$d" ]] || { echo "missing $d: unzip TSB-AD's data first (see README)" >&2; exit 1; }
done
# All released runs used one BLAS/OpenMP thread per process; Sub-MCD's values depend on the
# thread count at the 1e-3 level.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

core() {
# Sec. 4.2: TSB-AD's detectors with per-window vs. global normalization (CPU)
$PY -m scripts.run_detector_normalization --subset $U
$PY -m scripts.run_detector_normalization --subset $M --detectors KNN,LOF,HBOS,OCSVM
$PY -m scripts.run_detector_normalization --subset $M --detectors MCD,IForest,PCA --part _e5
# Sec. 4.2, 5.1: two more per-window families, Matrix Profile and KMeansAD (registered H10)
$PY -m scripts.run_family_toggle --split eva
$PY -m scripts.run_family_toggle --split rest
for sd in 1 2; do                               # H10 amendment: KMeans seeds 1 and 2
  $PY -m scripts.run_family_toggle --split eva --seed $sd --detectors KMeansAD_U
  $PY -m scripts.run_family_toggle --split rest --seed $sd --detectors KMeansAD_U
done
$PY -m scripts.run_detector_normalization --subset $U --ocsvm-clean --detectors Sub_OCSVM
$PY -m scripts.run_detector_normalization --subset $M --ocsvm-clean --detectors OCSVM
# Sec. 5.1, held-out replications: H6 (tuning split) and H9 (472 series in neither split);
# near-identical series across splits (raw values only), used for H9's secondary analysis
$PY -m scripts.find_near_duplicates
$PY -m scripts.run_detector_normalization --subset $U --split tuning \
  --detectors Sub_KNN,Sub_MCD,Sub_OCSVM,Sub_HBOS,Sub_IForest,Sub_LOF
$PY -m scripts.run_detector_normalization --subset $M --split tuning
$PY -m scripts.run_detector_normalization --subset $U --split rest \
  --detectors Sub_KNN,Sub_MCD,Sub_OCSVM,Sub_HBOS,Sub_IForest,Sub_LOF
$PY -m scripts.run_detector_normalization --subset $U --split rest --ocsvm-clean --detectors Sub_OCSVM
# Sec. 5.2, H11: TSB-UAD sources absent from TSB-AD (needs data/tsb_uad/TSB-UAD-Public.zip, see README)
$PY -m scripts.convert_tsbuad
for SP in uad_primary uad_secondary; do
  $PY -m scripts.run_detector_normalization --subset $U --split $SP \
    --detectors Sub_KNN,Sub_MCD,Sub_OCSVM,Sub_HBOS,Sub_IForest,Sub_LOF
  $PY -m scripts.run_official_protocol --subset $U --split $SP --arms raw_knn,raw_knn_wz,raw_flatten --tag knn_$SP
done
# Sec. 5.2, TR: the YAHOO series of TSB-UAD that TSB-AD left out (values-only match, then the toggle)
$PY -m scripts.match_tsbuad_shared
$PY -m scripts.convert_tsbuad_shared
$PY -m scripts.run_detector_normalization --subset $U --split uad_shared \
  --detectors Sub_KNN,Sub_MCD,Sub_OCSVM,Sub_HBOS,Sub_IForest,Sub_LOF
# Sec. 5.4, P: TimeEval's GutenTAG collection (needs data/timeeval_gutentag/raw/gutentag.zip, see README);
# predictions are fixed in artifacts/preregistration/gutentag_predictions.json (--check recomputes them)
$PY -m scripts.convert_gutentag
$PY -m scripts.gutentag_prediction --check
$PY -m scripts.run_detector_normalization --subset $U --split gutentag
# Sec. 5.2, SP: anomalies injected into real backgrounds
$PY -m scripts.run_spikein plan
$PY -m scripts.run_spikein run
$PY -m scripts.run_spikein run --fixed-w            # exploratory: detector window fixed at the background's W
# Sec. 4.2, 5.1, RT: both variants re-tuned on the tuning split (the grid is long; shard with --shard/--part)
$PY -m scripts.run_retune grid
$PY -m scripts.run_retune select
for SPL in eva rest; do $PY -m scripts.run_retune apply --split $SPL; done
# Sec. 5.3, M: registered replication prediction on RT and on TimeEval's published results (the table is
# not redistributed; download it from the link in the README and set TIMEEVAL_CSV, checked by SHA-256)
TE=${TIMEEVAL_CSV:-data/timeeval_results/timeeval_results.csv}
if [[ -f "$TE" ]]; then
  echo "49961990bda2450b6b1ec9a5632d3a29cdd43ce55c02192b67d2fe8c5f5d8288  $TE" | sha256sum -c -
  $PY -m scripts.replication_prediction --timeeval "$TE"
  $PY -m scripts.replication_prediction_explore --timeeval "$TE"   # exploratory follow-ups (not registered)
else echo "skipping M: $TE not found (the released replication_prediction.json is kept)" >&2; fi
# Sec. 4.4, TSB-AD's code run unmodified: the commit of the original analysis and the current HEAD
PYTHONPATH=$TSBAD_SRC_PINNED $PY -m scripts.run_tsbad_pinned_detectors --tsbad-src "$TSBAD_SRC_PINNED"
PYTHONPATH=$TSBAD_SRC_HEAD $PY -m scripts.run_tsbad_pinned_detectors --tsbad-src "$TSBAD_SRC_HEAD" \
  --subset $U --detectors Sub_OCSVM --tag head
PYTHONPATH=$TSBAD_SRC_HEAD $PY -m scripts.run_tsbad_pinned_detectors --tsbad-src "$TSBAD_SRC_HEAD" \
  --subset $M --tag head
# Sec. 8: training-free baselines, default and cross-fitted, and the stride-1 check (CPU)
for S in $U $M; do
  $PY -m scripts.run_official_protocol --subset $S --arms raw_meanpool,raw_flatten --tag raw
  $PY -m scripts.run_official_protocol --subset $S --arms raw_flatten --crossfit-train --tag rawcf
  $PY -m scripts.run_official_protocol --subset $S --arms raw_knn,raw_knn_wz --tag knn
  $PY -m scripts.run_official_protocol --subset $S --arms raw_knn,raw_knn_wz --crossfit-train --tag knncf
  $PY -m scripts.run_official_protocol --subset $S --arms raw_flatten --score-stride 1 --tag stride1
done
# Sec. 8: the baselines on the 472 held-out series (exploratory)
$PY -m scripts.run_official_protocol --subset $U --split rest --arms raw_knn,raw_knn_wz,raw_flatten --tag knn_rest
# Sec. 6.1: H1 probe, H2 (lambda chosen on the tuning split: 3 univariate, 10 multivariate), factorial
for S in $U $M; do
  $PY -m scripts.run_official_protocol --subset $S --arms raw_amp --tag amp
  $PY -m scripts.run_official_protocol --subset $S --split tuning --arms raw_knn,raw_knn_wz,raw_amp --tag tune_knn
  for l in 1 3 10 30; do
    $PY -m scripts.run_official_protocol --subset $S --split tuning --arms raw_aug --aug-lambda $l --tag tune_aug$l
  done
done
$PY -m scripts.run_official_protocol --subset $U --arms raw_aug --aug-lambda 3 --tag aug
$PY -m scripts.run_official_protocol --subset $M --arms raw_aug --aug-lambda 10 --tag aug
$PY -m scripts.run_normalization_ablation
$PY -m scripts.run_boundary_synthetic          # registered B1-B4
$PY -m scripts.run_boundary_followup           # registered F1-F2; F3 aggregates B1-B4
# Sec. 7: drift, H5 and the threshold studies (CPU)
$PY -m scripts.run_drift_injection
$PY -m scripts.run_drift_far
$PY -m scripts.run_drift_threshold
$PY -m scripts.run_drift_threshold2
$PY -m scripts.run_threshold_sub --list-eligible
$PY -m scripts.run_threshold_sub               # registered T1-T2 (Sub- detectors)
$PY -m scripts.threshold_sub_boundary_check     # exploratory, after T1's result
# EXTENDED_RESULTS.md: cost (prepared windows, CPU)
$PY -m scripts.bench_official_cost --scaling --no-vision
$PY -m scripts.bench_scaling
# Sec. 4.1/8: constant-score VUS-PR, table history and behavioural checks of three TSB-AD code states
$PY -m scripts.constant_score_vus && $PY -m scripts.constant_score_vus --merge
for SP in rest uad_primary; do                 # chance floor for the held-out and H11 series
  $PY -m scripts.constant_score_vus --split $SP && $PY -m scripts.constant_score_vus --split $SP --merge
done
$PY -m scripts.compare_published_tables
$PY -m scripts.series_features                 # cached anomaly length and train-test range
for pair in "A:$TSBAD_SRC_UPLOAD" "B:$TSBAD_SRC_PINNED" "C:$TSBAD_SRC_HEAD"; do
  OMP_NUM_THREADS=1 PYTHONPATH=${pair#*:} $PY -m scripts.check_preprocessing --tsbad-src "${pair#*:}" --tag "${pair%%:*}"
done
$PY -m scripts.check_preprocessing --summarize A B C
# Robustness to TSB-AD's fixed seeds: Sub-IForest and Sub-MCD with two other seeds
for sd in 1 2; do
  for SP in eva rest; do
    $PY -m scripts.run_detector_normalization --subset $U --split $SP --detectors Sub_IForest,Sub_MCD --seed $sd
  done
done
}

# Rendered pipeline from earlier work (GPU; not in the paper)
legacy() {
for S in $U $M; do
  $PY -m scripts.run_official_protocol --subset $S --arms vision --tag vision
  $PY -m scripts.run_official_protocol --subset $S --arms vision --crossfit-train --tag visioncf
  $PY -m scripts.run_official_protocol --subset $S --arms random_vision --tag randvision
  $PY -m scripts.run_official_protocol --subset $S --arms vision_fixed --tag vfixed
  $PY -m scripts.run_official_protocol --subset $S --arms vision_fixed --crossfit-train --tag vfixedcf
  for m in 0 0.25 0.5 1.0; do
    $PY -m scripts.run_official_protocol --subset $S --split tuning \
      --arms vision_fixed,raw_flatten,vision --range-margin $m --tag tune_m$m
  done
done
$PY -m scripts.bench_official_cost --scaling   # adds the rendered pipeline's cost
}

[[ "$MODE" == legacy ]] || core
[[ "$MODE" == core ]] || legacy
$PY -m scripts.collect_official_artifacts --replace
# Caches computed once from the raw series: drop them so that they are recomputed here.
rm -f artifacts/official_protocol/z_ratio.json artifacts/official_protocol/z_ratio_rest.json \
      artifacts/official_protocol/train_contamination.json
regenerate write
