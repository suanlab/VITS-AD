#!/usr/bin/env bash
# Reproduce the ICDE paper on window normalization in TSB-AD.
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
# produced. They stop if a listed series is missing and if an existing output was started with a
# different configuration (see scripts/run_contract.py). Families that a mode does not re-run keep
# their released files; collect_official_artifacts lists them in collect_report.json.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

regenerate() {
  $PY -m scripts.fetch_tsbad_official
  $PY -m scripts.summarize_official --variant default
  $PY -m scripts.summarize_official --variant crossfit
  $PY -m scripts.paper_numbers
  $PY -m scripts.make_paper_tables --out paper/icde2027/tables
  $PY -m scripts.make_normalization_figures --out paper/icde2027/figures
  $PY -m scripts.make_release_manifest
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
# Sec. V: TSB-AD's detectors with per-window vs. global normalization (CPU)
$PY -m scripts.run_detector_normalization --subset $U
$PY -m scripts.run_detector_normalization --subset $M --detectors KNN,LOF,HBOS,OCSVM
$PY -m scripts.run_detector_normalization --subset $M --detectors MCD,IForest,PCA --part _e5
# Sec. V: two more per-window families, Matrix Profile and KMeansAD (registered H10)
$PY -m scripts.run_family_toggle --split eva
$PY -m scripts.run_family_toggle --split rest
$PY -m scripts.run_detector_normalization --subset $U --ocsvm-clean --detectors Sub_OCSVM
$PY -m scripts.run_detector_normalization --subset $M --ocsvm-clean --detectors OCSVM
# Sec. V, held-out replications: H6 (tuning split) and H9 (472 series in neither split);
# near-identical series across splits (raw values only), used for H9's secondary analysis
$PY -m scripts.find_near_duplicates
$PY -m scripts.run_detector_normalization --subset $U --split tuning \
  --detectors Sub_KNN,Sub_MCD,Sub_OCSVM,Sub_HBOS,Sub_IForest,Sub_LOF
$PY -m scripts.run_detector_normalization --subset $M --split tuning
$PY -m scripts.run_detector_normalization --subset $U --split rest \
  --detectors Sub_KNN,Sub_MCD,Sub_OCSVM,Sub_HBOS,Sub_IForest,Sub_LOF
$PY -m scripts.run_detector_normalization --subset $U --split rest --ocsvm-clean --detectors Sub_OCSVM
# Sec. V, TSB-AD's code run unmodified: the commit of the original analysis and the current HEAD
PYTHONPATH=$TSBAD_SRC_PINNED $PY -m scripts.run_tsbad_pinned_detectors --tsbad-src "$TSBAD_SRC_PINNED"
PYTHONPATH=$TSBAD_SRC_HEAD $PY -m scripts.run_tsbad_pinned_detectors --tsbad-src "$TSBAD_SRC_HEAD" \
  --subset $U --detectors Sub_OCSVM --tag head
PYTHONPATH=$TSBAD_SRC_HEAD $PY -m scripts.run_tsbad_pinned_detectors --tsbad-src "$TSBAD_SRC_HEAD" \
  --subset $M --tag head
# Sec. VI: training-free baselines, default and cross-fitted, and the stride-1 check (CPU)
for S in $U $M; do
  $PY -m scripts.run_official_protocol --subset $S --arms raw_meanpool,raw_flatten --tag raw
  $PY -m scripts.run_official_protocol --subset $S --arms raw_flatten --crossfit-train --tag rawcf
  $PY -m scripts.run_official_protocol --subset $S --arms raw_knn,raw_knn_wz --tag knn
  $PY -m scripts.run_official_protocol --subset $S --arms raw_knn,raw_knn_wz --crossfit-train --tag knncf
  $PY -m scripts.run_official_protocol --subset $S --arms raw_flatten --score-stride 1 --tag stride1
done
# Sec. VII: H1 probe, H2 (lambda chosen on the tuning split: 3 univariate, 10 multivariate), factorial
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
# Sec. VIII: drift, H5 and the threshold studies (CPU)
$PY -m scripts.run_drift_injection
$PY -m scripts.run_drift_far
$PY -m scripts.run_drift_threshold
$PY -m scripts.run_drift_threshold2
$PY -m scripts.run_threshold_sub --list-eligible
$PY -m scripts.run_threshold_sub               # registered T1-T2 (Sub- detectors)
# Sec. VI-A: cost (prepared windows, CPU)
$PY -m scripts.bench_official_cost --scaling --no-vision
$PY -m scripts.bench_scaling
# Sec. IV/X: constant-score VUS-PR, table history and behavioural checks of three TSB-AD code states
$PY -m scripts.constant_score_vus && $PY -m scripts.constant_score_vus --merge
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
regenerate
