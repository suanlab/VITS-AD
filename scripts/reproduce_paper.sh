#!/usr/bin/env bash
# Reproduce the ICDE paper on window normalization in TSB-AD.
#
#   bash scripts/reproduce_paper.sh          # GPU-free: regenerate every number and figure
#                                            # from artifacts/official_protocol/
#   bash scripts/reproduce_paper.sh full     # re-run every experiment (needs the TSB-AD series
#                                            # in data/tsb_ad/, see README; GPU for vision arms)
#
# The full mode runs each script unsharded; every script accepts --shard k --num-shards K to
# split the work across processes, which is how the released results were produced.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}

regenerate() {
  $PY -m scripts.fetch_tsbad_official
  $PY -m scripts.summarize_official --variant default
  $PY -m scripts.summarize_official --variant crossfit
  $PY -m scripts.paper_numbers
  $PY -m scripts.make_normalization_figures --out figures
}

if [[ "${1:-}" != "full" ]]; then
  regenerate
  exit 0
fi

U=TSB-AD-U; M=TSB-AD-M
# Sec. IV: TSB-AD's detectors under per-window vs. global normalization (CPU)
$PY -m scripts.run_detector_normalization --subset $U
$PY -m scripts.run_detector_normalization --subset $M
# Sec. IV-A: TSB-AD's detectors unmodified at the pinned commit (CPU)
TSBAD_SRC=${TSBAD_SRC:?set TSBAD_SRC to an unpacked TSB-AD checkout at 6beac72e}
PYTHONPATH=$TSBAD_SRC $PY -m scripts.run_tsbad_pinned_detectors --tsbad-src "$TSBAD_SRC"
# Sec. V: training-free baselines, default and cross-fitted, and the stride-1 check (CPU)
for S in $U $M; do
  $PY -m scripts.run_official_protocol --subset $S --arms raw_meanpool,raw_flatten --tag raw
  $PY -m scripts.run_official_protocol --subset $S --arms raw_flatten --crossfit-train --tag rawcf
  $PY -m scripts.run_official_protocol --subset $S --arms raw_knn,raw_knn_wz --tag knn
  $PY -m scripts.run_official_protocol --subset $S --arms raw_knn,raw_knn_wz --crossfit-train --tag knncf
  $PY -m scripts.run_official_protocol --subset $S --arms raw_flatten --score-stride 1 --tag stride1
done
# Sec. VI: factorial (CPU)
$PY -m scripts.run_normalization_ablation
# Sec. VII: rendered pipeline (GPU)
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
# Sec. VIII and IX: cost (GPU) and drift injection (CPU)
$PY -m scripts.bench_official_cost --scaling
$PY -m scripts.run_drift_injection
$PY -m scripts.collect_official_artifacts
regenerate
