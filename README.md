# Window normalization on TSB-AD

Code and per-series results for *One Preprocessing Step Reorders a Benchmark---and Fails to
Replicate: Window Normalization in Time-Series Anomaly Detection on TSB-AD* (Suan Lee; under review). The study
re-runs the subsequence detectors of the [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) benchmark
under per-window and global normalization, compares training-free raw-window baselines with
TSB-AD's published results, and analyses why normalization matters.

The repository name and the `src/` package come from the project's earlier work on rendered,
frozen-vision detectors (see *Legacy* below). That pipeline is no longer part of the paper; its
results stay under `artifacts/official_protocol/` (`render`, `render_lift` in `paper_numbers.json`)
for provenance.

## Reproducing the paper without a GPU

Every number in the paper is computed from per-series results under `artifacts/official_protocol/`
and from TSB-AD's published per-series tables, downloaded from a pinned TSB-AD commit
(`303dffbf`) and checked against SHA-256 hashes. The tables of the paper are generated from the
same numbers: the fragments in `paper/icde2027/tables/` and both figures are generated, while
Tables I and II and the numbers in the text are written by hand from `paper_numbers.json`.

```bash
pip install -r requirements-regen.txt     # enough to regenerate tables and figures (Python 3.12)
pip install -e .[full]                    # to re-run experiments; includes TSB-AD==1.5
# exact environment used: pip install -r requirements-paper-lock.txt && pip install -e . --no-deps
bash scripts/reproduce_paper.sh           # runs the steps below
python -m scripts.fetch_tsbad_official    # TSB-AD tables, hash-checked
python -m scripts.summarize_official --variant default
python -m scripts.summarize_official --variant crossfit
python -m scripts.paper_numbers           # -> artifacts/official_protocol/paper_numbers.json
python -m scripts.make_paper_tables --out paper/icde2027/tables
python -m scripts.make_normalization_figures --out paper/icde2027/figures
python -m scripts.make_release_manifest     # -> artifacts/RELEASE_MANIFEST.json
python -m scripts.make_release_manifest --verify   # check a checkout against the manifest
```

`paper/icde2027/tables/` and `paper/icde2027/figures/` are the files the manuscript includes. Bootstrap
intervals (10,000 resamples, seed 0) are reproduced exactly with `requirements-paper-lock.txt`; other
library versions can order series differently and move interval endpoints by about 0.001. `artifacts/RELEASE_MANIFEST.json`
records, for the commit it was written at, every family of per-series result files with its
expected and present series, the series with a result per detector or arm and, for
normalization toggles, the series where both variants succeeded; which `paper_numbers.json` keys
each table and figure uses; and the SHA-256 of every artifact file and of the tracked code, tests,
configuration, environment files and this README (`scripts/`, `src/`, `tests/`, `configs/`,
`paper/`, `requirements*.txt`, `pyproject.toml`, `LICENSE`). The manuscript source itself is not in
the repository. `--verify` reports every file that differs from, or is missing in, the manifest.

| Paper element | Key in `paper_numbers.json` | Per-series inputs (`artifacts/official_protocol/`) | Produced by |
|---|---|---|---|
| Sec. IV, Table II: normalization audit; published IForest/LOF vs. a constant score; history of the published tables | `constant_score` | `../normalization_audit/norm_audit.csv`, `constant_score.json`, `published_table_history.json` | code reading, see `norm_audit.md`; `constant_score_vus.py`; `compare_published_tables.py` |
| Sec. X: behavioural preprocessing checks on three TSB-AD commits | — | `../normalization_audit/checks_{A,B,C}.json`, `checks_summary.md` | `check_preprocessing.py` (CPU, minutes; needs the TSB-AD source of each commit) |
| Sec. V, Table III, Fig. 1: per-window vs. global, TSB-AD's Sub- detectors | `toggle`, `other_measures`, `ocsvm_clean` | `TSB-AD-{U,M}/detnorm*_shard*.json` | `run_detector_normalization.py` (CPU) |
| Sec. V: reproduction detail, determinism, failure policy, source-level resampling, leave-one-source-out, 64-setting rank sensitivity; Sec. VII-A within-source H1 | `robustness`, `repeat_determinism`, `failure_policy`, `rank_sensitivity`, `seed_robustness` | the above + `near_duplicates.json` | `paper_numbers.py`, `find_near_duplicates.py` |
| Sec. III, Table I, Sec. V, Table IV, Fig. 2: registered hypotheses, held-out replications H6, H9, source decomposition, anomaly length and drift | `held_out`, `held_out_472`, `held_out_472_regime`, `source_decomposition`, `h9_explanation` (needs `series_features.json`) | `tuning/*/detnorm_tuning_*`, `TSB-AD-U/detnorm_rest*` | `run_detector_normalization.py --split tuning/rest` |
| Sec. V: Matrix Profile and KMeansAD toggle (registered H10) | `family_toggle` | `TSB-AD-U/family{,_rest}_shard*.json` | `run_family_toggle.py` (CPU; hours for the longest series) |
| Sec. V: TSB-AD's code run unmodified | `pinned`, `pinned_head` | `TSB-AD-{U,M}/{pinned,head}_shard*.json` | `run_tsbad_pinned_detectors.py` (CPU) |
| Sec. V, Table I: H11, the toggle on TSB-UAD sources outside TSB-AD | `tsbuad` | `TSB-AD-U/detnorm_uad_{primary,secondary}_shard*.json`, `per_series_knn_uad_*`, `tsbuad_listing.csv` | `convert_tsbuad.py`, `run_detector_normalization.py --split uad_primary/uad_secondary` |
| Sec. VI, Table V: training-free baselines, cross-fitting, stride 1, held-out series | `baselines`, `stride_both`, `baselines_held_out` | `per_series_{knn,knncf,raw,rawcf,stride1,knn_rest}_shard*.json` | `run_official_protocol.py` (CPU) |
| Sec. VII-B, Table VI: registered synthetic test B1-B4 (edges vs. interior) | `boundary_synthetic` | `synthetic/boundary.json` | `run_boundary_synthetic.py` (CPU, about 1.5 h) |
| Sec. VII-A: H1, H2 | `preregistered` | `per_series_{amp,aug}_shard*.json`, `tuning/` | `run_official_protocol.py --arms raw_amp / raw_aug` |
| Sec. VII-C, Table VII: factorial | `factorial` | `TSB-AD-U/ablation_shard*.json` | `run_normalization_ablation.py` (CPU) |
| Sec. VII-D, Table VIII: amplitude terciles | `tercile_sub_detectors`, `regime` | the above + `z_ratio.json` | `paper_numbers.py` |
| Sec. VIII, Table IX: drift, thresholds (H5), equal-FAR operating points | `drift`, `far`, `threshold`, `threshold2` | `TSB-AD-U/{drift,far,threshold,threshold2}_shard*.json` | `run_drift_injection.py`, `run_drift_far.py`, `run_drift_threshold.py`, `run_drift_threshold2.py` |
| Sec. VIII, Table I: TSB-AD's Sub-KNN/IForest/HBOS at a threshold (registered T1-T2; offline, transductive) | `threshold_sub` | `TSB-AD-U/threshold_sub_shard*.json`, `threshold_sub_expected.json`; post-result check `threshold_sub_boundary.json` | `run_threshold_sub.py`, `threshold_sub_boundary_check.py` |
| Sec. VI-A: cost and scalability (the scaling grid is `paper/icde2027/tables/scaling_rows.tex`, not in the paper) | `cost`, `scaling` | `cost.json`, `scaling.json` | `bench_official_cost.py`, `bench_scaling.py` |
| (not in the paper) rendered pipeline from earlier work | `render`, `render_lift` | `per_series_{vision,randvision,vfixed}_shard*.json`, `cost.json` | `run_official_protocol.py` (GPU), `bench_official_cost.py` |
| Sec. IX: contaminated training parts | `contamination` | `train_contamination.json` (cached flags) | `paper_numbers.py` |

`z_ratio.json` and `train_contamination.json` cache values computed once from the raw series, so the
regeneration above does not need them.

### Registered hypotheses
`artifacts/preregistration/` holds the hypotheses with their decision rules
(`normalization_hypotheses.md`: H1, H2, H4, H5, H6, H8; `heldout472_hypothesis.md`: H9; `boundary_hypothesis.md`: B1-B4; `boundary_followup.md`: F1-F3; `family_hypothesis.md`: H10 and its KMeans-seed amendment;
`threshold_sub_hypothesis.md`: T1-T2; `tsbuad_hypothesis.md`: H11), committed before the corresponding analyses; an analysis
plan without a decision rule for the baselines on the held-out series (`raw_baseline_heldout_plan.md`,
written after a reviewer estimate was seen); `results.md` with every outcome; and `push_events/`,
the repository host's records of when each was pushed (`activity_all_until_20261031.json` holds every push since the repository was created). Some registration files mention internal planning notes and
simulated (mock) reviews that motivated a test; those notes are not part of the release. The commit history and the
repository host's push events give their timing.

## Re-running the experiments

The raw TSB-AD series come from TSB-AD's own release (links from its README):

| File | URL | SHA-256 of the copy we used |
|---|---|---|
| TSB-AD-U.zip | https://www.thedatum.org/datasets/TSB-AD-U.zip | `0c47020d3423723c70773736dbd800369f2b487328becbf339450d1ae5020961` |
| TSB-AD-M.zip | https://www.thedatum.org/datasets/TSB-AD-M.zip | `7de86ac27f30eeb48d833bb061055670e3f3de07defd995cf2bd5db10ccc9a0d` |

Unzip them so that the series lie in `data/tsb_ad/TSB-AD-U/*.csv` and `data/tsb_ad/TSB-AD-M/*.csv`.
For H11, download TSB-UAD's public series (https://www.thedatum.org/datasets/TSB-UAD-Public.zip,
SHA-256 `ff4aa83a5a111835d410d962152e8dbebcda1039b778bae45b6b9c3f46dd49a1`, Apache-2.0) to
`data/tsb_uad/`, unzip it into `data/tsb_uad/public/`, and run `python -m scripts.convert_tsbuad`.
The three TSB-AD code states that full mode needs can be fetched with

```bash
for c in 35ba9c3b 6beac72e 303dffbf; do
  mkdir -p tsbad_src/$c
  curl -sL https://github.com/TheDatumOrg/TSB-AD/archive/$c.tar.gz | tar -xz -C tsbad_src/$c --strip-components 1
done
export TSBAD_SRC_UPLOAD=$PWD/tsbad_src/35ba9c3b TSBAD_SRC_PINNED=$PWD/tsbad_src/6beac72e TSBAD_SRC_HEAD=$PWD/tsbad_src/303dffbf
```

The GPU-free regeneration needs network access once, to download TSB-AD's published tables
(`fetch_tsbad_official`, checked against SHA-256). The source datasets inside TSB-AD keep their own
licenses and terms of use (see TSB-AD's README); cite TSB-AD (Liu and Paparrizos, NeurIPS 2024 Datasets
and Benchmarks) when using them.
`bash scripts/reproduce_paper.sh core` lists and runs every CPU command that produced the per-series
results the paper reports, `legacy` the rendered-pipeline runs of earlier work (GPU, not in the
paper), and `full` both (set `TSBAD_SRC_UPLOAD`, `TSBAD_SRC_PINNED` and `TSBAD_SRC_HEAD` to unpacked
TSB-AD checkouts at `35ba9c3b`, `6beac72e` and `303dffbf`). Sharding flags are omitted there; the
per-series runners (`run_detector_normalization`, `run_family_toggle`, `run_official_protocol`,
`run_normalization_ablation`, `run_drift_threshold2`, `run_threshold_sub`) accept
`--shard k --num-shards K`, stop when a listed series is missing (unless `--allow-missing`), and
refuse to resume an output started with a different configuration (`<output>.fingerprint`, see
`scripts/run_contract.py`; it also records library versions and thread settings, and logs a change).
These checks were added on 2026-10-29: the T1-T2, KMeans-seed and held-out baseline outputs carry
fingerprints, the families produced earlier do not. Failed detector runs are kept as `null` with
their exception under `errors`; earlier outputs record only the `null`. Families a mode does not re-run keep their released files, listed in
`artifacts/official_protocol/collect_report.json`. Runs write to
`results/official_protocol/`; `python -m scripts.collect_official_artifacts` copies them into
`artifacts/`. The CPU experiments take roughly a day on 16 cores; the rendered-pipeline runs need
a GPU and about a day on four RTX 3090s. `run_tsbad_pinned_detectors.py` needs the TSB-AD source
at the pinned commit on `PYTHONPATH` (see its docstring).

## Tests

```bash
pytest tests/ -m "not slow" -q
```

## Legacy

Earlier work in this repository developed a rendered, frozen-vision detector (PatchTraj with
dual-signal scoring, Hydra configs under `configs/`) and an audit on a window-level protocol
(`artifacts/` subfolders other than `official_protocol/`). Those artifacts and scripts
(`scripts/reproduce_method_paper_legacy.sh`, `scripts/make_figures.py`, `scripts/train_patchtraj.py`
and others) are kept for provenance and are not used by the paper.

## License

MIT, see `LICENSE`. TSB-AD's data and published results are distributed by TSB-AD under its own
terms (Apache-2.0 for its code and curation); we redistribute neither.
