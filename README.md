# Window normalization on TSB-AD

Code and per-series results for *When Do Benchmark Conclusions Hold? Preprocessing, Source
Composition and Code Versions in Time-Series Anomaly Detection* (Suan Lee; submitted
to PVLDB Vol. 20 as an Experiment, Analysis & Benchmark paper). The release that matches the
submitted manuscript is the tag `pvldb-vol20-r4`. Earlier tags are kept unchanged:
`pvldb-vol20-submission` (d421157) is the state before the November registrations (TR, SP, RT, M, P)
and does not contain their results; `pvldb-vol20-r2` (791c74c) and `pvldb-vol20-r3` (24c54a5) are
earlier release candidates (r3 has the contract records' `fit_score` field with the inverted meaning
fixed in r4). Registered analyses whose runs are not finished are listed under
`registered_analyses_pending` in `artifacts/RELEASE_MANIFEST.json`; their keys in `paper_numbers.json`
then hold only a status. The paper was prepared for, but never submitted to, ICDE 2027; the tag
`icde2027-submission` and the ICDE section numbers in some registration files date from that plan. Section, table and figure numbers
below follow the PVLDB manuscript. The study
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
same numbers: the fragments in `paper/pvldb2027/tables/` and all figures are generated, while
Tables 1 and 2 and the numbers in the text are written by hand from `paper_numbers.json` and
`replication_prediction.json`. Material the paper refers to as being in the artifact is in
`artifacts/EXTENDED_RESULTS.md`, generated from `paper_numbers.json`.

```bash
pip install -r requirements-regen.txt     # enough to regenerate tables and figures (Python 3.12)
# to re-run experiments, use the exact environment of the released runs (TSB-AD 1.5, scikit-learn,
# torch and their versions as recorded in the run fingerprints):
pip install -r requirements-paper-lock.txt && pip install -e . --no-deps
# (pip install -e .[full] also works but resolves newer library versions)
bash scripts/reproduce_paper.sh           # runs the steps below
python -m scripts.fetch_tsbad_official    # TSB-AD tables, hash-checked
python -m scripts.summarize_official --variant default
python -m scripts.summarize_official --variant crossfit
python -m scripts.paper_numbers           # -> artifacts/official_protocol/paper_numbers.json
python -m scripts.make_paper_tables --out paper/pvldb2027/tables
python -m scripts.make_normalization_figures --out paper/pvldb2027/figures
python -m scripts.make_extended_results     # -> artifacts/EXTENDED_RESULTS.md
python -m scripts.make_contract_records     # -> artifacts/preprocessing_contract/ (schema + TSB-AD records)
python -m scripts.make_release_manifest --verify   # regen mode: compare with the released manifest
python -m scripts.make_release_manifest     # experiment modes: write a new manifest
```

`paper/pvldb2027/tables/` and `paper/pvldb2027/figures/` are the files the manuscript includes. Three generated
files are not used by the current manuscript: `figures/fig_toggle.pdf`, `tables/factorial_rows.tex` and
`tables/scaling_rows.tex` (their content is in `artifacts/EXTENDED_RESULTS.md`). Bootstrap
intervals (10,000 resamples, seed 0) are reproduced exactly with `requirements-paper-lock.txt`; other
library versions can order series differently and move interval endpoints by about 0.001. `artifacts/RELEASE_MANIFEST.json`
records, for the commit it was written at, every family of per-series result files with its
expected and present series, the series with a result per detector or arm and, for
normalization toggles, the series where both variants succeeded; which `paper_numbers.json` keys
each table and figure uses; and the SHA-256 of every artifact file and of the tracked code, tests,
configuration, environment files and this README (`scripts/`, `src/`, `tests/`, `configs/`,
`paper/`, `requirements*.txt`, `pyproject.toml`, `LICENSE`). The manuscript source itself is not in
the repository. `--verify` reports every file that differs from, or is missing in, the manifest, and
`bash scripts/reproduce_paper.sh` (regen mode) then exits non-zero.

| Paper element | Key in `paper_numbers.json` | Per-series inputs (`artifacts/official_protocol/`) | Produced by |
|---|---|---|---|
| Sec. 4.1, Table 2: normalization audit; published IForest/LOF vs. a constant score; history of the published tables | `constant_score` | `../normalization_audit/norm_audit.csv`, `constant_score.json`, `published_table_history.json` | code reading, see `norm_audit.md`; `constant_score_vus.py`; `compare_published_tables.py` |
| Sec. 8: preprocessing contracts as records (JSON Schema, one record per TSB-AD entry and audited commit) | — | `../normalization_audit/norm_audit.csv` | `make_contract_records.py` -> `artifacts/preprocessing_contract/` |
| Sec. 8: behavioural preprocessing checks on three TSB-AD commits (contract checks: fit/score match, dataset-class normalization, batch context; degeneracy; capability; fingerprints) | — | `../normalization_audit/checks_{A,B,C}.json`, `checks_summary.md` | `check_preprocessing.py` (CPU, minutes; needs the TSB-AD source of each commit) |
| Sec. 4.2, Table 3: per-window vs. global, TSB-AD's Sub- detectors | `toggle`, `other_measures`, `ocsvm_clean` | `TSB-AD-{U,M}/detnorm*_shard*.json` | `run_detector_normalization.py` (CPU) |
| Sec. 4.2: reproduction detail, determinism, failure policy, source-level resampling, leave-one-source-out, 64-setting rank sensitivity; Sec. 6.1 within-source H1 | `robustness`, `repeat_determinism`, `failure_policy`, `rank_sensitivity`, `seed_robustness` | the above + `near_duplicates.json` | `paper_numbers.py`, `find_near_duplicates.py` |
| Sec. 3, Table 1, Sec. 5.1-5.2, Table 4, Fig. 2: registered hypotheses, held-out replications H6, H9, source decomposition, anomaly length and drift | `held_out`, `held_out_472`, `held_out_472_regime`, `source_decomposition`, `h9_explanation` (needs `series_features.json`) | `tuning/*/detnorm_tuning_*`, `TSB-AD-U/detnorm_rest*` | `run_detector_normalization.py --split tuning/rest` |
| Sec. 4.2, 5.1: Matrix Profile and KMeansAD toggle (registered H10) | `family_toggle` | `TSB-AD-U/family{,_rest}_shard*.json` | `run_family_toggle.py` (CPU; hours for the longest series) |
| Sec. 4.4: TSB-AD's code run unmodified | `pinned`, `pinned_head` | `TSB-AD-{U,M}/{pinned,head}_shard*.json` | `run_tsbad_pinned_detectors.py` (CPU) |
| Sec. 5.1, `EXTENDED_RESULTS.md`: held-out results under all four measures, win counts, chance floor (constant-score VUS-PR) | `other_measures_held_out`, `chance_floor` | `constant_score_{rest,uad_primary}.json` and the per-series files above | `paper_numbers.py`, `constant_score_vus.py --split rest/uad_primary` |
| Sec. 5.2: split composition over all 870 TSB-AD-U series, random-effects model over sources (between-source SD, I², prediction interval), random 350-series re-splits (exploratory) | `composition` | `detnorm{,_tuning,_rest}_shard*.json` | `paper_numbers.py` |
| Sec. 4.3, 8: composition range, flip radius (share of a split's weight that must move to other sources to cancel a gain; also for Sub-KNN's first rank) and systematic error terms per entry (exploratory); the tool works on any per-item results with a source column (`python -m scripts.composition_sensitivity results.csv --a A --b B`) | `composition_sensitivity` | `detnorm{,_tuning,_rest}_shard*.json`, `pinned_shard*`, `head_shard*` | `paper_numbers.py`, `composition_sensitivity.py` |
| Sec. 4.3, Fig. 1: census of TSB-AD's published univariate leaderboard (flip radius and Holm-significance of every method pair; exploratory, no detector run) and its controls (sources with at least five series, permutation null over source labels, split-composition distances) | `leaderboard_census`, `census_controls` | TSB-AD's published tables, `TSB-AD-U-Eva.csv` | `paper_numbers.py`, `make_normalization_figures.py` |
| Sec. 6.2: the toggle by anomaly length over each detector's window, and on series where TSB-AD's window estimator falls back to 125 (exploratory) | `window_estimation` | `series_features.json` | `series_features.py`, `paper_numbers.py` |
| Sec. 5.2: the toggle by anomaly type (point, sequence, both; exploratory) | `anomaly_type` | `series_features.json` (cached from the raw labels) | `series_features.py`, `paper_numbers.py` |
| Sec. 5.2, Table 1: TR, the toggle on the 102 YAHOO series of TSB-UAD that TSB-AD left out, against a prediction from TSB-AD's YAHOO by Yahoo S5 subset (registered values in `transport_hypothesis.md`) | `transport` | `TSB-AD-U/detnorm_uad_shared_shard*.json`, `tsbuad_shared_match.json`, `tsbuad_shared_listing.csv` | `match_tsbuad_shared.py`, `convert_tsbuad_shared.py`, `run_detector_normalization.py --split uad_shared` |
| Sec. 5.2, Table 1: SP, level, scale and shape anomalies injected into real YAHOO and other backgrounds | `spikein` | `synthetic/spikein_shard*.json`, `spikein_plan.json` | `run_spikein.py plan`, `run_spikein.py run` |
| Sec. 5.2: SP's detector windows (TSB-AD's wrappers re-estimate the window on the injected series) and the fixed-window sensitivity run (exploratory) | `spikein.detector_windows`, `spikein.fixed_window_exploratory` | `spikein_windows.json`, `synthetic/spikein_fixedw_shard*.json` | `run_spikein.py run --fixed-w` |
| Sec. 4.2: ranking among the eight detectors run in one environment, over all 256 settings (exploratory) | `same_environment_ranking` | `detnorm_shard*.json`, `family_shard*.json` | `paper_numbers.py` |
| Sec. 4.2, 5.1, Tables 1, 3, 4: RT, both normalization variants re-tuned on the tuning split with TSB-AD's own grid, then run on the evaluation split and the 472 held-out series | `retune` | `TSB-AD-U/retune_grid_shard*.json`, `retune_selection.json`, `retune_{eva,rest}_shard*.json` (configurations equal to TSB-AD's reuse `detnorm{,_rest}_shard*.json`) | `run_retune.py grid`, `select`, `apply` |
| Sec. 5.3, Tables 1 and 5, Fig. 3: M, registered prediction of which pairwise conclusions replicate on new sources (RT on the evaluation split and the 472 held-out series; TimeEval's published results by collection) | `replication_prediction.json` (written by the script, not by paper_numbers) | RT files above; TimeEval's published table (not redistributed: download from the TimeEval authors' share link https://my.hidrive.com/api/sharelink/download?id=HwCGA82U, SHA-256 49961990bda2450b6b1ec9a5632d3a29cdd43ce55c02192b67d2fe8c5f5d8288) | `replication_prediction.py --timeeval <csv>` (outside regen mode; save the table as `data/timeeval_results/timeeval_results.csv`; `reproduce_paper.sh core` checks its SHA-256; run with one BLAS thread, `OPENBLAS_NUM_THREADS=1`, for byte-identical output) |
| Sec. 5.3: exploratory follow-ups to M (centre versus variance of p_new, t quantiles, scores per held-out collection; not registered, after review round 18) | `replication_prediction_explore.json` (written by the script) | as for M | `replication_prediction_explore.py --timeeval <csv>` |
| Sec. 5.4, Table 1: P, the toggle on TimeEval's GutenTAG collection (168 univariate synthetic series, a collection outside the TSB-AD family) against predictions fixed before any run (`gutentag_hypothesis.md`, `gutentag_predictions.json`) | `gutentag` | `TSB-AD-U/detnorm_gutentag_shard*.json`, `gutentag_listing.csv` | `convert_gutentag.py` (archive from the TimeEval authors' share link https://my.hidrive.com/api/sharelink/download?id=j2imAkqU, served as `GutenTAG.zip`; save it as `data/timeeval_gutentag/raw/gutentag.zip`; SHA-256 3f3c65772adc20b06eb3abb78a8346757bc3031cba2186188dd4e51197c94d51, MIT licence; not redistributed), `gutentag_prediction.py`, `run_detector_normalization.py --split gutentag` |
| Sec. 5.2, Table 1: H11, the toggle on TSB-UAD sources outside TSB-AD | `tsbuad` | `TSB-AD-U/detnorm_uad_{primary,secondary}_shard*.json`, `per_series_knn_uad_*`, `tsbuad_listing.csv` | `convert_tsbuad.py`, `run_detector_normalization.py --split uad_primary/uad_secondary` |
| Sec. 8, Table 9: training-free baselines, cross-fitting, stride 1, held-out series | `baselines`, `stride_both`, `baselines_held_out` | `per_series_{knn,knncf,raw,rawcf,stride1,knn_rest}_shard*.json` | `run_official_protocol.py` (CPU) |
| Sec. 6.2, Table 7: registered synthetic test B1-B4 (edges vs. interior) | `boundary_synthetic` | `synthetic/boundary.json` | `run_boundary_synthetic.py` (CPU, about 1.5 h) |
| Sec. 6.1: H1, H2 | `preregistered` | `per_series_{amp,aug}_shard*.json`, `tuning/` | `run_official_protocol.py --arms raw_amp / raw_aug` |
| Sec. 6.1, `EXTENDED_RESULTS.md`: factorial | `factorial` | `TSB-AD-U/ablation_shard*.json` | `run_normalization_ablation.py` (CPU) |
| Sec. 6.1, Table 6: amplitude terciles | `tercile_sub_detectors`, `regime` | the above + `z_ratio.json` | `paper_numbers.py` |
| Sec. 7, Table 8: drift, thresholds (H5), equal-FAR operating points | `drift`, `far`, `threshold2` (Table 8); `threshold` is an earlier calibration not used in the paper | `TSB-AD-U/{drift,far,threshold,threshold2}_shard*.json` | `run_drift_injection.py`, `run_drift_far.py`, `run_drift_threshold.py`, `run_drift_threshold2.py` |
| Sec. 7, Table 1: TSB-AD's Sub-KNN/IForest/HBOS at a threshold (registered T1-T2; offline, transductive) | `threshold_sub` | `TSB-AD-U/threshold_sub_shard*.json`, `threshold_sub_expected.json`; post-result check `threshold_sub_boundary.json` | `run_threshold_sub.py`, `threshold_sub_boundary_check.py` |
| `EXTENDED_RESULTS.md`: cost and scalability (not in the paper) | `cost`, `scaling` | `cost.json`, `scaling.json` | `bench_official_cost.py`, `bench_scaling.py` |
| (not in the paper) rendered pipeline from earlier work | `render`, `render_lift` | `per_series_{vision,randvision,vfixed}_shard*.json`, `cost.json` | `run_official_protocol.py` (GPU), `bench_official_cost.py` |
| Sec. 9: contaminated training parts | `contamination` | `train_contamination.json` (cached flags) | `paper_numbers.py` |

`z_ratio.json` and `train_contamination.json` cache values computed once from the raw series, so the
regeneration above does not need them.

### Registered hypotheses
`artifacts/preregistration/` holds the hypotheses with their decision rules
(`normalization_hypotheses.md`: H1, H2, H4, H5, H6, H8; `heldout472_hypothesis.md`: H9; `boundary_hypothesis.md`: B1-B4; `boundary_followup.md`: F1-F3; `family_hypothesis.md`: H10 and its KMeans-seed amendment;
`threshold_sub_hypothesis.md`: T1-T2; `tsbuad_hypothesis.md`: H11; `retune_hypothesis.md`: RT1-RT3; `transport_hypothesis.md`: TR1-TR2; `spikein_hypothesis.md`: SP1-SP3; `replication_prediction.md`: M; `gutentag_hypothesis.md`: P1-P3), committed before the corresponding analyses; an analysis
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
These checks were added on 2026-10-29: the KMeans-seed, held-out baseline and H11 outputs, and the
T1-T2 continuation files written after that date (4 of 21), carry fingerprints; files produced earlier do not. Failed detector runs are kept as `null` with
their exception under `errors`; earlier outputs record only the `null`. Families a mode does not re-run keep their released files; after an experiment mode,
`artifacts/official_protocol/collect_report.json` lists them (the file does not exist in a fresh checkout). Runs write to
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
