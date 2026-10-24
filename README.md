# Window normalization on TSB-AD

Code and per-series results for *One Line of Preprocessing Reorders a Benchmark: Window
Normalization in Time-Series Anomaly Detection on TSB-AD* (Suan Lee; under review). The study
re-runs the subsequence detectors of the [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) benchmark
under per-window and global normalization, compares training-free raw-window baselines with
TSB-AD's published results, and analyses why normalization matters.

The repository name and the `src/` package come from the project's earlier work on rendered,
frozen-vision detectors (see *Legacy* below); the rendered pipeline is the case study of the
paper's Section VII.

## Reproducing the paper without a GPU

Every number in the paper is computed from per-series results under `artifacts/official_protocol/`
and from TSB-AD's published per-series tables, downloaded from a pinned TSB-AD commit
(`303dffbf`) and checked against SHA-256 hashes. The tables of the paper are generated from the
same numbers, so they cannot drift from the artifacts.

```bash
pip install -e .[full]                    # Python >= 3.10; includes TSB-AD==1.5
# exact environment used: pip install -r requirements-paper-lock.txt && pip install -e . --no-deps
bash scripts/reproduce_paper.sh           # runs the steps below
python -m scripts.fetch_tsbad_official    # TSB-AD tables, hash-checked
python -m scripts.summarize_official --variant default
python -m scripts.summarize_official --variant crossfit
python -m scripts.paper_numbers           # -> artifacts/official_protocol/paper_numbers.json
python -m scripts.make_paper_tables --out paper/icde2027/tables
python -m scripts.make_normalization_figures --out paper/icde2027/figures
python -m scripts.make_release_manifest     # -> artifacts/RELEASE_MANIFEST.json
```

`paper/icde2027/tables/` and `paper/icde2027/figures/` are the files the manuscript includes. Bootstrap
intervals (10,000 resamples, seed 0) are reproduced exactly with `requirements-paper-lock.txt`; other
library versions can order series differently and move interval endpoints by about 0.001. `artifacts/RELEASE_MANIFEST.json`
records, for the commit it was written at, the expected, completed and failed series of every
experiment, which `paper_numbers.json` keys each table and figure uses, and the SHA-256 of every
artifact file.

| Paper element | Key in `paper_numbers.json` | Per-series inputs (`artifacts/official_protocol/`) | Produced by |
|---|---|---|---|
| Sec. IV, Table II: normalization audit; published IForest/LOF vs. a constant score; history of the published tables | `constant_score` | `../normalization_audit/norm_audit.csv`, `constant_score.json`, `published_table_history.json` | code reading, see `norm_audit.md`; `constant_score_vus.py`; `compare_published_tables.py` |
| Sec. XI: behavioural preprocessing checks on three TSB-AD commits | — | `../normalization_audit/checks_{A,B,C}.json`, `checks_summary.md` | `check_preprocessing.py` (CPU, minutes; needs the TSB-AD source of each commit) |
| Sec. V, Table III, Fig. 1: per-window vs. global, TSB-AD's Sub- detectors | `toggle`, `other_measures`, `ocsvm_clean` | `TSB-AD-{U,M}/detnorm*_shard*.json` | `run_detector_normalization.py` (CPU) |
| Sec. V: reproduction detail, determinism, failure policy, source-level resampling, leave-one-source-out, 64-setting rank sensitivity; Sec. VII-A within-source H1 | `robustness`, `repeat_determinism`, `failure_policy`, `rank_sensitivity`, `seed_robustness` | the above + `near_duplicates.json` | `paper_numbers.py`, `find_near_duplicates.py` |
| Sec. III, Table I, Sec. V, Table IV, Fig. 2: registered hypotheses, held-out replications H6, H9, source decomposition | `held_out`, `held_out_472`, `held_out_472_regime`, `source_decomposition` | `tuning/*/detnorm_tuning_*`, `TSB-AD-U/detnorm_rest*` | `run_detector_normalization.py --split tuning/rest` |
| Sec. V: TSB-AD's code run unmodified | `pinned`, `pinned_head` | `TSB-AD-{U,M}/{pinned,head}_shard*.json` | `run_tsbad_pinned_detectors.py` (CPU) |
| Sec. VI, Table V: training-free baselines, cross-fitting, stride 1 | `baselines`, `stride_both` | `per_series_{knn,knncf,raw,rawcf,stride1}_shard*.json` | `run_official_protocol.py` (CPU) |
| Sec. VII-B, Table VII: registered synthetic test B1-B4 (edges vs. interior) | `boundary_synthetic` | `synthetic/boundary.json` | `run_boundary_synthetic.py` (CPU, about 1.5 h) |
| Sec. VII-A: H1, H2 | `preregistered` | `per_series_{amp,aug}_shard*.json`, `tuning/` | `run_official_protocol.py --arms raw_amp / raw_aug` |
| Sec. VII-C, Table VIII, Fig. 3: factorial | `factorial` | `TSB-AD-U/ablation_shard*.json` | `run_normalization_ablation.py` (CPU) |
| Sec. VII-D, Table IX: amplitude terciles | `tercile_sub_detectors`, `regime` | the above + `z_ratio.json` | `paper_numbers.py` |
| Sec. VIII, Table X: drift, thresholds (H5), equal-FAR operating points | `drift`, `far`, `threshold`, `threshold2` | `TSB-AD-U/{drift,far,threshold,threshold2}_shard*.json` | `run_drift_injection.py`, `run_drift_far.py`, `run_drift_threshold.py`, `run_drift_threshold2.py` |
| Sec. VI-A, Table VI: cost and scalability | `cost`, `scaling` | `cost.json`, `scaling.json` | `bench_official_cost.py`, `bench_scaling.py` |
| Sec. IX: rendered pipeline, cost | `render`, `render_lift`, `cost` | `per_series_{vision,randvision,vfixed}_shard*.json`, `cost.json` | `run_official_protocol.py` (GPU), `bench_official_cost.py` |
| Sec. X: contaminated training parts | `contamination` | `train_contamination.json` (cached flags) | `paper_numbers.py` |

`z_ratio.json` and `train_contamination.json` cache values computed once from the raw series, so the
regeneration above does not need them.

### Registered hypotheses
`artifacts/preregistration/` holds the hypotheses with their decision rules
(`normalization_hypotheses.md`: H1, H2, H4, H5, H6, H8; `heldout472_hypothesis.md`: H9; `boundary_hypothesis.md`: B1-B4), committed
before the corresponding analyses, and `results.md` with every outcome. The commit history and the
repository host's push events give their timing.

## Re-running the experiments

The raw TSB-AD series come from TSB-AD's own release (links from its README):

| File | URL | SHA-256 of the copy we used |
|---|---|---|
| TSB-AD-U.zip | https://www.thedatum.org/datasets/TSB-AD-U.zip | `0c47020d3423723c70773736dbd800369f2b487328becbf339450d1ae5020961` |
| TSB-AD-M.zip | https://www.thedatum.org/datasets/TSB-AD-M.zip | `7de86ac27f30eeb48d833bb061055670e3f3de07defd995cf2bd5db10ccc9a0d` |

Unzip them so that the series lie in `data/tsb_ad/TSB-AD-U/*.csv` and `data/tsb_ad/TSB-AD-M/*.csv`.
`bash scripts/reproduce_paper.sh full` lists and runs every command that produced the per-series
results (set `TSBAD_SRC_UPLOAD`, `TSBAD_SRC_PINNED` and `TSBAD_SRC_HEAD` to unpacked TSB-AD checkouts at `35ba9c3b`, `6beac72e` and `303dffbf`) (sharding flags omitted; every script accepts `--shard k --num-shards K`). Runs write to
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
