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
and from TSB-AD's published per-series tables, which are downloaded from a pinned TSB-AD commit and
checked against SHA-256 hashes.

```bash
pip install -e .[full]                               # Python >= 3.10; includes TSB-AD==1.5
bash scripts/reproduce_paper.sh                      # = the four steps below
python -m scripts.fetch_tsbad_official               # TSB-AD tables @ 6beac72e, hash-checked
python -m scripts.summarize_official --variant default
python -m scripts.summarize_official --variant crossfit
python -m scripts.paper_numbers                      # -> artifacts/official_protocol/paper_numbers.json
python -m scripts.make_normalization_figures --out figures
```

`paper_numbers.json` holds every reported number, by paper section:

| Paper element | Key in `paper_numbers.json` | Per-series inputs (`artifacts/official_protocol/`) | Produced by |
|---|---|---|---|
| Sec. IV, Table I: TSB-AD detectors, per-window vs. global | `toggle` | `TSB-AD-{U,M}/detnorm_shard*.json` | `run_detector_normalization.py` (CPU) |
| Sec. IV-A: TSB-AD's current code | `pinned` | `TSB-AD-U/pinned_shard*.json` | `run_tsbad_pinned_detectors.py` (CPU) |
| Sec. V, Table II: training-free baselines | `baselines` | `per_series_{knn,knncf,raw,rawcf}_shard*.json` | `run_official_protocol.py` (CPU) |
| Sec. V: stride-1 check | `stride` | `per_series_stride1_shard*.json` | `run_official_protocol.py --score-stride 1` |
| Sec. VI-A, Table III, Fig. 1: factorial | `factorial` | `TSB-AD-U/ablation_shard*.json` | `run_normalization_ablation.py` (CPU) |
| Sec. VI-B: amplitude terciles | `regime` | the above + `z_ratio.json` | `paper_numbers.py` |
| Sec. VII, Table IV: rendered pipeline | `render` | `per_series_{vision,randvision,vfixed,vfixedcf}_shard*.json`, `tuning/` | `run_official_protocol.py` (GPU) |
| Sec. VIII: cost | `cost` | `cost.json` | `bench_official_cost.py` (GPU) |
| Sec. IX: drift injection | `drift` | `TSB-AD-U/drift_shard*.json` | `run_drift_injection.py` (CPU) |

`z_ratio.json` caches the label-derived amplitude ratio of every series, so the regeneration above
does not need the raw series.

## Re-running the experiments

The raw TSB-AD series come from TSB-AD's own release (links from its README):

| File | URL | SHA-256 of the copy we used |
|---|---|---|
| TSB-AD-U.zip | https://www.thedatum.org/datasets/TSB-AD-U.zip | `0c47020d3423723c70773736dbd800369f2b487328becbf339450d1ae5020961` |
| TSB-AD-M.zip | https://www.thedatum.org/datasets/TSB-AD-M.zip | `7de86ac27f30eeb48d833bb061055670e3f3de07defd995cf2bd5db10ccc9a0d` |

Unzip them so that the series lie in `data/tsb_ad/TSB-AD-U/*.csv` and `data/tsb_ad/TSB-AD-M/*.csv`.
`bash scripts/reproduce_paper.sh full` lists and runs every command that produced the per-series
results (sharding flags omitted; every script accepts `--shard k --num-shards K`). Runs write to
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
