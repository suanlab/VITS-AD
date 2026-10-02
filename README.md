# VITS — Vision-Informed Time Series Anomaly Detection

Companion repository for a benchmark-scale audit of frozen-vision time-series anomaly detection.
It contains the code, configurations and per-series result artifacts behind every number in the
paper. The paper is under review; publication details will be added here afterwards.

VITS renders sliding time-series windows as images, extracts patch tokens from
a frozen DINOv2 backbone, and scores anomalies via dual-signal fusion of
Mahalanobis distributional distance (primary) and trajectory prediction
residuals (regularizer).

## Installation

```bash
conda create -n vits python=3.10 -y && conda activate vits
pip install -e .[full]
```

The `full` extra installs the vision backbone (transformers, timm, Pillow) and
the dev tools (pytest, ruff, mypy). Use `pip install -e .[vision]` for
inference-only or `pip install -e .[dev]` for development without the backbone.

## Quick start

```bash
# 1. Train PatchTraj on the default dataset (SMD entity machine-1-1, line plot)
python scripts/train_patchtraj.py

# 2. Run detection with the trained checkpoint
python scripts/detect.py

# 3. Override any Hydra config from the CLI
python scripts/detect.py data=psm render=line_plot \
    scoring.dual_signal.alpha=0.1 scoring.dual_signal.auto_alpha=false
```

After installation no `PYTHONPATH` is required — `pip install -e .` registers
`src/` as the `vits` package.

## Reproducing paper tables

All paper tables are regenerated from the canonical artifact tree:

```bash
python scripts/regenerate_paper_tables.py     # Tables 1, 3, 4
python scripts/build_ucr_canonical.py         # 109 eligible UCR series
bash scripts/run_spatial_benchmark.sh         # 28-entity SMD benchmark (4 GPUs)
```

The `results/` directory layout is:

```
results/
├── benchmark_smd_spatial/      # 28 entities × {LP, RP} × seed_42
├── multiseed/                  # 5 seeds × {PSM, MSL, SMAP}
├── ucr_canonical/              # 109 UCR series (eligible_list.json + per_series/)
├── ablation/                   # alpha sweep, spatial-attention ablation
└── reports/                    # aggregated paper tables
```

## Tests

```bash
pytest tests/                       # full test suite
pytest tests/ -m "not slow"         # skip slow tests
pytest tests/ --cov=src             # coverage report
```

## Configuration

All hyperparameters use Hydra YAML:

- `configs/data/{smd,psm,msl,smap,ucr}.yaml` — dataset settings
- `configs/render/{line_plot,recurrence_plot,multi_view}.yaml` — renderer
- `configs/model/{dinov2_base,clip_base}.yaml` — vision backbone
- `configs/experiment/{patchtraj_default,patchtraj_spatial}.yaml` — full configs

## Citation

The paper is under review; a citation entry will be added once it is published.

## License

MIT — see `LICENSE` for details.
