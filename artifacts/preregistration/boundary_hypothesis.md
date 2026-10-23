# Pre-registration: anomaly boundaries, duration and per-window normalization (B1-B4)

Written and committed before the experiment below was run. Before committing, the script was only
linted and executed once on seed 999 (outside the registered seeds) to check that it runs; no
measure was computed or displayed. Results will be reported whatever they are. Changes after this commit, if any, will
be listed at the end with reason and date.

## Motivation
Per-window z-normalization removes a window's level and scale (paper, Sec. VII-A). A window that lies
entirely inside a level or scale anomaly therefore looks normal after it, but a window that straddles
the anomaly's start or end does not: the step itself is part of its shape. Per-window detectors may
thus see the edges of long level/scale anomalies but not their interior, which point- and
range-based metrics weigh differently. Suggested by a mock review (paper/icde2027/IDEA_ICDE.md, 6.4).

## Design (fixed)
- Script: `scripts/run_boundary_synthetic.py` (written with this file).
- Series: n = 6000, training part 3000. Normal signal x_t = sin(2 pi t / 50) + 0.5 sin(2 pi t / 170)
  + e_t, e_t ~ N(0, 0.2^2). sigma = standard deviation of the training part.
- One anomaly per series, duration d = r W with W = 100 and r in {0.25, 1, 4, 16}, start drawn
  uniformly from [3200, 6000 - d - 200].
  * level: x[a:a+d] += delta sigma, delta in {1, 3}
  * scale: x[a:a+d] = m + s (x[a:a+d] - m), m = training mean, s in {2, 3}
  * shape: x[a:a+d] replaced by a sawtooth of period 50 with the mean and standard deviation of the
    replaced segment (one magnitude)
- 20 series per cell (seeds 0-19: noise and start position).
- Detector: Raw-1NN as in `run_official_protocol.py` (W = 100, stride 1 for training and scored
  windows), with global normalization (training mean and std) or per-window z-normalization
  (ddof = 1); window scores to points with `to_points` (window centres).
- Regions of the anomaly: boundary = anomalous points within W/2 of its first or last point;
  interior = anomalous points at least W/2 from both (exists only for r in {4, 16}).
- Measures per series: ROC-AUC of interior points against normal test points; the same for boundary
  points; VUS-PR from TSB-AD's `get_metrics` with slidingWindow = 50. Cell means over the 20 series.

## Hypotheses and decision rules
- **B1 (interior blindness).** In the 8 cells {level, scale} x {two magnitudes} x r in {4, 16}: the
  per-window interior AUC is <= 0.65 and the global interior AUC is >= 0.90.
  Supported if both hold in >= 7 of 8 cells; refuted if the per-window interior AUC is >= 0.75 in >= 3
  cells; otherwise inconclusive.
- **B2 (edges stay visible).** In the same 8 cells, the per-window boundary AUC is >= 0.80.
  Supported if in >= 7 of 8; refuted if < 0.65 in >= 3; otherwise inconclusive.
- **B3 (cost grows with duration).** For level anomalies, the mean VUS-PR gap (global minus
  per-window) at r = 16 exceeds that at r = 0.25, for both magnitudes, with a 95% bootstrap interval
  of the difference of gaps (10,000 resamples of series within each cell, seed 0) excluding 0.
  Supported if for both magnitudes; refuted if the interval lies below 0 for either; otherwise
  inconclusive.
- **B4 (shape anomalies).** For shape anomalies, the per-window VUS-PR is at least the global one
  minus 0.05 at every r. Supported if at all four r; refuted if it is lower by more than 0.10 at two or
  more r; otherwise inconclusive.

## Changes after registration
(none)
