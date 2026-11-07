# Pre-registration: SP, known anomalies injected into real YAHOO and non-YAHOO backgrounds

Written and pushed before any detector, score or metric was computed on the injected series.
Results will be reported whatever they are. Changes after this commit, if any, will be listed at
the end with reason and date.

## What was known when this was written
- All results of the paper. On the 472 held-out series YAHOO decides most signs, and its anomaly
  length and train-test drift are nearly the same variable as YAHOO there, so the paper cannot tell
  whether its reversal reflects the series' dynamics or their labelled anomalies.
- The synthetic test B1-B4 (sinusoidal backgrounds, Raw-1NN): per-window normalization misses the
  interior of level and scale anomalies longer than a window.
- Exploratory, computed for the TR registration on known results: in TSB-AD-U's YAHOO the toggle gain
  is positive on the real subset A1 and mostly negative on the synthetic A2 and A3/A4.
- The plan below was built from values and labels only (longest label-free stretch, TSB-AD's
  window-length estimate); no detector has been run on any planned series. The code was executed
  once on one unplanned injection (an other19 background shifted by 7 points, level, r = 4) to check
  that it runs; only its error status and output keys were printed.

## Design (fixed): `scripts/run_spikein.py`, plan `artifacts/official_protocol/spikein_plan.json`
(SHA-256 861c36c51fdf03f240163c4457f7041e852a6cd1f8ed3dae0b2e7887e685e6cb)
- **Backgrounds:** the first 1,000 points of the longest label-free stretch of a real series, with a
  training part of 500 (YAHOO's own), kept if TSB-AD's window-length estimate W is at most 60 (so a
  4W anomaly fits in the test part). Candidates in a seeded random order (seed 0):
  - **yahoo:** TSB-AD-U YAHOO series (any split): 20 backgrounds, W 24 to 48 (median 24);
  - **other19:** TSB-AD-U-Eva series of the 19 sources the 472 held-out series lack, at most 3 per
    source: 20 backgrounds (OPPORTUNITY, NAB, MGAB, SMD, Exathlon, IOPS 3 each; Daphnet, SWaT 1),
    W 6 to 49 (median 19).
  - Backgrounds need long label-free stretches, so they are not a random sample of either group.
- **Injections**, one per series, length r W with r in {0.5, 4}, start uniform in the test part
  with 50-point margins (seeded), sigma and m = training-part standard deviation and mean:
  level x += 3 sigma; scale x = m + 3 (x - m); shape: a sawtooth of period max(4, W/2) with the
  replaced segment's mean and standard deviation (as B1-B4). Labels: the injected segment only.
  2 groups x 20 backgrounds x 3 kinds x 2 ratios = 240 series.
- **Detectors:** TSB-AD's Sub-KNN and Sub-OCSVM, TSB-AD 1.5 classes, tuned hyperparameters, the
  toggle's per-window vs global replacement (training part 500); VUS-PR from get_metrics with the
  background's W as sliding window.
- **Statistic:** Delta = global - per-window per series. Within a group, a cell's test is the
  two-sided Wilcoxon signed-rank test of Delta; between groups, the two-sided Mann-Whitney test of
  Delta(yahoo) against Delta(other19). Direction convention as RT and TR (a Holm-significant result
  counts in a direction only when the mean and the rank statistic agree).

## Hypotheses and decision rules
**SP1 (long level and scale anomalies favour global in both backgrounds).** Sub-KNN, cells {level,
scale} x {yahoo, other19} at r = 4, Holm over the 4 cells.
- **Supported** if all 4 cells show a Holm-significant gain; **refuted** if any cell shows a
  Holm-significant loss; otherwise **inconclusive**.

**SP2 (YAHOO's background does not reverse the effect).** For each detector separately (Sub-KNN,
Sub-OCSVM), the 6 cells kind x r; D = mean Delta(yahoo) - mean Delta(other19); Mann-Whitney, Holm
over the 6 cells.
- **Background-neutral** if |D| <= 0.05 in at least 5 of 6 cells and no cell has a Holm-significant
  D < 0.
- **Background reverses** if at least 2 cells have a Holm-significant D < 0 with mean Delta(yahoo) < 0.
- Otherwise **inconclusive**.
- Registered reading: if both detectors are background-neutral, YAHOO's reversal on its real series
  is attributed to its labelled anomalies (their kinds, lengths and labels), not to its normal
  dynamics; if either reverses, to its dynamics.

**SP3 (shape anomalies).** Sub-KNN: the per-window VUS-PR is at least the global one minus 0.05 in
each of the 4 cells shape x {0.5, 4} x {yahoo, other19}. **Supported** if in all 4; **refuted** if it
is lower by more than 0.10 in 2 or more; otherwise **inconclusive**.

## Secondary (reported, no rule)
- All cells for both detectors; Delta against the background's train-test drift (drift_level,
  drift_scale in the plan).
- Failed runs.

## Compute cutoff
Series not finished by 2026-11-15 00:00 KST are left out and listed.

## Changes after registration
(none)
