# Pre-registration: follow-up to B1-B4 (F1-F3), separating duration from prevalence

Written and committed before running the analyses below (the F2 generator was only checked on seed 999 for anomaly placement; no measure was computed). They follow a mock review
(paper/icde2027/MODIFY.md, 3.3) of the registered B1-B4 (boundary_hypothesis.md, outcomes in
results.md, all supported). B1-B4 and their outcomes stay as registered; F1-F3 are secondary checks
of how to read them. Results will be reported whatever they are.

## Issues raised
1. In B1, the interior ROC-AUC compares interior points with all normal test points, including
   normal points whose centred window overlaps the anomaly (their score is raised by the edge).
2. In B3, duration and anomaly prevalence change together (the anomaly covers 0.8% to 53% of the
   test part), and VUS-PR is computed on the whole series.
3. B3's bootstrap resampled the two cells independently although they share seeds.

## F1 (clean negatives; rerun of the B1-B4 design)
Same series, detector and seeds as B1-B4. Additional measure: interior ROC-AUC against clean
normal points only, i.e. normal test points at least W = 100 steps from every anomalous point.
Rule: B1's reading ("per-window normalization is blind inside long level and scale anomalies")
stands if the per-window clean interior AUC is <= 0.65 in >= 7 of the 8 B1 cells; otherwise we
report that B1 depended on edge-contaminated negatives.

## F2 (duration at fixed prevalence)
- Series: training part 3000, test part 12000 (n = 15000), same normal signal as B1-B4.
- Total anomalous test points P in {400, 1600}; durations d in {25, 100, 400, 1600} with d <= P;
  k = P / d anomalies placed one per equal slot of the test part at a random offset that leaves at
  least W normal steps before each anomaly inside its slot (so anomalies are at least W apart; with
  64 anomalies of 25 steps a wider gap does not fit).
- Kinds: level +1 sigma, scale x2, shape (as in B1-B4). 10 series per cell (seeds 0-9).
- Raw-1NN, global vs per-window, as in B1-B4.
- Measures: test-part ROC-AUC and average precision (AUC-PR) of point scores, VUS-PR on the whole
  series (slidingWindow 50), interior and clean interior AUC as in F1.
- Rule: "the cost grows with duration at fixed prevalence" is supported if, for level and for scale,
  at both P, the per-window minus global test AUC-PR is lower (more negative) at the longest d than
  at the shortest d by at least 0.05; refuted if it is higher at the longest d for two or more of the
  four (kind, P) pairs; otherwise inconclusive.

## F3 (paired B3)
From the stored B1-B4 records: per seed, (gap at r = 16) - (gap at r = 0.25), gap = global minus
per-window VUS-PR, level anomalies, both magnitudes; mean and 95% bootstrap interval over the 20 seeds
(10,000 resamples, seed 0). Reported beside B3, not replacing it.

## Changes after registration
(none)
