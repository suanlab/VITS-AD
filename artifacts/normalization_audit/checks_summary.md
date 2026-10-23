| check (kind) | detector | A | B | C |
|---|---|---|---|---|
| fit_score_match (contract) | HBOS | pass | pass | pass |
| fit_score_match (contract) | MCD | pass | pass | pass |
| fit_score_match (contract) | OCSVM | VIOLATION | pass | pass |
| fit_score_match (contract) | PCA | pass | pass | pass |
| non_constant (degeneracy) | HBOS | degenerate | pass | pass |
| non_constant (degeneracy) | IForest | degenerate | pass | pass |
| non_constant (degeneracy) | LOF | degenerate | pass | pass |
| non_constant (degeneracy) | Sub_HBOS | pass | pass | pass |
| non_constant (degeneracy) | Sub_IForest | pass | pass | pass |
| non_constant (degeneracy) | Sub_KNN | pass | pass | pass |
| non_constant (degeneracy) | Sub_LOF | pass | pass | pass |
| non_constant (degeneracy) | Sub_MCD | pass | pass | pass |
| non_constant (degeneracy) | Sub_OCSVM | pass | error (crash) | pass |
| non_constant (degeneracy) | Sub_PCA | pass | pass | pass |
| channel_shift (capability) | HBOS | warning | warning | pass |
| channel_shift (capability) | IForest | warning | pass | pass |
| channel_shift (capability) | KNN | warning | warning | pass |
| channel_shift (capability) | LOF | warning | warning | pass |
| channel_shift (capability) | PCA | pass | warning | pass |
| affine_order (info) | HBOS | tau nan | tau 1.000 | tau 1.000 |
| affine_order (info) | IForest | tau nan | tau 1.000 | tau 1.000 |
| affine_order (info) | LOF | tau nan | tau 1.000 | tau 1.000 |
| affine_order (info) | Sub_HBOS | tau 1.000 | tau 1.000 | tau 1.000 |
| affine_order (info) | Sub_IForest | tau 1.000 | tau 1.000 | tau 1.000 |
| affine_order (info) | Sub_KNN | tau 1.000 | tau 1.000 | tau 1.000 |
| affine_order (info) | Sub_LOF | tau 1.000 | tau 1.000 | tau 1.000 |
| affine_order (info) | Sub_MCD | tau 0.649 | tau 1.000 | tau 1.000 |
| affine_order (info) | Sub_OCSVM | tau 1.000 | error (crash) | tau 1.000 |
| affine_order (info) | Sub_PCA | tau 1.000 | tau 1.000 | tau 1.000 |
| deep_input_norm (contract) | ReconstructDataset | VIOLATION | pass | pass |
| deep_input_norm (contract) | ForecastDataset | VIOLATION | pass | pass |
| channel_shift (capability) | control_shift_invariant_1nn | fail (declared invariance) | fail (declared invariance) | fail (declared invariance) |
| channel_shift (capability) | control_plain_1nn | pass | pass | pass |
| fit_score_match (contract) | control_fit_only_scaling | VIOLATION | VIOLATION | VIOLATION |
| fit_score_match (contract) | control_consistent_scaling | pass | pass | pass |
| fingerprint (version) | HBOS | ff6698a6e831ffcf | changed | unchanged |
| fingerprint (version) | IForest | ff6698a6e831ffcf | changed | unchanged |
| fingerprint (version) | LOF | ff6698a6e831ffcf | changed | unchanged |
| fingerprint (version) | Sub_HBOS | 2b7d4c04970fbbfd | changed | unchanged |
| fingerprint (version) | Sub_IForest | 9764bb1814deddbf | changed | unchanged |
| fingerprint (version) | Sub_KNN | c91dff78e5f09107 | changed | unchanged |
| fingerprint (version) | Sub_LOF | 547d9d9c08fc744c | changed | unchanged |
| fingerprint (version) | Sub_MCD | 25ad1977adef6f12 | changed | unchanged |
| fingerprint (version) | Sub_OCSVM | 8d9b83db19f77068 | changed | changed |
| fingerprint (version) | Sub_PCA | abfa6138537d73d9 | changed | unchanged |
