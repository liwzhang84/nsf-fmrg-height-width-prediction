# Final Height-width model lock v1

- Lock hash: `2fb60cbb83a04ea9819c3bfff9fa39b9765dc9bad74b4a7d0693103af013be1b`
- Development Tracks: 8, 10, 14.
- Final held-out Track: 21.
- Cohort A development rows: 1155.
- Cohort B development rows: 678.
- Required final models: training mean, x-only, fixed Thermal shape ridge, SEM summary ridge, Thermal+SEM summary ridge.
- Sensitivity only: upstream SEM summary ridge.
- No PCA, CNN, new feature search, or random position split.
- Locked cohort-A alphas: `{"fixed_thermal_shape_ridge": 1000.0, "x_only": 1000.0}`.
- Locked cohort-B alphas: `{"fixed_thermal_shape_ridge": 1000.0, "sem_summary_ridge": 1000.0, "sem_summary_upstream_ridge": 100.0, "thermal_plus_sem_summary_ridge": 1000.0, "x_only": 1000.0}`.
- Mapping confirmation was archived before final-test source inspection.
