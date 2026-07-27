# Final Track-21 visualization summary v1

## Frozen sources

- `outputs/models/final_track21_height_width_metrics.csv` — SHA-256 `7a541a5cdd91fc6581f661185a29befc9f91785a6194d439f942131974c8bd13`
- `outputs/models/final_track21_height_width_predictions.csv` — SHA-256 `8ef1ad999a56aba522556fe7ed26fe05f281c2f1ab1af87fdd5febfa02cae78d`
- Creation timestamp (UTC): `2026-07-26T03:59:05.919766+00:00`

No model fitting, estimator refitting, feature generation, label regeneration, model
selection, or metric recalculation was used to change the frozen results. Immutable
source and final-report hashes were identical before and after figure creation.

## Plotted models and values

- MAE figure model IDs, in order: `["cohort_b_x_only", "cohort_b_fixed_thermal_shape_ridge", "cohort_b_sem_summary_ridge", "cohort_b_thermal_plus_sem_summary_ridge"]`
- Scatter model IDs, in order: `["cohort_b_x_only", "cohort_b_sem_summary_ridge", "cohort_b_thermal_plus_sem_summary_ridge"]`
- Scatter sample count per model: `233`
- Exact MAE values (mm): `{"cohort_b_fixed_thermal_shape_ridge": 0.1084509327881748, "cohort_b_sem_summary_ridge": 0.1098367509669706, "cohort_b_thermal_plus_sem_summary_ridge": 0.110342590016758, "cohort_b_x_only": 0.1118561489811449}`
- Actual SEM improvement: `0.002019398014 mm`
- Project-defined predeclared confirmation threshold: `0.003355684469 mm`
- Relative SEM improvement: `1.805352708%`
- Target standard deviation (`ddof=0`): `0.136355147107 mm`
- Prediction standard deviations (`ddof=0`): `{"cohort_b_sem_summary_ridge": 0.0159016787975175, "cohort_b_thermal_plus_sem_summary_ridge": 0.032086806159535464, "cohort_b_x_only": 0.006647309063154287}`

## Figures

![Track-21 predictions and absolute errors](../figures/final_track21_height_width_v2/final_track21_predictions_and_errors.png)

![Track-21 Cohort-B MAE comparison](../figures/final_track21_height_width_v2/final_track21_cohort_b_mae_comparison.png)

![SEM improvement threshold](../figures/final_track21_height_width_v2/final_track21_sem_improvement_threshold.png)

![Predicted versus frozen target](../figures/final_track21_height_width_v2/final_track21_predicted_vs_target_scatter.png)

Vector PDF versions are stored beside each PNG.

Generated PNG and PDF paths: `["outputs/figures/final_track21_height_width_v2/final_track21_cohort_b_mae_comparison.png", "outputs/figures/final_track21_height_width_v2/final_track21_cohort_b_mae_comparison.pdf", "outputs/figures/final_track21_height_width_v2/final_track21_sem_improvement_threshold.png", "outputs/figures/final_track21_height_width_v2/final_track21_sem_improvement_threshold.pdf", "outputs/figures/final_track21_height_width_v2/final_track21_predicted_vs_target_scatter.png", "outputs/figures/final_track21_height_width_v2/final_track21_predicted_vs_target_scatter.pdf"]`.
The preserved pre-existing prediction/error PNG is
`outputs/figures/final_track21_height_width_v2/final_track21_predictions_and_errors.png`.

## Interpretation

SEM summary produced a small positive MAE reduction relative to x-only, but the reduction
did not meet the project-defined predeclared confirmation threshold. This criterion was
defined for conservative interpretation and was not an official challenge requirement.
Not meeting it does not mean that the challenge submission failed. Thermal + SEM did not
improve over SEM-only.
The fixed Thermal-only Track-21 result is descriptive and does not overturn the lack of
stable Thermal generalization across development Tracks.

VISUALIZATIONS GENERATED FROM FROZEN FINAL RESULTS
