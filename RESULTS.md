# Results

## 1. Development baseline summary

Development used complete-Track leave-one-track-out evaluation on Tracks 8, 10, and 14.
The common leakage-safe SEM cohort contained 678 samples. These results are development
evidence, not held-out confirmation.

## 2. Thermal representation audit

The audit concluded `THERMAL SIGNAL NOT GENERALIZABLE ACROSS DEVELOPMENT TRACKS`.
Any Thermal-only Track-21 value is descriptive and does not overturn the lack of stable
Thermal generalization across development Tracks.

## 3. Development SEM LOTO results

The frozen development reports record the complete-track fold results. They motivated
freezing the SEM summary representation but do not constitute final held-out evidence.

## 4. Final Track-21 results

| Model | Cohort-B MAE |
|---|---:|
| x-only | 0.111856 mm |
| SEM summary | 0.109837 mm |
| Thermal + SEM | 0.110343 mm |

### Final-result visualizations

#### Predictions along the physical track

![Track-21 predictions and absolute errors](outputs/figures/final_track21_height_width_v2/final_track21_predictions_and_errors.png)

This panel shows the frozen predictions and absolute errors along the physical track.

#### Cohort-B MAE comparison

![Track-21 Cohort-B model MAE comparison](outputs/figures/final_track21_height_width_v2/final_track21_cohort_b_mae_comparison.png)

The MAE values of all main Cohort-B models are close. The fixed Thermal-only result is
descriptive, not confirmatory.

#### SEM improvement versus the project-defined threshold

![SEM improvement compared with the project-defined threshold](outputs/figures/final_track21_height_width_v2/final_track21_sem_improvement_threshold.png)

SEM summary produced a positive reduction, but it remained below the project-defined threshold.

#### Predicted width versus frozen target

![Track-21 predicted width versus frozen target](outputs/figures/final_track21_height_width_v2/final_track21_predicted_vs_target_scatter.png)

The scatter plots show that predictions occupy a narrower range than the frozen
Height-derived target. These figures add no new scientific conclusion.

The observed SEM reduction was 0.002019 mm (1.805%). Thermal + SEM was 0.000506 mm
worse than SEM alone.

## 5. Project-defined predeclared confirmation criterion

The project predeclared a conservative confirmation threshold of
`max(0.003 mm, 3% of the x-only MAE)`. For Track 21 this corresponded to 0.003356 mm.
This was an internal scientific interpretation criterion, not an official challenge
scoring or eligibility requirement.

The observed SEM reduction was 0.002019 mm, or 1.805%. The improvement was positive but
below the project-defined threshold. Additive Thermal value over SEM was also unsupported.
See [Terminology and interpretation](docs/TERMINOLOGY_AND_INTERPRETATION.md).

## 6. Production-equivalence audit

The exact generalized N-to-1 implementation reproduced 1192/1192 development mappings,
678/678 development cohort IDs, 396/396 Track-21 mappings, and 233/233 Track-21 cohort
IDs. Prediction differences were 0.0, immutable hashes were unchanged, and mismatches
were zero.

## 7. Final scientific interpretation

**SEM IMPROVEMENT NOT CONFIRMED ON FINAL TRACK 21**

Masked neighboring SEM context produced a small held-out MAE reduction, but the
predeclared confirmatory criterion was not met. This does not establish that SEM has no
effect or performs worse than x-only. Thermal and multimodal benefit are not confirmed.
This conclusion concerns confirmation under the project's conservative criterion. It
does not determine the official challenge ranking or submission status.
