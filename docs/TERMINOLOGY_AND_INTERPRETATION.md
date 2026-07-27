# Terminology and interpretation

## Observed SEM improvement

The numerical difference between the frozen x-only and SEM-summary held-out MAEs was
0.002019 mm, or 1.805%.

## Project-defined predeclared confirmation threshold

The internally defined criterion was:

`max(0.003 mm, 3% of x-only MAE)`

For Track 21, this equals 0.003356 mm. The project froze this criterion before final
evaluation to support conservative scientific interpretation.

This threshold was defined by the project for conservative scientific interpretation.
It was not an official challenge scoring, eligibility, or submission requirement.

## Official challenge requirement

No uploaded or packaged official challenge document establishes the project's
0.003 mm / 3% threshold as an official scoring, eligibility, or submission requirement.

## Challenge outcome

The project-defined confirmation result alone does not determine challenge success,
failure, ranking, eligibility, or acceptance. Not meeting this project-defined threshold
does not mean that the challenge submission failed.

## Scientific conclusion

SEM produced a small positive held-out improvement, but the project did not treat that
improvement as sufficiently large for confirmatory interpretation.

The frozen scientific status remains:

`SEM IMPROVEMENT NOT CONFIRMED ON FINAL TRACK 21`

## v2 terms

- **Isotherm anchor**: `anchor(x) = k* x iso_extent(x)`, a linear function
  of an in-situ measurement, calibrated on development tracks. It carries
  the cross-power level of the prediction; learned models handle only the
  residual.
- **Residual quantile model**: linear quantile regression on
  `width - anchor` at q = 0.05...0.95.
- **Level inflation**: quantiles are widened about the median using the
  RMS of the leave-one-track-out fold-level median errors.
- **v2 criterion**: improvement is confirmed only if the 95% moving-block
  bootstrap CI lower bound of the MAE difference exceeds zero and the
  relative improvement is at least 5%. This replaces the v1 fixed-mm
  threshold for v2 comparisons.
- **Post-registration revision**: track 21 was opened once by the v1
  test; v2 results are therefore labeled as a revision, not a second
  independent confirmation.
- **Sensitivity cohort**: the same frozen v2 settings evaluated under the
  validity-run label estimator. Reported separately from the primary
  labels; neither replaces the other.
