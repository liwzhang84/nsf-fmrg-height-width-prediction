# Project-defined threshold terminology consistency audit v1

## Scope and classification

Frozen scientific artifacts were inspected but not edited. Editable source and generated
public documentation were checked for project-qualified terminology. In a preserved frozen
report, “Required reduction” refers to the project-defined predeclared confirmation
criterion; the public terminology note explains that it is not an official challenge rule.

- Terminology note: `docs/TERMINOLOGY_AND_INTERPRETATION.md`
- Disallowed phrases remaining in editable public files: 0

## Files containing the numeric threshold

| File | Line | Exact surrounding phrase | Classification | Acceptable | Explanatory note | Incorrect official implication |
|---|---:|---|---|---|---|---|
| `outputs/reports/final_track21_height_width_test_review_v2.md` | 21 | - Required reduction: 0.003356 mm (`max(0.003 mm, 3%)`) | frozen scientific artifact | yes | yes | no |
| `scripts/create_final_track21_result_visualizations_v1.py` | 337 | and abs(required - 0.003356) <= 1e-6 | editable/public | yes | document-level explanation | no |
| `scripts/create_github_release_v1.py` | 237 | - Project-defined predeclared confirmation threshold: 0.003356 mm | editable/public | yes | linked terminology note | no |
| `scripts/create_github_release_v1.py` | 390 | `max(0.003 mm, 3% of the x-only MAE)`. For Track 21 this corresponded to 0.003356 mm. | editable/public | yes | linked terminology note | no |
| `docs/TERMINOLOGY_AND_INTERPRETATION.md` | 14 | For Track 21, this equals 0.003356 mm. The project froze this criterion before final | editable/public | yes | yes | no |
| `github_release/nsf-fmrg-height-width-prediction-v1/README.md` | 79 | - Project-defined predeclared confirmation threshold: 0.003356 mm | editable/public | yes | linked terminology note | no |
| `github_release/nsf-fmrg-height-width-prediction-v1/RESULTS.md` | 62 | `max(0.003 mm, 3% of the x-only MAE)`. For Track 21 this corresponded to 0.003356 mm. | editable/public | yes | linked terminology note | no |
| `github_release/nsf-fmrg-height-width-prediction-v1/docs/TERMINOLOGY_AND_INTERPRETATION.md` | 14 | For Track 21, this equals 0.003356 mm. The project froze this criterion before final | editable/public | yes | yes | no |

## Disallowed-phrase scan

No editable public-facing file contains any of the following phrases:

- `official challenge threshold`
- `challenge-required improvement`
- `required by the challenge`
- `challenge passing threshold`
- `failed the challenge`
- `challenge failure`
- `did not pass the challenge`

## Frozen integrity context

- `configs/final_height_width_model_lock_v1.yaml` — SHA-256 `2fb60cbb83a04ea9819c3bfff9fa39b9765dc9bad74b4a7d0693103af013be1b`
- `configs/final_test_sem_source_compatibility_amendment_v1.yaml` — SHA-256 `3c0de4b78b5417cd60df27bbb9e059bcf27986aaadd8964d14e51dbf831a847a`
- `outputs/models/final_track21_height_width_metrics.csv` — SHA-256 `7a541a5cdd91fc6581f661185a29befc9f91785a6194d439f942131974c8bd13`
- `outputs/models/final_track21_height_width_predictions.csv` — SHA-256 `8ef1ad999a56aba522556fe7ed26fe05f281c2f1ab1af87fdd5febfa02cae78d`
- `outputs/reports/final_track21_height_width_build_report_v2.md` — SHA-256 `e8639be68ad93dfa7504ac46f42f734333365f7331ad940f1fd9ec698ebe29d7`
- `outputs/reports/final_track21_height_width_leakage_audit_v2.md` — SHA-256 `f9d5a54979fd63eb5b65374809f488a74514030e7f829706e6693c7ef5cbfd2a`
- `outputs/reports/final_track21_height_width_test_review_v2.md` — SHA-256 `b9c1fae8be842973d3689f9eb21c1597a100eb3c01d0a6021a503164478c995b`
- `outputs/reports/generalized_sem_mapping_production_equivalence_v1.md` — SHA-256 `85fb5c67ca540dbb374e5351dcfd0720e65ab360945b629b07f7f5eb5f49c149`

The threshold formula and value remain unchanged: `max(0.003 mm, 3% of x-only MAE)`
and `0.003356 mm` on Track 21. The frozen status remains
`SEM IMPROVEMENT NOT CONFIRMED ON FINAL TRACK 21`.

PROJECT-DEFINED THRESHOLD TERMINOLOGY CONSISTENT
