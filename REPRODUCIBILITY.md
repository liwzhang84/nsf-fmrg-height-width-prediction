# Reproducibility

Track 21 must never be used to select features, models, alpha, masking protocol,
coordinate correction, or the project-defined predeclared confirmation threshold.
The criterion was frozen before final evaluation to preserve interpretive integrity.
Its use does not imply that it was specified by the challenge organizers.

| Stage | Command | Required inputs | Key output/check | Held-out access | Fitting permitted |
|---|---|---|---|---|---|
| 1. Environment | `python3 -m pip install -r requirements.txt` | Python environment | imports succeed | none | no |
| 2. Raw placement | see `DATA_AVAILABILITY.md` | authorized MAT/TIFF/ASC files | Tracks 8,10,14,21 present | files placed only | no |
| 3. Height labels | `python3 scripts/build_height_quality_gate_review.py` | Height data, frozen Height config | development labels; 1192 rows | no Track 21 | frozen estimator only |
| 4. Thermal features | `python3 scripts/run_thermal_representation_audit.py` | Thermal data, development labels | representation audit reports | no Track 21 | development LOTO only |
| 5. Development SEM | `python3 scripts/run_sem_multimodal_deadline_models.py` | SEM tiles, frozen mapping confirmation | 678 common samples | no Track 21 | development LOTO only |
| 6. Model lock | `python3 scripts/freeze_final_height_width_model_v1.py` | development artifacts | lock hash `2fb60cbb83a04ea9819c3bfff9fa39b9765dc9bad74b4a7d0693103af013be1b` | no Track 21 | final development fit only |
| 7. Tile-14 amendment | `python3 scripts/prepare_final_test_sem_compatibility_amendment_v1.py` | filenames and TIFF metadata | amendment hash `3c0de4b78b5417cd60df27bbb9e059bcf27986aaadd8964d14e51dbf831a847a` | metadata only; no pixels/labels | no |
| 8. Final test | `python3 scripts/run_final_track21_height_width_test_v2.py` | frozen artifacts and Track-21 raw data | 396 labels, 372 eligible, 233 Cohort-B | one-time test | no |
| 9. Equivalence | `MPLCONFIGDIR=/tmp/nsf_mplconfig python3 scripts/audit_generalized_sem_mapping_production_equivalence_v1.py` | all compact frozen outputs and raw SEM/Thermal | PASS; maximum prediction difference 0.0 | reproduction only | no |

## 10. Expected outputs

Development has 1192 mapping rows and 678 common SEM samples. Track 21 has 396
Height-supported rows, 372 primary eligible rows, and 233 Cohort-B rows. Expected
development feature hash:
`4f8473e69b6c05d0049c920676b6cc0891b708bca176c86fd6acc51815033203`.

## 11. Troubleshooting

- Missing MAT/TIFF/ASC errors: verify the exact layout in `DATA_AVAILABILITY.md`.
- Matplotlib cache errors: set `MPLCONFIGDIR` to a writable temporary directory.
- Lock/amendment mismatch: stop; do not regenerate or substitute frozen files.
- Tile sequence error: require contiguous canonical IDs 1 through N.
- Equivalence failure: inspect audit CSV/JSON; do not tune against Track 21.

The packaged outputs permit result inspection without raw data. Data-dependent stages
must not be run until authorized source data are locally available.
