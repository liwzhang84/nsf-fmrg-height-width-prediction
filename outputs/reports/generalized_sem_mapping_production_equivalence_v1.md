# Generalized SEM mapping production-implementation equivalence v1

## Scope distinction

The previous report demonstrated old-pipeline self-reproduction through the development `prepare` path. This audit does not use that path as the generalized implementation. It independently executes the exact production geometry, tile selection, native bounds, masking, and summary extraction used by the completed final runner.

## Development N=13

- Reconstructed mapping rows: 1192 / 1192
- Cohort-B IDs: 678 / 678; identical: True
- Mapping/bounds mismatch count: 0
- SEM-summary hash: `4f8473e69b6c05d0049c920676b6cc0891b708bca176c86fd6acc51815033203`
- Required frozen hash: `4f8473e69b6c05d0049c920676b6cc0891b708bca176c86fd6acc51815033203`
- Upstream-summary hash: `661295a1a6383079e9aba2be49220d02ed8671f4ba38755159dd9c17b1ae6810`
- Native-bound hash: `953f6849819cfbc1a521bc3c63165706c45d52796751b7d624df15eb06cac3b3`
- Per-Track summary hashes: `{"10": "9a753d17936e1c38d275314d14cd42d96b838e080d67d6ea2feba5db031e1fae", "14": "b6c9cc05b1e83dd3532237169d1e90f3f380c52b690bd2510ca01a7858d10bf9", "8": "b9a889d5ab6f6cd4569da8a9073b9438578589947d42423c0f7e68d5605be13e"}`

## Track-21 N=14

- Reconstructed mapping rows: 396 / 396
- Cohort-B IDs: 233 / 233; identical: True
- Mapping mismatch count: 0
- Manifest/native bounds identical: True
- SEM-summary hash: `276dc5fcfd9e3403e81c00e7b4a4af8d6cec9cb25b55e706a7e470a0bc4d1719`
- Upstream-summary hash: `9388ae48631a2527867d55f45c6ae70397c1111e9359a49ccae3c303571d3742`

## Prediction reproduction

Maximum absolute differences: `{"cohort_b_sem_summary_ridge": 0.0, "cohort_b_sem_summary_upstream_ridge": 0.0, "cohort_b_thermal_plus_sem_summary_ridge": 0.0}`. Required tolerance: `1e-10 mm`.

## Immutable integrity and mismatches

- All immutable hashes unchanged: True
- Mismatch counts: `{}`
- Existing final interpretation remains unchanged: **SEM IMPROVEMENT NOT CONFIRMED ON FINAL TRACK 21**

GENERALIZED SEM PRODUCTION IMPLEMENTATION EQUIVALENCE PASS
