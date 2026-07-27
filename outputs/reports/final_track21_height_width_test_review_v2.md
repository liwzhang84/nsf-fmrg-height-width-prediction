# Final Track-21 Height-width test review v2

## Frozen execution

- Original model-lock hash: `2fb60cbb83a04ea9819c3bfff9fa39b9765dc9bad74b4a7d0693103af013be1b`
- Compatibility-amendment hash: `3c0de4b78b5417cd60df27bbb9e059bcf27986aaadd8964d14e51dbf831a847a`
- No fitted artifact, coefficient, alpha, feature, mask, target policy, Thermal representation, or success rule changed.
- Tile 14 was retained under the metadata-authorized contiguous variable-N rule.

## Cohorts

- Official Height-supported positions: 396
- Cohort A primary eligible rows: 372
- Cohort B identical common SEM/Thermal rows: 233

## Primary comparison

- Cohort-B x-only MAE: 0.111856 mm
- SEM-summary MAE: 0.109837 mm
- Absolute MAE reduction: 0.002019 mm
- Required reduction: 0.003356 mm (`max(0.003 mm, 3%)`)
- Relative reduction: 1.805%
- RMSE difference (SEM − x): -0.004906 mm
- Signed-bias difference (SEM − x): -0.018727 mm
- Absolute-bias worsening: -0.018727 mm
- Largest 20-mm-region contribution fraction: 0.581
- SEM generalization supported: **False**

## Secondary comparison

- Thermal+SEM MAE: 0.110343 mm
- MAE reduction versus SEM: -0.000506 mm
- Required reduction: 0.002 mm
- Absolute-bias worsening: 0.000473 mm
- Confirmed additive Thermal value: **False**

Subgroup and historical-human results are descriptive only and did not alter the locked interpretation.

SEM IMPROVEMENT NOT CONFIRMED ON FINAL TRACK 21
