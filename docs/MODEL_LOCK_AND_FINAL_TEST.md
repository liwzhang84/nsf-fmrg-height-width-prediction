# Model lock and final test

The final lock fixes target policy, Thermal representation, SEM mapping and mask,
features, model families, coefficients, ridge alphas, cohorts, and the project-defined
predeclared confirmation criterion.
Its immutable SHA-256 is `2fb60cbb83a04ea9819c3bfff9fa39b9765dc9bad74b4a7d0693103af013be1b`.

Track 21 remained untouched during development. The first attempted runner stopped
before a model test because it detected a source-schema incompatibility: Track 21 had
canonical Tile 14 while development Tracks had 13 tiles. No held-out pixels, labels,
predictions, metrics, or human references had been loaded, so this was not a failed
model test.

A metadata-only review established that Tile 14 was a canonical contiguous-sequence
member. The compatibility amendment generalized the already documented N-to-1 rule
without changing the scientific model. Its SHA-256 is `3c0de4b78b5417cd60df27bbb9e059bcf27986aaadd8964d14e51dbf831a847a`. Only after that
amendment was the one-time final test executed. No post-test tuning is permitted.

The criterion `max(0.003 mm, 3% of x-only MAE)` was defined by this project before the
final evaluation for conservative interpretation. It was not an official challenge
scoring, eligibility, or submission requirement. See
[Terminology and interpretation](TERMINOLOGY_AND_INTERPRETATION.md).

The frozen interpretation remains: **SEM IMPROVEMENT NOT CONFIRMED ON FINAL TRACK 21**.

## v2 protocol cycle (post-registration revision)

A second cycle adds an isotherm-extent thermal representation, a
development-calibrated linear width anchor, residual linear quantile
models (same JSON artifact schema, applied by the same predict function),
level-uncertainty inflation, and a moving-block-bootstrap confirmation
criterion. Frozen inputs: `configs/isotherm_anchor_v2.yaml` and
`configs/final_height_width_model_lock_v2.yaml` (hash
8f18122e2192d714ce2e8b3766c3e993ac15d933df2b1fd7c191c274082622de), both
derived from development tracks only. Track 21 had already been opened by
the v1 final test, so the v2 evaluation is a protocol revision; a clean
confirmation of the v2 criterion requires a new held-out track. Outcomes:
NOT CONFIRMED on the primary gradient-edge labels; CONFIRMED (70.1%
improvement) under the validity-run sensitivity labels. See the README
v2 section for the interpretation.
