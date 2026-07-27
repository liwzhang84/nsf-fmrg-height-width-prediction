# Leakage controls

- Complete Tracks, rather than random positions, define development folds.
- Track 21 is excluded from feature, model, alpha, protocol, coordinate, and
  project-defined confirmation-criterion selection.
- Height maps are targets only and never model features.
- The SEM central region `x ± 0.40 mm` is excluded in native tile coordinates.
- Upstream and downstream contexts are each fixed at 1.00 mm.
- Masking precedes resizing, filtering, and feature extraction.
- Context must be fully supported by one native tile; unsupported cases fail closed.
- Human references are diagnostic and do not replace frozen automatic targets.
- The Tile-14 amendment uses source metadata only, not pixels, labels, predictions, or error.
- Final artifacts are loaded for prediction without fitting.
- The post-test equivalence audit compares mappings, cohorts, features, and predictions without selection.

The project-defined predeclared confirmation threshold was frozen before final
evaluation. It was not specified as an official challenge requirement.

## v2 additions

Isotherm extents are computed from in-situ frames recorded before the
final surface exists and cannot encode the target. The anchor constant
and the count selection use development labels only. The residual model
hyperparameter, the level-uncertainty scale, and the confirmation
criterion were all fixed in the v2 lock before the v3 test script was
run. The validity-run sensitivity study recalibrates the anchor on
development tracks only and inherits every other frozen setting.
