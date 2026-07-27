#!/usr/bin/env python3
"""Build the self-contained, public, frozen v1 GitHub release package."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "github_release/nsf-fmrg-height-width-prediction-v1"
LOCK_HASH = "2fb60cbb83a04ea9819c3bfff9fa39b9765dc9bad74b4a7d0693103af013be1b"
AMEND_HASH = "3c0de4b78b5417cd60df27bbb9e059bcf27986aaadd8964d14e51dbf831a847a"
FINAL_STATUS = "SEM IMPROVEMENT NOT CONFIRMED ON FINAL TRACK 21"

CORE_FILES = [
    "nsf_fmrg_data.py",
    "scripts/create_github_release_v1.py",
    "scripts/build_height_quality_gate_review.py",
    "scripts/run_thermal_height_baselines.py",
    "scripts/run_thermal_representation_audit.py",
    "scripts/run_sem_multimodal_deadline_models.py",
    "scripts/freeze_final_height_width_model_v1.py",
    "scripts/prepare_final_test_sem_compatibility_amendment_v1.py",
    "scripts/run_final_track21_height_width_test_v2.py",
    "scripts/audit_generalized_sem_mapping_production_equivalence_v1.py",
    "scripts/generalized_sem_mapping_v1.py",
    "scripts/create_final_track21_result_visualizations_v1.py",
    "scripts/audit_threshold_terminology_v1.py",
    "docs/TERMINOLOGY_AND_INTERPRETATION.md",
    "src/__init__.py",
    "src/height_boundary_estimator_v1.py",
    "src/official_coordinates.py",
    "src/human_review/__init__.py",
    "src/human_review/data_access.py",
]

CONFIG_FILES = [
    "configs/height_boundary_estimator_v1.yaml",
    "configs/height_label_policy_v1.yaml",
    "configs/final_height_width_model_lock_v1.yaml",
    "configs/sem_mapping_engineering_confirmation_v1.yaml",
    "configs/sem_mapping_engineering_confirmation_final_v1.yaml",
    "configs/sem_mapping_deadline_v1.yaml",
    "configs/sem_mask_protocol_deadline_v1.yaml",
    "configs/thermal_representation_audit_v1.yaml",
    "configs/final_test_sem_source_compatibility_amendment_v1.yaml",
]

OUTPUT_FILES = [
    "outputs/height_labels/height_labels_v1_development.csv",
    "outputs/height_labels/height_labels_v1_track21_final_test.csv",
    "outputs/manifests/thermal_height_manifest_v1_development.csv",
    "outputs/manifests/thermal_height_manifest_v1_track21_final_test.csv",
    "outputs/manifests/sem_height_manifest_deadline_v1_development.csv",
    "outputs/manifests/sem_height_manifest_v1_track21_final_test.csv",
    "outputs/sem_deadline/sem_mapping_deadline_v1.csv",
    "outputs/sem_final_test_v2/track21_tile14_metadata_compatibility_audit.csv",
    "outputs/sem_final_test_v2/sem_mapping_v1_track21.csv",
    "outputs/sem_final_test_v2/amendment_file_hash.txt",
    "outputs/models/final_track21_height_width_predictions.csv",
    "outputs/models/final_track21_height_width_metrics.csv",
    "outputs/models/final_track21_height_width_cohort_summary.csv",
    "outputs/models/final_track21_height_width_subgroup_metrics.csv",
    "outputs/models/final_track21_historical_human_diagnostics.csv",
    "outputs/models/sem_multimodal_deadline_aggregate_metrics.csv",
    "outputs/models/sem_multimodal_deadline_fold_metrics.csv",
    "outputs/models/thermal_residual_aggregate_metrics.csv",
    "outputs/models/thermal_residual_fold_metrics.csv",
    "outputs/models/thermal_lag_diagnostic_metrics.csv",
    "outputs/reports/track21_tile14_metadata_compatibility_review.md",
    "outputs/reports/generalized_sem_mapping_development_equivalence.md",
    "outputs/reports/generalized_sem_mapping_production_equivalence_v1.md",
    "outputs/reports/final_track21_visualization_summary_v1.md",
    "outputs/reports/final_track21_visualization_validation_v1.json",
    "outputs/reports/final_track21_height_width_test_review_v2.md",
    "outputs/reports/final_track21_height_width_leakage_audit_v2.md",
    "outputs/reports/final_track21_height_width_build_report_v2.md",
    "outputs/reports/final_height_width_model_lock_v1.md",
    "outputs/reports/sem_mapping_provenance_reconciliation.md",
    "outputs/reports/height_quality_gate_freeze_review.md",
    "outputs/reports/thermal_representation_and_alignment_review.md",
    "outputs/reports/thermal_source_and_loader_audit.md",
    "outputs/reports/thermal_distribution_shift_review.md",
    "outputs/reports/thermal_representation_leakage_audit.md",
    "outputs/reports/thermal_representation_build_report.md",
    "outputs/reports/sem_multimodal_deadline_review.md",
    "outputs/reports/sem_multimodal_deadline_leakage_audit.md",
    "outputs/reports/sem_multimodal_deadline_build_report.md",
    "outputs/audits/generalized_sem_mapping_production_equivalence_v1.csv",
    "outputs/audits/generalized_sem_mapping_production_equivalence_hashes_v1.json",
    "outputs/figures/final_track21_height_width_v2/final_track21_predictions_and_errors.png",
    "outputs/figures/final_track21_height_width_v2/final_track21_cohort_b_mae_comparison.png",
    "outputs/figures/final_track21_height_width_v2/final_track21_cohort_b_mae_comparison.pdf",
    "outputs/figures/final_track21_height_width_v2/final_track21_sem_improvement_threshold.png",
    "outputs/figures/final_track21_height_width_v2/final_track21_sem_improvement_threshold.pdf",
    "outputs/figures/final_track21_height_width_v2/final_track21_predicted_vs_target_scatter.png",
    "outputs/figures/final_track21_height_width_v2/final_track21_predicted_vs_target_scatter.pdf",
]

BLOCKED_FILES = {
    "outputs/reports/final_track21_height_width_test_review.md":
        "outputs/reports/archive/first_blocked_attempt/final_track21_height_width_test_review.md",
    "outputs/reports/final_track21_height_width_leakage_audit.md":
        "outputs/reports/archive/first_blocked_attempt/final_track21_height_width_leakage_audit.md",
    "outputs/reports/final_track21_height_width_build_report.md":
        "outputs/reports/archive/first_blocked_attempt/final_track21_height_width_build_report.md",
    "outputs/sem_final_test/track21_sem_source_compatibility.json":
        "outputs/sem_final_test/archive/first_blocked_attempt/track21_sem_source_compatibility.json",
}

TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".cff", ".gitignore"}
RAW_SUMMARIES = ("thermal", "sem", "height", "data/raw", "raw_data")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sanitize(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    root = str(ROOT)
    if root in text:
        text = text.replace(root + "/", "${PROJECT_ROOT}/").replace(root, "${PROJECT_ROOT}")
        changes.append("project-root absolute paths")
    # Raw roots are normally below project root, but handle remaining absolute variants.
    for raw in ("thermal", "sem", "height"):
        candidate = f"${{PROJECT_ROOT}}/{raw}/"
        if candidate in text:
            text = text.replace(candidate, f"${{DATA_ROOT}}/{raw}/")
            changes.append(f"{raw} dataset paths")
    home_pattern = re.compile(r"${USER_HOME}/\\s,\"']+")
    if home_pattern.search(text):
        text = home_pattern.sub("${USER_HOME}", text)
        changes.append("remaining user-home prefixes")
    return text, sorted(set(changes))


def doc_templates() -> dict[str, str]:
    today = date.today().isoformat()
    no_license = (
        "No open-source license has yet been assigned to this repository. Source code "
        "and artifacts are provided for inspection and reproducibility; reuse terms "
        "must be clarified by the repository owner."
    )
    readme = f"""# Leakage-Safe Multimodal Prediction of Local DED Track Width

## Overview

This repository packages the frozen NSF Future Manufacturing Data Challenge workflow
for testing whether Thermal data and leakage-masked SEM context predict Height-derived
local deposited-track width under complete-Track holdout. The dataset is titled
*A Multimodal DED Dataset for Probabilistic Local Geometry Prediction in Laser Tracks*.

## Dataset modalities

The source dataset contains Thermal image sequences, SEM tiles, and post-process Height
maps for SS316L single DED tracks over an approximately 20–100 mm physical interval.
Raw challenge data are not included in this repository.

## Prediction target

`width_mm = y_right_mm - y_left_mm`

`center_mm = (y_left_mm + y_right_mm) / 2`

Height maps provide the target only; Height data are not model inputs.

## Data split

Development uses complete Tracks 8, 10, and 14. Final held-out testing uses Track 21.
Complete-Track holdout avoids leakage from nearby positions on the same physical track
that a random position split could introduce.

## Frozen pipeline

The workflow freezes Height estimator v1, the all-finite Height label policy, a fixed
Thermal shape representation, generalized contiguous N-to-1 SEM mapping, native-coordinate
SEM target masking, model coefficients, ridge alphas, and the project-defined
predeclared confirmation threshold.
There was no Track-21 fitting or model selection.

## Leakage-safe SEM protocol

- Protocol: `symmetric_two_strip_1mm_mask0p4mm_v1`
- Central exclusion: `x ± 0.40 mm`
- Upstream context: `1.00 mm`
- Downstream context: `1.00 mm`

Masking occurs in native SEM coordinates before resizing, filtering, or feature extraction.
The model input is **masked neighboring SEM context**; it is not described as substrate-only.

## Final results

| Model | Track-21 Cohort-B MAE |
|---|---:|
| x-only | 0.111856 mm |
| SEM summary | 0.109837 mm |
| Thermal + SEM | 0.110343 mm |

### Final Track-21 visualizations

#### Predictions along the physical track

![Track-21 predictions and absolute errors](outputs/figures/final_track21_height_width_v2/final_track21_predictions_and_errors.png)

#### Cohort-B MAE comparison

![Track-21 Cohort-B model MAE comparison](outputs/figures/final_track21_height_width_v2/final_track21_cohort_b_mae_comparison.png)

#### SEM improvement versus the project-defined threshold

![SEM improvement compared with the project-defined threshold](outputs/figures/final_track21_height_width_v2/final_track21_sem_improvement_threshold.png)

#### Predicted width versus frozen target

![Track-21 predicted width versus frozen target](outputs/figures/final_track21_height_width_v2/final_track21_predicted_vs_target_scatter.png)

The figures visualize the frozen Track-21 results. They do not represent additional
fitting or post-test model selection.

- SEM absolute improvement: 0.002019 mm
- SEM relative improvement: 1.805%
- Project-defined predeclared confirmation threshold: 0.003356 mm
- Additive Thermal improvement over SEM: -0.000506 mm

The threshold was defined by this project before the held-out Track-21 evaluation to
avoid overinterpreting very small improvements. It was not an official challenge scoring,
eligibility, or submission requirement.

SEM summary reduced the held-out MAE by 0.002019 mm, corresponding to a 1.805%
improvement over x-only. Therefore, the statement “SEM improvement not confirmed” is a
project-specific scientific interpretation and does not mean that the challenge
submission failed.

Under the project-defined predeclared criterion:

**{FINAL_STATUS}**

Masked neighboring SEM context produced a small held-out MAE reduction on Track 21, but
the reduction did not reach the project-defined predeclared confirmation threshold.
Adding the fixed Thermal representation did not improve upon the SEM-only model.
See [Terminology and interpretation](docs/TERMINOLOGY_AND_INTERPRETATION.md).

## Reproducibility and integrity

- 1192 / 1192 development mappings reproduced
- 678 / 678 development Cohort-B IDs reproduced
- 396 / 396 Track-21 mappings reproduced
- 233 / 233 Track-21 Cohort-B IDs reproduced
- Maximum prediction reproduction difference: 0.0
- Immutable files unchanged
- Production-equivalence audit passed

## Repository structure

`scripts/` contains the frozen execution and audit code; `src/` contains its local
helpers; `configs/` contains frozen protocols and locks; `outputs/` contains compact
published artifacts; `docs/` explains workflow, leakage controls, lock, and provenance.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.

## Data setup

Obtain authorized access independently, then place the source files at the repository
root using the layouts described in [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md):
`${{DATA_ROOT}}/thermal/`, `${{DATA_ROOT}}/sem/`, and `${{DATA_ROOT}}/height/`.
For unchanged scripts, `${{DATA_ROOT}}` is the repository root.

## Reproduction commands

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The final stages are:

```bash
python3 scripts/prepare_final_test_sem_compatibility_amendment_v1.py
python3 scripts/run_final_track21_height_width_test_v2.py
MPLCONFIGDIR=/tmp/nsf_mplconfig python3 scripts/audit_generalized_sem_mapping_production_equivalence_v1.py
```

These require the original raw data and the preceding frozen artifacts. Earlier
development stages are documented in the reproduction guide.

## Expected checks

The model-lock hash is `{LOCK_HASH}`. The amendment hash is `{AMEND_HASH}`.
The development SEM-summary hash is
`4f8473e69b6c05d0049c920676b6cc0891b708bca176c86fd6acc51815033203`.
The production-equivalence terminal status is
`GENERALIZED SEM PRODUCTION IMPLEMENTATION EQUIVALENCE PASS`.

## Limitations

Automatic Height labels contain local measurement noise; SEM mapping is a low-confidence
full-mosaic endpoint-linear fallback; single-native-tile context reduces coverage; the
SEM gain was below the project-defined threshold; Thermal lacked stable additive value; and only a
small number of complete Tracks were available. See [LIMITATIONS.md](LIMITATIONS.md).

## Citation

See [CITATION.cff](CITATION.cff). Its author placeholder must be completed by the
repository owner before publication.

## License

{no_license}
"""
    results = f"""# Results

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

**{FINAL_STATUS}**

Masked neighboring SEM context produced a small held-out MAE reduction, but the
predeclared confirmatory criterion was not met. This does not establish that SEM has no
effect or performs worse than x-only. Thermal and multimodal benefit are not confirmed.
This conclusion concerns confirmation under the project's conservative criterion. It
does not determine the official challenge ranking or submission status.
"""
    reproducibility = f"""# Reproducibility

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
| 6. Model lock | `python3 scripts/freeze_final_height_width_model_v1.py` | development artifacts | lock hash `{LOCK_HASH}` | no Track 21 | final development fit only |
| 7. Tile-14 amendment | `python3 scripts/prepare_final_test_sem_compatibility_amendment_v1.py` | filenames and TIFF metadata | amendment hash `{AMEND_HASH}` | metadata only; no pixels/labels | no |
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
"""
    data = """# Data availability

Raw challenge data are not redistributed. Users must independently obtain authorized
access; this release does not provide or invent an external download URL or legal terms.

Expected repository-root layout:

```text
thermal/
  Thermal_8.mat
  Thermal_10.mat
  Thermal_14.mat
  Thermal_21.mat
sem/
  SEM_8/PlainImages/*.tif
  SEM_10/PlainImages/*.tif
  SEM_14/PlainImages/*.tif
  SEM_21/PlainImages/*.tif
height/
  [original Track 8, 10, 14, and 21 Height-map files in the source layout]
```

The exact Height filenames and nested paths are resolved by `nsf_fmrg_data.py` and the
packaged Height data-access helpers. Preserve the source dataset layout rather than
renaming files. Expected source types are Thermal `.mat`, SEM `.tif`, and Height-map
ASCII data.

Compact frozen labels, mappings, manifests, model JSON files, metrics, reports, audits,
and the final figure are included. They can be inspected without raw data. Raw-dependent
features and evaluations can be regenerated after data placement. TIFF and MAT payloads
are excluded because they are large source data and are not authorized for redistribution.
"""
    limitations = """# Limitations

- Height estimator uncertainty creates local target noise.
- The all-finite policy treats every finite automatic estimate as the frozen target.
- Evidence comes from a limited number of complete Tracks.
- SEM uses a low-confidence full-mosaic endpoint-linear mapping fallback.
- Adjacent SEM tiles were not registered with high-precision overlap alignment.
- Requiring complete context within one native tile reduces SEM coverage.
- Neighboring deposited-track information remains visible outside the central mask.
- The small SEM gain was below the project-defined predeclared confirmation threshold.
- The fixed Thermal representation was not stable across development folds.
- Results do not establish causal prediction.
- The model is not claimed to reconstruct rapid local target fluctuations.
- Targets are frozen automatic estimates rather than manually validated width everywhere.

No post-hoc Track-21 tuning is proposed or permitted by the frozen protocol.
"""
    workflow = """# Project workflow

```text
Height target development
→ Height estimator freeze
→ Thermal baseline and representation audit
→ SEM source mapping
→ leakage-safe native masking
→ complete-Track LOTO development modeling
→ final model lock
→ Track-21 metadata-only compatibility amendment
→ one-time Track-21 final test
→ production-implementation equivalence audit
→ frozen final interpretation
```

Height supplies labels only. Thermal and masked neighboring SEM context are candidate
inputs. Development decisions use Tracks 8/10/14. Track 21 remains held out until the
locked one-time final test. The equivalence audit reproduces implementation behavior
without retraining or revising the interpretation.

The project-defined predeclared confirmation threshold was frozen before Track 21 and
was an internal scientific interpretation criterion, not an official challenge rule.
"""
    leakage = """# Leakage controls

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
"""
    lockdoc = f"""# Model lock and final test

The final lock fixes target policy, Thermal representation, SEM mapping and mask,
features, model families, coefficients, ridge alphas, cohorts, and the project-defined
predeclared confirmation criterion.
Its immutable SHA-256 is `{LOCK_HASH}`.

Track 21 remained untouched during development. The first attempted runner stopped
before a model test because it detected a source-schema incompatibility: Track 21 had
canonical Tile 14 while development Tracks had 13 tiles. No held-out pixels, labels,
predictions, metrics, or human references had been loaded, so this was not a failed
model test.

A metadata-only review established that Tile 14 was a canonical contiguous-sequence
member. The compatibility amendment generalized the already documented N-to-1 rule
without changing the scientific model. Its SHA-256 is `{AMEND_HASH}`. Only after that
amendment was the one-time final test executed. No post-test tuning is permitted.

The criterion `max(0.003 mm, 3% of x-only MAE)` was defined by this project before the
final evaluation for conservative interpretation. It was not an official challenge
scoring, eligibility, or submission requirement. See
[Terminology and interpretation](TERMINOLOGY_AND_INTERPRETATION.md).

The frozen interpretation remains: **{FINAL_STATUS}**.
"""
    notes = f"""# Release notes

## Version 1.0.0 — {today}

- Public packaging of the completed frozen workflow and compact scientific artifacts.
- Added three publication-ready frozen-result visualizations in PNG and PDF, plus their
  source-hash validation and visualization summary.
- Clarified that the 0.003 mm / 3% criterion is project-defined and predeclared, not an
  official challenge scoring, eligibility, or submission requirement.
- No model retraining, selection, scientific algorithm change, or result reinterpretation.
- Absolute local paths are sanitized only in release copies and documented in provenance.
- First blocked-attempt evidence is relocated to explicit archive directories. The release
  audit script receives a path-only fallback so it can verify those archived hashes.
- Direct runtime dependencies were derived from imports in included scripts. Installed
  versions were recorded when reliably available; no global `pip freeze` was copied.
- `CITATION.cff` contains an editable repository-owner placeholder because complete,
  publication-ready software-author metadata was not available. Replace it before publication.
- No open-source license was found or invented.
"""
    citation = f"""cff-version: 1.2.0
message: "If you use this software, please cite it using the metadata below."
title: "Leakage-Safe Multimodal Prediction of Local DED Track Width"
type: software
version: 1.0.0
date-released: "{today}"
authors:
  - family-names: "OWNER PLACEHOLDER"
    given-names: "EDIT BEFORE PUBLICATION"
"""
    archive = """# Archived development code and evidence

Files placed in archive directories document historical or blocked development states.
They are not part of the final execution path. In particular, the first blocked Track-21
attempt was a source-schema stop before held-out model evaluation, not a failed model test.
"""
    tests = """# Tests

The original research workspace contains a larger regression suite that is not duplicated
here. Release validation compiles every packaged script, parses JSON/YAML/CSV files, checks
Markdown links, scans privacy/secrets, verifies critical hashes, and checks file sizes.

After raw data placement, run the production-equivalence audit documented in
`../REPRODUCIBILITY.md`.
"""
    return {
        "README.md": readme, "RESULTS.md": results, "REPRODUCIBILITY.md": reproducibility,
        "DATA_AVAILABILITY.md": data, "LIMITATIONS.md": limitations,
        "RELEASE_NOTES.md": notes, "CITATION.cff": citation,
        "docs/PROJECT_WORKFLOW.md": workflow, "docs/LEAKAGE_CONTROLS.md": leakage,
        "docs/MODEL_LOCK_AND_FINAL_TEST.md": lockdoc,
        "scripts/archive/README.md": archive, "tests/README.md": tests,
    }


def requirements() -> str:
    distributions = [
        ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
        ("Pillow", "Pillow"), ("matplotlib", "matplotlib"),
        ("opencv-python", "opencv-python"), ("PyYAML", "PyYAML"),
    ]
    lines = []
    for package, distribution in distributions:
        try:
            lines.append(f"{package}=={importlib.metadata.version(distribution)}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(package)
    return "\n".join(lines) + "\n"


def gitignore() -> str:
    return """.venv/
venv/
env/
__pycache__/
*.py[cod]
.DS_Store
.ipynb_checkpoints/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
*.log
*.tmp
*.bak
thermal/
sem/
height/
data/raw/
raw_data/
"""


def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, text=True, capture_output=True)


def build(output: Path, overwrite: bool, skip_git: bool) -> int:
    if output.exists():
        if not overwrite:
            raise RuntimeError(f"Release already exists; use --overwrite-release: {output}")
        if ROOT not in output.parents:
            raise RuntimeError("Refusing to replace an output outside the project root")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    if sha256(ROOT / "configs/final_height_width_model_lock_v1.yaml") != LOCK_HASH:
        raise RuntimeError("Original model-lock hash mismatch")
    if sha256(ROOT / "configs/final_test_sem_source_compatibility_amendment_v1.yaml") != AMEND_HASH:
        raise RuntimeError("Original compatibility-amendment hash mismatch")

    selected = CORE_FILES + CONFIG_FILES + OUTPUT_FILES
    terminology_audit = "outputs/reports/threshold_terminology_consistency_audit_v1.md"
    if (ROOT / terminology_audit).is_file():
        selected.append(terminology_audit)
    selected += [
        str(p.relative_to(ROOT))
        for p in sorted((ROOT / "outputs/models/final_height_width_v1").glob("*"))
        if p.is_file()
    ]
    provenance: list[dict] = []
    sanitized_count = 0

    def copy_one(source_rel: str, release_rel: str | None = None) -> None:
        nonlocal sanitized_count
        src = ROOT / source_rel
        if not src.is_file() or src.is_symlink():
            raise RuntimeError(f"Required source missing or symlinked: {source_rel}")
        dst_rel = release_rel or source_rel
        dst = output / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        original_hash = sha256(src)
        changes: list[str] = []
        if src.suffix.lower() in TEXT_SUFFIXES or src.name == ".gitignore":
            raw = src.read_text(encoding="utf-8")
            public, changes = text_sanitize(raw)
            dst.write_text(public, encoding="utf-8")
        else:
            shutil.copy2(src, dst, follow_symlinks=False)
        if changes:
            sanitized_count += 1
        provenance.append({
            "release_path": dst_rel, "source_project_path": source_rel,
            "category": dst_rel.split("/", 1)[0], "included": True,
            "exclusion_reason": "", "file_size_bytes": dst.stat().st_size,
            "original_sha256": original_hash, "release_sha256": sha256(dst),
            "path_sanitized": bool(changes), "contains_generated_results": source_rel.startswith("outputs/"),
            "required_for_reproduction": source_rel.startswith(("scripts/", "src/", "configs/"))
                or source_rel == "nsf_fmrg_data.py",
            "notes": "; ".join(changes),
        })

    for rel in sorted(set(selected)):
        copy_one(rel)
    for source_rel, release_rel in BLOCKED_FILES.items():
        copy_one(source_rel, release_rel)

    # Release-only path fallback: preserve archived evidence while keeping immutable hashes.
    audit = output / "scripts/audit_generalized_sem_mapping_production_equivalence_v1.py"
    audit_text = audit.read_text()
    old = 'p += [ROOT/x for x in amendment["first_blocked_attempt_paths"]]'
    new = '''\n def blocked_path(x):\n  direct=ROOT/x\n  if direct.exists():return direct\n  name=Path(x).name\n  archive=(ROOT/"outputs/reports/archive/first_blocked_attempt"/name if x.startswith("outputs/reports/") else\n           ROOT/"outputs/sem_final_test/archive/first_blocked_attempt"/name)\n  return archive\n p += [blocked_path(x) for x in amendment["first_blocked_attempt_paths"]]'''
    if old not in audit_text:
        raise RuntimeError("Could not apply release-only archived-evidence path fallback")
    audit.write_text(audit_text.replace(old, new))
    for row in provenance:
        if row["release_path"] == str(audit.relative_to(output)):
            row["release_sha256"] = sha256(audit)
            row["notes"] = "release-only archived-evidence path fallback; scientific algorithm unchanged"

    for rel, content in doc_templates().items():
        dst = output / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content, encoding="utf-8")
    (output / "requirements.txt").write_text(requirements())
    (output / ".gitignore").write_text(gitignore())

    # Provenance is generated after all copied-file sanitization decisions are known.
    prov_lines = [
        "# File provenance", "",
        "Original files were copied, never moved. SHA-256 values below distinguish original",
        "scientific artifacts from path-sanitized public copies.", "",
        "| Original project path | Release path | Original SHA-256 | Release SHA-256 | Sanitized | Sanitized fields/lines |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(provenance, key=lambda x: x["release_path"]):
        prov_lines.append(
            f"| `{row['source_project_path']}` | `{row['release_path']}` | "
            f"`{row['original_sha256']}` | `{row['release_sha256']}` | "
            f"{row['path_sanitized']} | {row['notes'] or 'none'} |"
        )
    (output / "docs/FILE_PROVENANCE.md").write_text("\n".join(prov_lines) + "\n")

    generated = [
        "README.md", "RESULTS.md", "REPRODUCIBILITY.md", "DATA_AVAILABILITY.md",
        "LIMITATIONS.md", "RELEASE_NOTES.md", "CITATION.cff", "requirements.txt",
        ".gitignore", "docs/PROJECT_WORKFLOW.md", "docs/LEAKAGE_CONTROLS.md",
        "docs/MODEL_LOCK_AND_FINAL_TEST.md", "docs/FILE_PROVENANCE.md",
        "scripts/archive/README.md", "tests/README.md",
    ]
    for rel in generated:
        p = output / rel
        provenance.append({
            "release_path": rel, "source_project_path": "", "category": rel.split("/", 1)[0],
            "included": True, "exclusion_reason": "", "file_size_bytes": p.stat().st_size,
            "original_sha256": "", "release_sha256": sha256(p), "path_sanitized": False,
            "contains_generated_results": False, "required_for_reproduction": rel in {
                "README.md", "REPRODUCIBILITY.md", "requirements.txt"
            }, "notes": "generated public-release documentation",
        })
    for rel in RAW_SUMMARIES:
        p = ROOT / rel
        provenance.append({
            "release_path": rel + "/", "source_project_path": rel + "/",
            "category": "excluded_raw_data", "included": False,
            "exclusion_reason": "raw challenge data not redistributed" if p.exists() else
                "raw-data location excluded by policy (not present under this name)",
            "file_size_bytes": "", "original_sha256": "", "release_sha256": "",
            "path_sanitized": False, "contains_generated_results": False,
            "required_for_reproduction": True, "notes": "one summary row for excluded directory",
        })

    manifest_path = output / "FILE_MANIFEST.csv"
    fieldnames = [
        "release_path", "source_project_path", "category", "included", "exclusion_reason",
        "file_size_bytes", "original_sha256", "release_sha256", "path_sanitized",
        "contains_generated_results", "required_for_reproduction", "notes",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(provenance, key=lambda x: (str(x["release_path"]), not x["included"])))

    # Structural validation.
    checks: dict[str, tuple[bool, str]] = {}
    files = [p for p in output.rglob("*") if p.is_file() and ".git" not in p.parts]
    symlinks = [p for p in output.rglob("*") if p.is_symlink()]
    largest = max(files, key=lambda p: p.stat().st_size)
    over50 = [p for p in files if p.stat().st_size > 50 * 1024 * 1024]
    over100 = [p for p in files if p.stat().st_size > 100 * 1024 * 1024]
    raw_payload = [p for p in files if p.suffix.lower() in {".tif", ".tiff", ".mat", ".asc"}]
    checks["model-lock hash"] = (sha256(output / "configs/final_height_width_model_lock_v1.yaml") == LOCK_HASH, LOCK_HASH)
    checks["amendment hash"] = (sha256(output / "configs/final_test_sem_source_compatibility_amendment_v1.yaml") == AMEND_HASH, AMEND_HASH)
    checks["no file above 100 MB"] = (not over100, ", ".join(str(p.relative_to(output)) for p in over100) or "none")
    checks["no raw TIFF/MAT/ASC"] = (not raw_payload, ", ".join(str(p.relative_to(output)) for p in raw_payload) or "none")
    checks["no symlinks"] = (not symlinks, ", ".join(str(p.relative_to(output)) for p in symlinks) or "none")

    compile_result = run([sys.executable, "-m", "compileall", "-q", "scripts", "src", "nsf_fmrg_data.py"], output)
    checks["scripts compile"] = (compile_result.returncode == 0, compile_result.stderr.strip() or "compileall succeeded")
    # Compilation is a validation step; bytecode/cache files are never release content.
    for cache in sorted(output.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache)
    for bytecode in output.rglob("*.py[co]"):
        bytecode.unlink()
    parse_errors = []
    for p in files:
        try:
            if p.suffix == ".json":
                json.loads(p.read_text())
            elif p.suffix in {".yaml", ".yml"}:
                try:
                    json.loads(p.read_text())
                except json.JSONDecodeError:
                    # PyYAML is optional in the packaging environment. Ruby's standard
                    # Psych parser provides an independent syntax check when unavailable.
                    yaml_check = run(
                        ["ruby", "-e", "require 'yaml'; YAML.load_file(ARGV[0])", str(p)],
                        output,
                    )
                    if yaml_check.returncode:
                        raise ValueError(yaml_check.stderr.strip() or "YAML parsing failed")
            elif p.suffix == ".csv":
                with p.open(newline="", encoding="utf-8") as stream:
                    if not next(csv.reader(stream), []):
                        raise ValueError("empty CSV header")
            elif p.suffix.lower() in {".md", ".txt", ".cff", ".py"}:
                p.read_text(encoding="utf-8")
        except Exception as exc:
            parse_errors.append(f"{p.relative_to(output)}: {exc}")
    checks["structured files parse"] = (not parse_errors, "; ".join(parse_errors) or "JSON/YAML/CSV/text readable")

    absolute_hits = []
    secret_hits = []
    secret_pattern = re.compile(
        r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|password|private[_-]?key)"
        r"\\s*[:=]\\s*[\"']?[A-Za-z0-9_+/.=-]{12,}"
    )
    for p in files:
        if p.suffix.lower() not in TEXT_SUFFIXES and p.name != ".gitignore":
            continue
        text = p.read_text(errors="replace")
        if re.search(r"${USER_HOME}/\\s,\"']+", text):
            absolute_hits.append(str(p.relative_to(output)))
        if secret_pattern.search(text):
            secret_hits.append(str(p.relative_to(output)))
    checks["no absolute personal paths"] = (not absolute_hits, ", ".join(absolute_hits) or "none")
    checks["privacy/secret scan"] = (not secret_hits, ", ".join(secret_hits) or "no obvious credentials")

    link_errors = []
    link_re = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for p in files:
        if p.suffix.lower() != ".md":
            continue
        for target in link_re.findall(p.read_text()):
            target = target.strip().split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (p.parent / target).resolve().exists():
                link_errors.append(f"{p.relative_to(output)} -> {target}")
    checks["documentation links"] = (not link_errors, "; ".join(link_errors) or "all relative links resolve")
    status_preserved = FINAL_STATUS in (output / "README.md").read_text() and FINAL_STATUS in (output / "RESULTS.md").read_text()
    checks["immutable scientific result"] = (status_preserved, FINAL_STATUS)

    critical_ok = all(value[0] for value in checks.values())
    validation = ["# Release validation", ""]
    for name, (passed, detail) in checks.items():
        validation += [f"## {'PASS' if passed else 'FAIL'} — {name}", "", detail or "No details.", ""]
    validation += [
        "## Size advisory", "",
        f"Largest included file: `{largest.relative_to(output)}` ({largest.stat().st_size} bytes).",
        f"Files above 50 MB: {', '.join(str(p.relative_to(output)) for p in over50) or 'none'}.", "",
        "Data-dependent stages were not executed because raw data are intentionally absent.", "",
    ]
    (output / "RELEASE_VALIDATION.md").write_text("\n".join(validation))

    # Manifest now includes validation; self rows intentionally omit self-referential hashes.
    with manifest_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        for rel in ("FILE_MANIFEST.csv", "SHA256SUMS.txt", "RELEASE_VALIDATION.md", "RELEASE_PACKAGING_SUMMARY.txt"):
            p = output / rel
            writer.writerow({
                "release_path": rel, "source_project_path": "", "category": "release_metadata",
                "included": True, "exclusion_reason": "",
                "file_size_bytes": p.stat().st_size if p.exists() else "",
                "original_sha256": "", "release_sha256": sha256(p) if p.exists() and rel == "RELEASE_VALIDATION.md" else "",
                "path_sanitized": False, "contains_generated_results": False,
                "required_for_reproduction": True,
                "notes": "generated metadata; self-referential hash omitted where applicable",
            })

    def write_sums() -> None:
        current = sorted(
            (p for p in output.rglob("*") if p.is_file() and ".git" not in p.parts
             and p.name != "SHA256SUMS.txt"),
            key=lambda p: p.relative_to(output).as_posix(),
        )
        (output / "SHA256SUMS.txt").write_text(
            "".join(f"{sha256(p)}  {p.relative_to(output).as_posix()}\n" for p in current)
        )

    git_status = "skipped by --skip-git-init"
    commit_after_metadata = False
    if critical_ok and not skip_git and shutil.which("git"):
        init = run(["git", "init", "-b", "main"], output)
        if init.returncode == 0:
            stage = run(["git", "add", "."], output)
            if stage.returncode == 0:
                name = run(["git", "config", "user.name"], output).stdout.strip()
                email = run(["git", "config", "user.email"], output).stdout.strip()
                git_status = "initialized on main; all files staged"
                if name and email:
                    commit_after_metadata = True
                else:
                    git_status += "; no commit because Git identity is not configured"
            else:
                git_status = f"initialization succeeded but staging failed: {stage.stderr.strip()}"
        else:
            git_status = f"initialization failed: {init.stderr.strip()}"
    elif critical_ok and not skip_git:
        git_status = "Git unavailable; repository not initialized"

    # Git changes do not affect release files, but summary and sums are finalized afterward.
    files = [p for p in output.rglob("*") if p.is_file() and ".git" not in p.parts]
    total_size = sum(p.stat().st_size for p in files)
    overall = "GITHUB RELEASE PACKAGE READY" if critical_ok else "GITHUB RELEASE PACKAGE FAILED VALIDATION"
    try:
        public_output = "${PROJECT_ROOT}/" + output.relative_to(ROOT).as_posix()
    except ValueError:
        public_output = "${USER_HOME}/" + output.name
    summary = f"""release directory: {public_output}
total included file count: {len(files) + 2}
total release size: {total_size} bytes
excluded raw-data directories: {len(RAW_SUMMARIES)}
sanitized files: {sanitized_count}
model-lock hash verification: {'PASS' if checks['model-lock hash'][0] else 'FAIL'}
amendment hash verification: {'PASS' if checks['amendment hash'][0] else 'FAIL'}
largest included file: {largest.relative_to(output)} ({largest.stat().st_size} bytes)
files above 50 MB: {', '.join(str(p.relative_to(output)) for p in over50) or 'none'}
scripts compile status: {'PASS' if checks['scripts compile'][0] else 'FAIL'}
documentation-link status: {'PASS' if checks['documentation links'][0] else 'FAIL'}
privacy/secret scan status: {'PASS' if checks['privacy/secret scan'][0] and checks['no absolute personal paths'][0] else 'FAIL'}
immutable scientific result status: {FINAL_STATUS}
Git initialization status: {git_status}
overall packaging status: {overall}
"""
    (output / "RELEASE_PACKAGING_SUMMARY.txt").write_text(summary)
    write_sums()
    if critical_ok and not skip_git and (output / ".git").exists():
        run(["git", "add", "."], output)
        if commit_after_metadata:
            commit = run(["git", "commit", "-m", "Release v1.0.0"], output)
            git_status = (
                "initialized on main; all files staged; commit created"
                if commit.returncode == 0
                else "initialized on main; files staged; commit not created"
            )
            # Keep the persisted summary synchronized with the final Git state.
            summary = re.sub(
                r"Git initialization status:.*",
                f"Git initialization status: {git_status}",
                summary,
            )
            (output / "RELEASE_PACKAGING_SUMMARY.txt").write_text(summary)
            write_sums()
            run(["git", "add", "."], output)
            if commit.returncode == 0:
                amend = run(["git", "commit", "--amend", "--no-edit"], output)
                if amend.returncode != 0:
                    git_status += "; final metadata remains staged"

    print(summary, end="")
    print("repository tree:")
    for p in sorted(output.rglob("*"), key=lambda x: x.relative_to(output).as_posix()):
        rel = p.relative_to(output)
        if ".git" in rel.parts or "__pycache__" in rel.parts:
            continue
        if len(rel.parts) <= 4:
            print(f"{'  ' * (len(rel.parts)-1)}{rel.name}{'/' if p.is_dir() else ''}")
    return 0 if critical_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite-release", action="store_true")
    parser.add_argument("--skip-git-init", action="store_true")
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    return build(output.resolve(), args.overwrite_release, args.skip_git_init)


if __name__ == "__main__":
    raise SystemExit(main())
