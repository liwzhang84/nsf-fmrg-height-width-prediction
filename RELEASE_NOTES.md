# Release notes

## Version 1.0.0 — 2026-07-25

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
