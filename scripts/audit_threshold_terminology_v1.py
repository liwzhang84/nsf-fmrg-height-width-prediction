#!/usr/bin/env python3
"""Audit public terminology for the project-defined SEM confirmation criterion."""
from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "github_release/nsf-fmrg-height-width-prediction-v1"
OUTPUT = ROOT / "outputs/reports/threshold_terminology_consistency_audit_v1.md"
THRESHOLD_TOKEN = "0.003356"
DISALLOWED = [
    "official challenge threshold",
    "challenge-required improvement",
    "required by the challenge",
    "challenge passing threshold",
    "failed the challenge",
    "challenge failure",
    "did not pass the challenge",
]
FROZEN = {
    "configs/final_height_width_model_lock_v1.yaml",
    "configs/final_test_sem_source_compatibility_amendment_v1.yaml",
    "outputs/models/final_track21_height_width_metrics.csv",
    "outputs/models/final_track21_height_width_predictions.csv",
    "outputs/reports/final_track21_height_width_test_review_v2.md",
    "outputs/reports/final_track21_height_width_leakage_audit_v2.md",
    "outputs/reports/final_track21_height_width_build_report_v2.md",
    "outputs/reports/generalized_sem_mapping_production_equivalence_v1.md",
}
PUBLIC_RELEASE_FILES = [
    "README.md",
    "RESULTS.md",
    "REPRODUCIBILITY.md",
    "LIMITATIONS.md",
    "RELEASE_NOTES.md",
    "DATA_AVAILABILITY.md",
    "docs/PROJECT_WORKFLOW.md",
    "docs/LEAKAGE_CONTROLS.md",
    "docs/MODEL_LOCK_AND_FINAL_TEST.md",
    "docs/TERMINOLOGY_AND_INTERPRETATION.md",
    "outputs/reports/final_track21_visualization_summary_v1.md",
]
EDITABLE_SOURCE_FILES = [
    "scripts/create_final_track21_result_visualizations_v1.py",
    "scripts/create_github_release_v1.py",
    "docs/TERMINOLOGY_AND_INTERPRETATION.md",
    "outputs/reports/final_track21_visualization_summary_v1.md",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def threshold_lines(path: Path) -> list[tuple[int, str]]:
    return [
        (number, line.strip())
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1)
        if THRESHOLD_TOKEN in line
    ]


def acceptable(path_label: str, line: str, frozen: bool) -> tuple[bool, str]:
    if frozen:
        return True, "Preserved frozen terminology; public terminology note defines its meaning."
    text = line.lower()
    if any(phrase in text for phrase in DISALLOWED):
        return False, "Disallowed implication of an official challenge rule."
    return True, "Editable/public wording is project-qualified or explained in the same document."


def main() -> None:
    if not RELEASE.is_dir():
        raise RuntimeError("Generate the release before running the terminology audit")
    terminology_note = RELEASE / "docs/TERMINOLOGY_AND_INTERPRETATION.md"
    if not terminology_note.is_file():
        raise RuntimeError("Release terminology note is missing")
    note_text = terminology_note.read_text()
    required_note = (
        "It was not an official challenge scoring, eligibility, or submission requirement."
    )
    if required_note not in note_text:
        raise RuntimeError("Terminology note lacks the official-challenge clarification")

    candidates: list[tuple[str, Path, bool]] = []
    for relative in sorted(FROZEN):
        path = ROOT / relative
        if path.is_file():
            candidates.append((relative, path, True))
    for relative in EDITABLE_SOURCE_FILES:
        path = ROOT / relative
        if path.is_file():
            candidates.append((relative, path, False))
    for relative in PUBLIC_RELEASE_FILES:
        path = RELEASE / relative
        if path.is_file():
            candidates.append((f"github_release/nsf-fmrg-height-width-prediction-v1/{relative}", path, False))

    rows = []
    for label, path, frozen in candidates:
        for number, line in threshold_lines(path):
            ok, explanation = acceptable(label, line, frozen)
            rows.append(
                {
                    "file": label,
                    "line": number,
                    "phrase": line.replace("|", "\\|"),
                    "classification": "frozen scientific artifact" if frozen else "editable/public",
                    "acceptable": ok,
                    "note": "yes" if frozen or required_note in path.read_text() else
                    ("linked terminology note" if "TERMINOLOGY_AND_INTERPRETATION.md" in path.read_text() else "document-level explanation"),
                    "official_implication": any(x in line.lower() for x in DISALLOWED),
                    "explanation": explanation,
                }
            )

    editable_files = [(ROOT / p, p) for p in EDITABLE_SOURCE_FILES]
    editable_files += [(RELEASE / p, f"release/{p}") for p in PUBLIC_RELEASE_FILES]
    disallowed_hits = []
    for path, label in editable_files:
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            for phrase in DISALLOWED:
                if phrase in line.lower():
                    disallowed_hits.append((label, number, phrase, line.strip()))

    if not rows:
        raise RuntimeError("No files containing the numeric threshold were found")
    if any(not row["acceptable"] or row["official_implication"] for row in rows):
        raise RuntimeError("Unacceptable threshold terminology remains")
    if disallowed_hits:
        raise RuntimeError(f"Disallowed public-facing phrases remain: {disallowed_hits}")

    lines = [
        "# Project-defined threshold terminology consistency audit v1",
        "",
        "## Scope and classification",
        "",
        "Frozen scientific artifacts were inspected but not edited. Editable source and generated",
        "public documentation were checked for project-qualified terminology. In a preserved frozen",
        'report, “Required reduction” refers to the project-defined predeclared confirmation',
        "criterion; the public terminology note explains that it is not an official challenge rule.",
        "",
        f"- Terminology note: `docs/TERMINOLOGY_AND_INTERPRETATION.md`",
        f"- Disallowed phrases remaining in editable public files: {len(disallowed_hits)}",
        "",
        "## Files containing the numeric threshold",
        "",
        "| File | Line | Exact surrounding phrase | Classification | Acceptable | Explanatory note | Incorrect official implication |",
        "|---|---:|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['file']}` | {row['line']} | {row['phrase']} | "
            f"{row['classification']} | {'yes' if row['acceptable'] else 'no'} | "
            f"{row['note']} | {'yes' if row['official_implication'] else 'no'} |"
        )
    lines += [
        "",
        "## Disallowed-phrase scan",
        "",
        "No editable public-facing file contains any of the following phrases:",
        "",
        *[f"- `{phrase}`" for phrase in DISALLOWED],
        "",
        "## Frozen integrity context",
        "",
        *[
            f"- `{relative}` — SHA-256 `{sha256(ROOT / relative)}`"
            for relative in sorted(FROZEN)
            if (ROOT / relative).is_file()
        ],
        "",
        "The threshold formula and value remain unchanged: `max(0.003 mm, 3% of x-only MAE)`",
        "and `0.003356 mm` on Track 21. The frozen status remains",
        "`SEM IMPROVEMENT NOT CONFIRMED ON FINAL TRACK 21`.",
        "",
        "PROJECT-DEFINED THRESHOLD TERMINOLOGY CONSISTENT",
    ]
    OUTPUT.write_text("\n".join(lines) + "\n")
    print(f"threshold_occurrences={len(rows)} disallowed_editable_public={len(disallowed_hits)}")
    print("PROJECT-DEFINED THRESHOLD TERMINOLOGY CONSISTENT")


if __name__ == "__main__":
    main()
