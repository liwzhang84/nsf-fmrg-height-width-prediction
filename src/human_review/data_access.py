"""Project-relative, official-coordinate data access for the review app."""
from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.official_coordinates import get_height_indices_in_window

PROJECT = Path(__file__).resolve().parents[2]
PILOTS = {
    "height_pilot_v2": PROJECT / "outputs/human_review/height_review_pilot_selection_v2.csv",
    "height_pilot_v1": PROJECT / "outputs/human_review/height_review_pilot_selection.csv",
}
REPEAT_SELECTION = PROJECT / "outputs/human_review/height_review_repeat_1_selection.csv"
CANDIDATES = PROJECT / "outputs/height_labels/candidates/height_label_candidates.csv"
THERMAL = PROJECT / "outputs/mappings/thermal_frame_mapping_official.csv"
SUPPORT = PROJECT / "outputs/mappings/thermal_height_support.csv"
MANIFEST = PROJECT / "outputs/mappings/master_manifest_provisional.csv"
LEVELED = PROJECT / "outputs/height_processing/leveled"
METHODS = ("official_robust_plane_v1", "local_side_linear_v1", "outside_quadratic_v1")


def finite_number(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def candidate_displayable_on(row, method):
    return (row.get("leveling_method_id") == method
            and finite_number(row.get("y_left_mm"))
            and finite_number(row.get("y_right_mm")))


def _csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_selection(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT / path
    return _csv(path)


@lru_cache(maxsize=4)
def pilot_rows(version="height_pilot_v2", review_round="pilot_1"):
    if review_round == "repeat_1":
        return _csv(REPEAT_SELECTION)
    return _csv(PILOTS[version])


@lru_cache(maxsize=1)
def candidate_rows():
    return _csv(CANDIDATES)


@lru_cache(maxsize=16)
def height_data(track, method):
    path = LEVELED / f"Track_{int(track)}_{method}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def profile(track, x_mm, half_width_mm, method):
    data = height_data(int(track), "official_robust_plane_v1" if method == "raw" else method)
    query = get_height_indices_in_window(int(track), float(x_mm), float(half_width_mm))
    z = data["Z_raw_mm"] if method == "raw" else data["Z_leveled_mm"]
    values = np.nanmedian(z[:, query.index_start:query.index_end], axis=1)
    return data["y_mm"], values, query


def candidates_for(track, x_mm):
    return [
        row for row in candidate_rows()
        if int(row["track_id"]) == int(track)
        and abs(float(row["thermal_x_center_mm"]) - float(x_mm)) < 1e-6
    ]


@lru_cache(maxsize=1)
def _official_neighbor_tables():
    thermal = {(int(r["track_id"]), int(r["segment_frame_index"])): r for r in _csv(THERMAL)}
    support = {(int(r["track_id"]), int(r["segment_frame_index"])): r for r in _csv(SUPPORT)}
    manifest = {(int(r["track_id"]), int(r["segment_frame_index"])): r for r in _csv(MANIFEST)}
    return thermal, support, manifest


def official_neighbors(track, segment_frame_index):
    """Return current ±1 official frames, with explicit Height query status."""
    thermal, support, manifest = _official_neighbor_tables()
    out = []
    for segment in (int(segment_frame_index)-1, int(segment_frame_index), int(segment_frame_index)+1):
        key = (int(track), segment)
        t, s, m = thermal.get(key), support.get(key), manifest.get(key)
        item = {
            "track_id": int(track), "segment_frame_index": segment,
            "sample_id": m["sample_id"] if m else "",
            "x_actual_mm": float(t["x_mm_center"]) if t else np.nan,
            "height_valid_fraction": float(m["height_valid_fraction"]) if m else np.nan,
            "query_status": "missing_official_frame",
        }
        if t and s and s["inside_height_coverage"] == "True":
            try:
                get_height_indices_in_window(int(track), float(t["x_mm_center"]), 0.0)
                item["query_status"] = "queryable"
            except ValueError:
                item["query_status"] = "outside_height_query_support"
        elif t:
            item["query_status"] = "outside_height_coverage"
        out.append(item)
    return out
