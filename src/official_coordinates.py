"""Typed, read-only access to the frozen official coordinate mapping CSVs.

New processing code must query these CSVs instead of reconstructing Thermal or
Height coordinates. Queries never extrapolate beyond measured support.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
THERMAL_MAPPING_FILE = PROJECT_ROOT / "outputs/mappings/thermal_frame_mapping_official.csv"
HEIGHT_MAPPING_FILE = PROJECT_ROOT / "outputs/mappings/heightmap_x_mapping_official.csv"


@dataclass(frozen=True)
class ThermalMappingRow:
    track_id: int
    segment_frame_index: int
    raw_frame_index: int
    x_mm_center: float
    x_bin_left_mm: float
    x_bin_right_mm: float
    thermal_file: str
    mapping_status: str
    source_mapping_file: str


@dataclass(frozen=True)
class ThermalQuery:
    track_id: int
    requested_x_mm: float
    selected_x_mm: float
    coordinate_error_mm: float
    segment_frame_index: int
    raw_frame_index: int
    index_start: int
    index_end: int
    mapping_status: str
    source_mapping_file: str


@dataclass(frozen=True)
class HeightMappingRow:
    track_id: int
    height_x_index: int
    x_local_mm: float
    x_actual_mm: float
    valid_fraction: float
    heightmap_file: str
    mapping_status: str
    source_mapping_file: str


@dataclass(frozen=True)
class HeightWindowQuery:
    track_id: int
    requested_x_mm: float
    requested_half_width_mm: float
    selected_center_x_mm: float
    coordinate_error_mm: float
    index_start: int
    index_end: int
    x_start_mm: float
    x_end_mm: float
    mapping_status: str
    source_mapping_file: str


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen official mapping is missing: {path}")


@lru_cache(maxsize=None)
def get_thermal_mapping(track_id: int) -> Tuple[ThermalMappingRow, ...]:
    """Return all 400 official Thermal mapping rows for ``track_id``."""
    _require_file(THERMAL_MAPPING_FILE)
    rows = []
    with THERMAL_MAPPING_FILE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["track_id"]) != int(track_id):
                continue
            rows.append(ThermalMappingRow(
                track_id=int(row["track_id"]),
                segment_frame_index=int(row["segment_frame_index"]),
                raw_frame_index=int(row["raw_frame_index"]),
                x_mm_center=float(row["x_mm_center"]),
                x_bin_left_mm=float(row["x_bin_left_mm"]),
                x_bin_right_mm=float(row["x_bin_right_mm"]),
                thermal_file=row["thermal_file"],
                mapping_status=row["mapping_status"],
                source_mapping_file=str(THERMAL_MAPPING_FILE),
            ))
    if not rows:
        raise KeyError(f"No official Thermal mapping for track {track_id}")
    return tuple(rows)


def get_thermal_frame_nearest_x(track_id: int, x_mm: float) -> ThermalQuery:
    """Return the nearest official Thermal center; never extrapolate."""
    rows = get_thermal_mapping(track_id)
    if x_mm < rows[0].x_mm_center or x_mm > rows[-1].x_mm_center:
        raise ValueError(
            f"Requested x={x_mm} mm is outside Thermal center support "
            f"[{rows[0].x_mm_center}, {rows[-1].x_mm_center}]"
        )
    selected = min(rows, key=lambda row: (abs(row.x_mm_center - x_mm), row.segment_frame_index))
    return ThermalQuery(
        track_id=int(track_id),
        requested_x_mm=float(x_mm),
        selected_x_mm=selected.x_mm_center,
        coordinate_error_mm=abs(selected.x_mm_center - float(x_mm)),
        segment_frame_index=selected.segment_frame_index,
        raw_frame_index=selected.raw_frame_index,
        index_start=selected.segment_frame_index,
        index_end=selected.segment_frame_index + 1,
        mapping_status=selected.mapping_status,
        source_mapping_file=selected.source_mapping_file,
    )


@lru_cache(maxsize=None)
def get_height_mapping(track_id: int) -> Tuple[HeightMappingRow, ...]:
    """Return the strictly increasing official Height mapping for a track."""
    _require_file(HEIGHT_MAPPING_FILE)
    rows = []
    with HEIGHT_MAPPING_FILE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["track_id"]) != int(track_id):
                continue
            rows.append(HeightMappingRow(
                track_id=int(row["track_id"]),
                height_x_index=int(row["height_x_index"]),
                x_local_mm=float(row["x_local_mm"]),
                x_actual_mm=float(row["x_actual_mm"]),
                valid_fraction=float(row["valid_fraction"]),
                heightmap_file=row["heightmap_file"],
                mapping_status=row["mapping_status"],
                source_mapping_file=str(HEIGHT_MAPPING_FILE),
            ))
    if not rows:
        raise KeyError(f"No official Height mapping for track {track_id}")
    if any(b.x_actual_mm <= a.x_actual_mm for a, b in zip(rows, rows[1:])):
        raise ValueError(f"Official Height mapping is not strictly increasing for track {track_id}")
    return tuple(rows)


def get_height_indices_in_window(
    track_id: int,
    x_mm: float,
    half_width_mm: float,
) -> HeightWindowQuery:
    """Return inclusive-start/exclusive-end Height indices within a physical window.

    The requested center must lie inside measured Height support. Window edges
    are clipped to measured support and reported explicitly.
    """
    if half_width_mm < 0:
        raise ValueError("half_width_mm must be non-negative")
    rows = get_height_mapping(track_id)
    lo, hi = rows[0].x_actual_mm, rows[-1].x_actual_mm
    if x_mm < lo or x_mm > hi:
        raise ValueError(f"Requested x={x_mm} mm is outside Height support [{lo}, {hi}]")
    nearest = min(rows, key=lambda row: (abs(row.x_actual_mm - x_mm), row.height_x_index))
    requested_lo = max(lo, float(x_mm) - float(half_width_mm))
    requested_hi = min(hi, float(x_mm) + float(half_width_mm))
    selected = [row for row in rows if requested_lo <= row.x_actual_mm <= requested_hi]
    if not selected:
        selected = [nearest]
    return HeightWindowQuery(
        track_id=int(track_id),
        requested_x_mm=float(x_mm),
        requested_half_width_mm=float(half_width_mm),
        selected_center_x_mm=nearest.x_actual_mm,
        coordinate_error_mm=abs(nearest.x_actual_mm - float(x_mm)),
        index_start=selected[0].height_x_index,
        index_end=selected[-1].height_x_index + 1,
        x_start_mm=selected[0].x_actual_mm,
        x_end_mm=selected[-1].x_actual_mm,
        mapping_status=nearest.mapping_status,
        source_mapping_file=nearest.source_mapping_file,
    )
