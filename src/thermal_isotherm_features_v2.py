"""Isotherm-extent thermal representation (v2 protocol cycle).

For each thermal frame, the cross-track extent of the c-count isotherm
is the width of the column span containing any pixel above c. The
deposited track width is set by the solidification isotherm, so at the
right count these extents track width across laser powers, which the
fixed shape features cannot do. Extents are in-situ quantities and
carry no information about the post-process height map.

Image convention (checked on the released frames): rows follow the scan
direction with the cooling tail at larger row indices; columns are the
cross-track direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.io import loadmat

PX_MM = 0.014
ISO_COUNTS = (1300, 1400, 1500, 1600, 1800)
ROLL_HALF = 10  # +-10 frames = +-2.0 mm at 0.2 mm/frame


def isotherm_extents(frame: np.ndarray) -> dict:
    f = np.asarray(frame, np.float32)
    out = {}
    for c in ISO_COUNTS:
        jj = np.nonzero((f > c).any(axis=0))[0]
        out[f"iso{c}_mm"] = float(jj[-1] - jj[0]) * PX_MM if len(jj) > 1 else 0.0
    return out


def build_track_isotherm_table(mat_path, raw_frame_indices) -> pd.DataFrame:
    """Isotherm extents for the given raw frame indices, in the order
    provided (expected: segment order), plus +-ROLL_HALF rolling means."""
    arr = np.asarray(loadmat(mat_path)["temperature_data"])
    rows = []
    for idx in raw_frame_indices:
        idx = int(idx)
        if not (0 <= idx < len(arr)):
            raise RuntimeError(f"raw frame index out of bounds: {idx}")
        rows.append({"raw_frame_index": idx, **isotherm_extents(arr[idx])})
    df = pd.DataFrame(rows)
    for c in ISO_COUNTS:
        a = df[f"iso{c}_mm"].to_numpy(float)
        df[f"iso{c}_mm_roll"] = [
            a[max(0, i - ROLL_HALF):i + ROLL_HALF + 1].mean()
            for i in range(len(a))]
    return df


FEATURES = ([f"iso{c}_mm" for c in ISO_COUNTS]
            + [f"iso{c}_mm_roll" for c in ISO_COUNTS])
