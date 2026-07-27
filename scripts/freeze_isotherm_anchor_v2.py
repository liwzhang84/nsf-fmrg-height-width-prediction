#!/usr/bin/env python3
"""Development-only calibration of the isotherm width anchor (v2).

Reads the frozen development labels and the development thermal stacks,
computes per-track isotherm extents, and calibrates

    anchor(x) = k_star * iso{c_star}_mm_roll(x)

by selecting the count c whose per-track ratio
median(target_width) / median(extent) is most consistent across the
three development tracks, with k_star the mean of those ratios.
Track 21 is not touched. Results and feature caches are written under
configs/ and outputs/thermal_isotherm_v2/.
"""
import hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from thermal_isotherm_features_v2 import (build_track_isotherm_table,
                                          ISO_COUNTS, ROLL_HALF)

DEV = (8, 10, 14)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    labels = pd.read_csv(ROOT / "outputs/height_labels/height_labels_v1_development.csv")
    ratios = {c: {} for c in ISO_COUNTS}
    for t in DEV:
        lt = labels[labels.track_id == t].sort_values("segment_frame_index")
        iso = build_track_isotherm_table(ROOT / f"thermal/Thermal_{t}.mat",
                                         lt.raw_frame_index.tolist())
        iso.insert(0, "track_id", t)
        iso.insert(1, "sample_id", lt.sample_id.tolist())
        iso.to_csv(ROOT / f"outputs/thermal_isotherm_v2/iso_features_track{t}.csv",
                   index=False)
        el = lt[lt.primary_label_eligible == True]
        w_med = float(el.target_width_mm.median())
        m = iso.set_index("raw_frame_index")
        for c in ISO_COUNTS:
            e_med = float(m.loc[el.raw_frame_index, f"iso{c}_mm_roll"].median())
            ratios[c][t] = w_med / e_med
        print(f"track {t}: median width {w_med:.3f} mm, "
              f"iso1400_roll {m[f'iso1400_mm_roll'].median():.3f} mm")
    stats = {}
    for c in ISO_COUNTS:
        v = np.array(list(ratios[c].values()))
        stats[c] = dict(ratios={str(k): float(x) for k, x in ratios[c].items()},
                        mean=float(v.mean()), cv=float(v.std() / v.mean()))
        print(f"iso{c}: ratios {[round(x,3) for x in v]}  cv={stats[c]['cv']:.3f}")
    c_star = min(ISO_COUNTS, key=lambda c: stats[c]["cv"])
    k_star = stats[c_star]["mean"]
    cfg = dict(protocol_version="isotherm_anchor_v2",
               anchor_definition="k_star * iso{c_star}_mm_roll",
               c_star=int(c_star), k_star=float(k_star),
               rolling_half_frames=ROLL_HALF,
               candidate_counts=list(ISO_COUNTS),
               selection_rule="minimum coefficient of variation of the "
                              "three development-track ratios "
                              "median(width)/median(extent)",
               development_tracks=list(DEV),
               per_count_statistics={str(c): stats[c] for c in ISO_COUNTS},
               source_module="src/thermal_isotherm_features_v2.py",
               source_module_hash=sha(ROOT / "src/thermal_isotherm_features_v2.py"),
               development_labels_hash=sha(
                   ROOT / "outputs/height_labels/height_labels_v1_development.csv"))
    out = ROOT / "configs/isotherm_anchor_v2.yaml"
    out.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"selected c*={c_star}, k*={k_star:.4f} -> {out}")


if __name__ == "__main__":
    main()
