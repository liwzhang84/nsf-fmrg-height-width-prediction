#!/usr/bin/env python3
"""Sensitivity study: the v2 anchor method under validity-run labels.

The primary v2 evaluation uses the frozen gradient-edge labels, whose
widths are nearly constant across laser power (dev medians
0.741-0.804 mm). The isotherm anchor encodes power-dependent
solidification physics, so under that label definition it cannot help
and the v2 criterion returns NOT CONFIRMED. This script repeats the
evaluation under an alternative label estimator that follows the
resolidified band directly: the per-y valid-pixel fraction of the
interferometer is near-binary across the smooth track / rough
substrate transition, and the boundaries are the sub-pixel edges of
the largest supra-threshold run. Anchor constants are recalibrated on
development tracks only; residual-model hyperparameters, quantiles,
inflation rule, and the confirmation criterion are inherited from the
frozen v2 lock. Results are written as a clearly labeled sensitivity
cohort; nothing in the primary chain is modified.
"""
import hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import QuantileRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from thermal_isotherm_features_v2 import ISO_COUNTS

DEV, TEST = (8, 10, 14), 21
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
PIXEL_MM = 0.003982
OUT = ROOT / "outputs/models/final_height_width_v2"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parse_wyko(path):
    with open(path, "rb") as f:
        head = f.read(4000).decode("latin1").splitlines()
    skip = next(i for i, l in enumerate(head) if l.startswith("RAW_DATA")) + 1
    nx = int(next(l for l in head if l.startswith("X Size")).split("\t")[1])
    ny = int(next(l for l in head if l.startswith("Y Size")).split("\t")[1])
    df = pd.read_csv(path, sep="\t", skiprows=skip, names=["x", "y", "z"],
                     na_values=["Bad"], engine="c")
    Z = df.z.to_numpy().reshape(nx, ny).T
    return Z, df.x.to_numpy()[::ny], df.y.to_numpy()[:ny]


def validity_run_labels(track):
    Z, x_asc, yy = parse_wyko(ROOT / f"height/Heightmap_{track}.ASC")
    x_act = 100.0 - x_asc
    V = np.isfinite(Z)
    kern = np.ones(5) / 5.0
    rows = []
    for c in 20.0 + 0.2 * (np.arange(400) + 0.5):
        m = (x_act >= c - 0.1) & (x_act < c + 0.1)
        rec = dict(track_id=track, x_actual_mm=round(c, 4),
                   target_width_mm=np.nan, eligible=False)
        if m.any():
            v = np.convolve(V[:, m].mean(axis=1), kern, mode="same")
            p10, p90 = np.percentile(v, [10, 90])
            if p90 - p10 >= 0.40:
                thr = 0.5 * (p10 + p90)
                ab = v > thr
                d = np.diff(np.concatenate([[0], ab.astype(int), [0]]))
                st, en = np.where(d == 1)[0], np.where(d == -1)[0] - 1
                if len(st):
                    k = np.argmax(en - st)
                    s, e = st[k], en[k]
                    if 3 <= s and e <= len(v) - 4:
                        yl, yr = yy[s], yy[e]
                        if s > 0 and v[s] != v[s - 1]:
                            yl = yy[s-1] + (thr - v[s-1]) / (v[s] - v[s-1]) * (yy[s] - yy[s-1])
                        if e < len(v) - 1 and v[e+1] != v[e]:
                            yr = yy[e] + (thr - v[e]) / (v[e+1] - v[e]) * (yy[e+1] - yy[e])
                        w = yr - yl
                        if 0.2 <= w <= 1.8:
                            rec.update(target_width_mm=float(w), eligible=True)
        rows.append(rec)
    return pd.DataFrame(rows)


def fit_q(X, y, q, alpha):
    med = np.nanmedian(X, axis=0)
    Xi = np.where(np.isfinite(X), X, med)
    mu, sd = Xi.mean(0), Xi.std(0)
    sd[sd == 0] = 1
    m = QuantileRegressor(quantile=q, alpha=alpha, solver="highs")
    m.fit((Xi - mu) / sd, y)
    return lambda Xn: (m.intercept_
                       + ((np.where(np.isfinite(Xn), Xn, med) - mu) / sd) @ m.coef_)


def main():
    lock = json.loads((ROOT / "configs/final_height_width_model_lock_v2.yaml").read_text())
    alpha = lock["model"]["alpha"]
    cc = lock["confirmation_criterion_v2"]
    feats = lock["model"]["features"]

    data = {}
    for t in (*DEV, TEST):
        lab = validity_run_labels(t)
        iso = pd.read_csv(ROOT / f"outputs/thermal_isotherm_v2/iso_features_track{t}.csv") \
            if t != TEST else None
        if iso is None:
            from thermal_isotherm_features_v2 import build_track_isotherm_table
            l21 = pd.read_csv(ROOT / "outputs/height_labels/height_labels_v1_track21_final_test.csv")
            l21 = l21.sort_values("segment_frame_index")
            iso = build_track_isotherm_table(ROOT / "thermal/Thermal_21.mat",
                                            l21.raw_frame_index.tolist())
            iso["x_actual_mm"] = l21.x_actual_mm.to_numpy().round(4)
        else:
            devlab = pd.read_csv(ROOT / "outputs/height_labels/height_labels_v1_development.csv")
            xmap = devlab[devlab.track_id == t].set_index("sample_id").x_actual_mm.round(4)
            iso["x_actual_mm"] = iso.sample_id.map(xmap)
        d = lab.merge(iso, on="x_actual_mm", how="inner")
        d = d[d.eligible].copy()
        d["x_actual_mm_squared"] = d.x_actual_mm ** 2
        data[t] = d.reset_index(drop=True)
        print(f"track {t}: {len(d)} eligible validity-run labels, "
              f"median width {d.target_width_mm.median():.3f} mm")

    ratios = {c: {t: float(data[t].target_width_mm.median()
                           / data[t][f"iso{c}_mm_roll"].median()) for t in DEV}
              for c in ISO_COUNTS}
    c_star = min(ISO_COUNTS, key=lambda c: np.std(list(ratios[c].values()))
                 / np.mean(list(ratios[c].values())))
    k_star = float(np.mean(list(ratios[c_star].values())))
    v = np.array(list(ratios[c_star].values()))
    print(f"sensitivity anchor: c*={c_star}, k*={k_star:.4f}, "
          f"dev ratio cv={v.std()/v.mean():.3f}")
    col = f"iso{c_star}_mm_roll"

    tr = pd.concat([data[t] for t in DEV])
    te = data[TEST]
    y = te.target_width_mm.to_numpy()
    x = te.x_actual_mm.to_numpy()
    res = tr.target_width_mm.to_numpy() - k_star * tr[col].to_numpy()
    anchor = k_star * te[col].to_numpy()
    Q = np.sort(anchor[:, None] + np.column_stack(
        [fit_q(tr[feats].to_numpy(), res, q, alpha)(te[feats].to_numpy())
         for q in QUANTILES]), axis=1)
    fx = [fit_q(tr[["x_actual_mm", "x_actual_mm_squared"]].to_numpy(),
                tr.target_width_mm.to_numpy(), q, alpha) for q in QUANTILES]
    Qx = np.sort(np.column_stack(
        [f(te[["x_actual_mm", "x_actual_mm_squared"]].to_numpy()) for f in fx]), axis=1)

    mae = float(np.mean(np.abs(y - Q[:, 2])))
    mae_x = float(np.mean(np.abs(y - Qx[:, 2])))
    mae_a = float(np.mean(np.abs(y - anchor)))
    rng = np.random.default_rng(cc["seed"])
    ea, eb = np.abs(y - Qx[:, 2]), np.abs(y - Q[:, 2])
    n, blk = len(y), cc["block_positions"]
    starts = np.arange(n - blk + 1)
    diffs = np.array([
        (lambda idx: ea[idx].mean() - eb[idx].mean())(
            np.concatenate([np.arange(s, s + blk)
                            for s in rng.choice(starts, int(np.ceil(n / blk)))])[:n])
        for _ in range(cc["n_bootstrap"])])
    ci = np.percentile(diffs, [2.5, 97.5])
    rel = (mae_x - mae) / mae_x
    ok = bool(ci[0] > 0 and rel >= 0.05)
    cov = float(np.mean((y >= Q[:, 0]) & (y <= Q[:, -1])))
    print(f"track 21 (validity-run labels, n={len(y)}): x_only={mae_x:.4f} "
          f"anchor_only={mae_a:.4f} anchor_residual={mae:.4f} cov90={cov:.3f}")
    print(f"dMAE = {mae_x-mae:.4f} mm ({rel*100:.1f}%), CI [{ci[0]:.4f}, {ci[1]:.4f}] "
          f"-> {'CONFIRMED' if ok else 'NOT CONFIRMED'} (sensitivity cohort)")

    summary = dict(study="validity_run_label_sensitivity_v2",
                   anchor=dict(c_star=int(c_star), k_star=k_star,
                               dev_ratio_cv=float(v.std() / v.mean())),
                   n_test=int(len(y)),
                   mae_mm=dict(x_only=mae_x, anchor_only=mae_a,
                               anchor_residual=mae),
                   coverage90=cov,
                   delta_mae_mm=float(mae_x - mae),
                   relative_improvement=float(rel),
                   ci95_mm=[float(ci[0]), float(ci[1])], confirmed=ok,
                   note=("Sensitivity study under an alternative label "
                         "estimator; the primary v2 result uses the frozen "
                         "gradient-edge labels and is reported separately."))
    (OUT / "sensitivity_validity_run_v2_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print("wrote", OUT / "sensitivity_validity_run_v2_summary.json")


if __name__ == "__main__":
    main()
