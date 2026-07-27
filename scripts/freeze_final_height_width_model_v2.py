#!/usr/bin/env python3
"""Freeze the v2 residual quantile models before the v2 final test.

Model:  width(x) = anchor(x) + Q_q(features),  anchor = k_star * iso_roll.
The residual learners are linear quantile regressions so their artifacts
keep the exact JSON schema of the v1 ridge artifacts (intercept,
coefficients, scaler, imputation medians) and can be applied with the
same predict function. Regularization strength is selected by
leave-one-track-out pinball loss on the development tracks, with the
anchor ratio re-fit on the two training tracks inside each fold. The
fold-level median errors give the level-uncertainty scale s_level used
to widen the predictive quantiles at test time. Track 21 is not opened.
"""
import hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from thermal_isotherm_features_v2 import FEATURES as ISO_FEATURES

DEV = (8, 10, 14)
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
ALPHAS = (1e-4, 1e-3, 1e-2)
LOCK = ROOT / "configs/final_height_width_model_lock_v2.yaml"
ART = ROOT / "outputs/models/final_height_width_v2"
FEATURES = ["x_actual_mm", "x_actual_mm_squared"] + ISO_FEATURES


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_dev():
    labels = pd.read_csv(ROOT / "outputs/height_labels/height_labels_v1_development.csv")
    frames = []
    for t in DEV:
        iso = pd.read_csv(ROOT / f"outputs/thermal_isotherm_v2/iso_features_track{t}.csv")
        lt = labels[labels.track_id == t].merge(
            iso.drop(columns=["track_id"]), on=["sample_id", "raw_frame_index"])
        frames.append(lt)
    d = pd.concat(frames)
    d = d[(d.primary_label_eligible == True) & d.target_width_mm.notna()].copy()
    d["x_actual_mm_squared"] = d.x_actual_mm ** 2
    return d


def pinball(y, q_pred, q):
    u = y - q_pred
    return float(np.mean(np.maximum(q * u, (q - 1) * u)))


def fit_quantile(train_X, train_y, q, alpha):
    med = np.nanmedian(train_X, axis=0)
    X = np.where(np.isfinite(train_X), train_X, med)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    m = QuantileRegressor(quantile=q, alpha=alpha, solver="highs")
    m.fit((X - mu) / sd, train_y)
    return dict(intercept=float(m.intercept_),
                coefficients=[float(c) for c in m.coef_],
                scaler_mean=[float(v) for v in mu],
                scaler_scale=[float(v) for v in sd],
                imputation_values=[float(v) for v in med])


def apply(art, X):
    X = np.asarray(X, float)
    med = np.array(art["imputation_values"])
    X = np.where(np.isfinite(X), X, med)
    Z = (X - np.array(art["scaler_mean"])) / np.array(art["scaler_scale"])
    return art["intercept"] + Z @ np.array(art["coefficients"])


def main():
    anchor_cfg = json.loads((ROOT / "configs/isotherm_anchor_v2.yaml").read_text())
    c_star = anchor_cfg["c_star"]
    anchor_col = f"iso{c_star}_mm_roll"
    ratios = {int(k): v for k, v in
              anchor_cfg["per_count_statistics"][str(c_star)]["ratios"].items()}
    data = load_dev()

    results, level_errors = {}, {a: [] for a in ALPHAS}
    for alpha in ALPHAS:
        fold_pin = []
        for hold in DEV:
            tr_tracks = [t for t in DEV if t != hold]
            k_fold = float(np.mean([ratios[t] for t in tr_tracks]))
            tr = data[data.track_id != hold]
            te = data[data.track_id == hold]
            res_tr = tr.target_width_mm.to_numpy() - k_fold * tr[anchor_col].to_numpy()
            y_te = te.target_width_mm.to_numpy()
            preds = {}
            for q in QUANTILES:
                art = fit_quantile(tr[FEATURES].to_numpy(), res_tr, q, alpha)
                preds[q] = k_fold * te[anchor_col].to_numpy() + apply(art, te[FEATURES].to_numpy())
            Q = np.sort(np.column_stack([preds[q] for q in QUANTILES]), axis=1)
            fold_pin.append(np.mean([pinball(y_te, Q[:, i], q)
                                     for i, q in enumerate(QUANTILES)]))
            level_errors[alpha].append(float(np.median(y_te) - np.median(Q[:, 2])))
        results[alpha] = float(np.mean(fold_pin))
        print(f"alpha={alpha}: LOTO pinball {results[alpha]:.4f}  "
              f"level errors {[round(e,3) for e in level_errors[alpha]]}")
    alpha_star = min(results, key=results.get)
    s_level = float(np.sqrt(np.mean(np.square(level_errors[alpha_star]))))
    k_star = anchor_cfg["k_star"]
    print(f"selected alpha={alpha_star}, s_level={s_level:.4f} mm")

    lock = dict(
        protocol_version="final_height_width_model_lock_v2",
        note=("Post-registration protocol revision: track 21 was already "
              "opened once by the v1 final test, so the v2 evaluation is a "
              "protocol revision, not a second independent confirmation. "
              "Every numerical choice in this lock derives from the "
              "development tracks only."),
        inherits="configs/final_height_width_model_lock_v1.yaml",
        v1_lock_hash="2fb60cbb83a04ea9819c3bfff9fa39b9765dc9bad74b4a7d0693103af013be1b",
        development_tracks=list(DEV), final_test_track=21,
        anchor=dict(file="configs/isotherm_anchor_v2.yaml",
                    hash=sha(ROOT / "configs/isotherm_anchor_v2.yaml"),
                    c_star=c_star, k_star=k_star, column=anchor_col),
        model=dict(structure="width = anchor + linear_quantile(residual)",
                   features=FEATURES, quantiles=list(QUANTILES),
                   alpha=alpha_star, solver="highs",
                   selected_by="mean LOTO pinball over development tracks"),
        s_level_mm=s_level,
        level_inflation="Q(q) = Q(0.5) + sign(z_q)*sqrt((Q(q)-Q(0.5))^2 + (z_q*s_level)^2)",
        confirmation_criterion_v2=dict(
            statement=("Improvement over x_only confirmed iff the 95% "
                       "moving-block bootstrap CI lower bound of "
                       "dMAE(x_only - anchor_residual) exceeds 0 and the "
                       "relative MAE improvement is at least 5%."),
            block_positions=25, n_bootstrap=2000, seed=0),
    )
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    lock_hash = sha(LOCK)

    ART.mkdir(parents=True, exist_ok=True)
    res_all = data.target_width_mm.to_numpy() - k_star * data[anchor_col].to_numpy()
    for q in QUANTILES:
        art = fit_quantile(data[FEATURES].to_numpy(), res_all, q, alpha_star)
        art.update(model_id=f"anchor_residual_q{int(q*100):02d}", quantile=q,
                   feature_names=FEATURES, alpha=alpha_star,
                   lock_hash=lock_hash)
        (ART / f"anchor_residual_q{int(q*100):02d}.json").write_text(
            json.dumps(art, indent=2) + "\n")
    anchor_only = dict(model_id="anchor_only",
                       residual_quantiles={f"q{int(q*100):02d}":
                                           float(np.quantile(res_all, q))
                                           for q in QUANTILES},
                       k_star=k_star, anchor_column=anchor_col,
                       lock_hash=lock_hash)
    (ART / "anchor_only.json").write_text(json.dumps(anchor_only, indent=2) + "\n")
    print(f"lock hash {lock_hash}")
    print(f"wrote {len(QUANTILES)+1} artifacts -> {ART}")


if __name__ == "__main__":
    main()
