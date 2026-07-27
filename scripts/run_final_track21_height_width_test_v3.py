#!/usr/bin/env python3
"""v2-protocol final evaluation on track 21 (script version 3).

Loads the frozen v1 track-21 labels (no rebuild), the frozen v1 model
artifacts, and the frozen v2 anchor + residual quantile artifacts, and
evaluates once. Adds probabilistic metrics (pinball, quantile-
approximated CRPS, interval coverage, calibration) and replaces the
fixed-mm confirmation threshold with a paired moving-block bootstrap
criterion. The v1 script and its outputs are untouched.

Status note recorded in the lock: track 21 was already opened by the v1
final test, so this run is a protocol revision rather than a second
independent single-shot confirmation.
"""
import hashlib, importlib.util, json, os, sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/nsf_mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from thermal_isotherm_features_v2 import build_track_isotherm_table, FEATURES as ISO_FEATURES

LOCK = ROOT / "configs/final_height_width_model_lock_v2.yaml"
ART1 = ROOT / "outputs/models/final_height_width_v1"
ART2 = ROOT / "outputs/models/final_height_width_v2"
FIG = ROOT / "outputs/figures/final_track21_v3"
V1_LOCK_HASH = "2fb60cbb83a04ea9819c3bfff9fa39b9765dc9bad74b4a7d0693103af013be1b"
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def predict_art(a, x):
    x = np.asarray(x, float)
    med = np.array(a.get("imputation_values", []))
    mean = np.array(a.get("scaler_mean", []))
    std = np.array(a.get("scaler_scale", []))
    if x.shape[1] == 0:
        return np.repeat(float(a["intercept"]), len(x))
    x = np.where(np.isfinite(x), x, med)
    return float(a["intercept"]) + ((x - mean) / std) @ np.array(a["coefficients"])


def point_metrics(y, p):
    e = np.asarray(p) - np.asarray(y)
    ae = np.abs(e)
    return dict(sample_count=int(len(y)), mae_mm=float(ae.mean()),
                rmse_mm=float(np.sqrt(np.mean(e * e))),
                median_absolute_error_mm=float(np.median(ae)),
                signed_bias_mm=float(e.mean()))


def quantile_metrics(y, Q):
    y = np.asarray(y, float)
    pin = [np.mean(np.maximum(q * (y - Q[:, i]), (q - 1) * (y - Q[:, i])))
           for i, q in enumerate(QUANTILES)]
    qs = np.asarray(QUANTILES)
    pl = np.stack([np.maximum(q * (y - Q[:, i]), (q - 1) * (y - Q[:, i]))
                   for i, q in enumerate(QUANTILES)], axis=1)
    trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    crps = float(np.mean(2.0 * trap(pl, qs, axis=1) / (qs[-1] - qs[0])))
    cov = float(np.mean((y >= Q[:, 0]) & (y <= Q[:, -1])))
    calib = {f"q{int(q*100):02d}": float(np.mean(y <= Q[:, i]))
             for i, q in enumerate(QUANTILES)}
    return dict(mean_pinball_mm=float(np.mean(pin)), crps_mm=crps,
                coverage90=cov, mean_pi90_width_mm=float(np.mean(Q[:, -1] - Q[:, 0])),
                calibration=calib)


def inflate(Q, s):
    med = Q[:, 2:3]
    out = Q.astype(float).copy()
    for i, q in enumerate(QUANTILES):
        if q == 0.5:
            continue
        z = norm.ppf(q)
        out[:, i] = med[:, 0] + np.sign(z) * np.sqrt((Q[:, i] - med[:, 0]) ** 2
                                                     + (z * s) ** 2)
    return np.sort(out, axis=1)


def block_bootstrap(y, pa, pb, block, n_boot, seed):
    rng = np.random.default_rng(seed)
    ea, eb = np.abs(y - pa), np.abs(y - pb)
    n = len(y)
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block)
                              for s in rng.choice(starts, k)])[:n]
        diffs[i] = ea[idx].mean() - eb[idx].mean()
    return diffs


def main():
    if sha(ROOT / "configs/final_height_width_model_lock_v1.yaml") != V1_LOCK_HASH:
        raise RuntimeError("v1 lock changed")
    lock = json.loads(LOCK.read_text())
    lock_hash = sha(LOCK)
    arts2 = {}
    for p in ART2.glob("*.json"):
        a = json.loads(p.read_text())
        if a["lock_hash"] != lock_hash:
            raise RuntimeError(f"v2 artifact lock mismatch: {p.name}")
        arts2[p.stem] = a
    arts1 = {p.stem: json.loads(p.read_text()) for p in ART1.glob("*.json")
             if p.name != "feature_schema.json"}

    labels = pd.read_csv(ROOT / "outputs/height_labels/height_labels_v1_track21_final_test.csv")
    labels = labels.sort_values("segment_frame_index").reset_index(drop=True)
    iso = build_track_isotherm_table(ROOT / "thermal/Thermal_21.mat",
                                     labels.raw_frame_index.tolist())
    labels = pd.concat([labels, iso.drop(columns=["raw_frame_index"])
                        .reset_index(drop=True)], axis=1)
    labels["x_actual_mm_squared"] = labels.x_actual_mm ** 2

    audit = module("scripts/run_thermal_representation_audit.py", "audit_v3")
    arr = np.asarray(loadmat(ROOT / "thermal/Thermal_21.mat")["temperature_data"])
    shape_feats = {}
    for r in labels.itertuples():
        fr = np.asarray(arr[int(r.raw_frame_index)], float)
        ok, cx, cy, mask = audit.hotspot(fr)
        shape_feats[r.sample_id] = audit.shape_features(fr, ok, cx, cy, mask)

    elig = labels[(labels.primary_label_eligible == True)
                  & labels.target_width_mm.notna()].reset_index(drop=True)
    prev = pd.read_csv(ROOT / "outputs/models/final_track21_height_width_predictions.csv")
    ids_b = set(prev.loc[prev.cohort_id == "cohort_B", "sample_id"])
    cohorts = {"cohort_A": elig, "cohort_B":
               elig[elig.sample_id.isin(ids_b)].reset_index(drop=True)}

    k_star = lock["anchor"]["k_star"]
    anchor_col = lock["anchor"]["column"]
    feats = lock["model"]["features"]
    s_level = lock["s_level_mm"]
    cc = lock["confirmation_criterion_v2"]

    metric_rows, pred_rows, verdicts = [], [], {}
    for cname, coh in cohorts.items():
        y = coh.target_width_mm.to_numpy(float)
        x = coh.x_actual_mm.to_numpy(float)
        tag = "cohort_a" if cname == "cohort_A" else "cohort_b"
        preds = {
            "training_mean": predict_art(arts1[f"{tag}_training_mean"],
                                         np.empty((len(coh), 0))),
            "x_only": predict_art(arts1[f"{tag}_x_only"],
                                  np.column_stack([x, x * x])),
            "thermal_v1": predict_art(arts1[f"{tag}_fixed_thermal_shape_ridge"],
                                      np.stack([shape_feats[s] for s in coh.sample_id])),
        }
        anchor = k_star * coh[anchor_col].to_numpy(float)
        ao = arts2["anchor_only"]
        Q_anchor = np.sort(anchor[:, None] + np.array(
            [ao["residual_quantiles"][f"q{int(q*100):02d}"] for q in QUANTILES])[None, :],
            axis=1)
        X = coh[feats].to_numpy(float)
        Q_res = np.sort(anchor[:, None] + np.column_stack(
            [predict_art(arts2[f"anchor_residual_q{int(q*100):02d}"], X)
             for q in QUANTILES]), axis=1)
        Q_inf = inflate(Q_res, s_level)
        preds["anchor_only"] = Q_anchor[:, 2]
        preds["anchor_residual"] = Q_res[:, 2]

        for name, p in preds.items():
            m = dict(cohort_id=cname, model_id=name, **point_metrics(y, p))
            if name == "anchor_only":
                m.update(quantile_metrics(y, Q_anchor))
            if name == "anchor_residual":
                m.update(quantile_metrics(y, Q_res))
                m2 = dict(cohort_id=cname, model_id="anchor_residual_inflated",
                          **point_metrics(y, Q_inf[:, 2]),
                          **quantile_metrics(y, Q_inf))
                metric_rows.append(m2)
            metric_rows.append(m)
        for i, s in enumerate(coh.sample_id):
            row = dict(cohort_id=cname, sample_id=s,
                       x_actual_mm=float(x[i]), target_width_mm=float(y[i]),
                       anchor_mm=float(anchor[i]))
            for name, p in preds.items():
                row[f"{name}_mm"] = float(p[i])
            for j, q in enumerate(QUANTILES):
                row[f"anchor_residual_q{int(q*100):02d}"] = float(Q_res[i, j])
                row[f"anchor_residual_inflated_q{int(q*100):02d}"] = float(Q_inf[i, j])
            pred_rows.append(row)

        diffs = block_bootstrap(y, preds["x_only"], preds["anchor_residual"],
                                cc["block_positions"], cc["n_bootstrap"], cc["seed"])
        ci = np.percentile(diffs, [2.5, 97.5])
        d = float(np.mean(np.abs(y - preds["x_only"]))
                  - np.mean(np.abs(y - preds["anchor_residual"])))
        rel = d / float(np.mean(np.abs(y - preds["x_only"])))
        ok = bool(ci[0] > 0 and rel >= 0.05)
        verdicts[cname] = dict(delta_mae_mm=d, relative_improvement=rel,
                               ci95_mm=[float(ci[0]), float(ci[1])], confirmed=ok)
        print(f"{cname}: n={len(y)} | " + " ".join(
            f"{n}={np.mean(np.abs(y-p)):.4f}" for n, p in preds.items()))
        print(f"  dMAE(x_only - anchor_residual) = {d:.4f} mm ({rel*100:.1f}%), "
              f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}] -> "
              + ("CONFIRMED" if ok else "NOT CONFIRMED") + " (v2 criterion)")

    out = ART2
    pd.DataFrame(metric_rows).to_csv(out / "final_track21_v3_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "final_track21_v3_predictions.csv", index=False)
    (out / "final_track21_v3_summary.json").write_text(json.dumps(dict(
        lock_hash=lock_hash, verdicts=verdicts,
        status_note=lock["note"]), indent=2) + "\n")

    # figures (cohort A)
    FIG.mkdir(parents=True, exist_ok=True)
    coh = cohorts["cohort_A"]
    y = coh.target_width_mm.to_numpy(float)
    P = pd.DataFrame(pred_rows)
    P = P[P.cohort_id == "cohort_A"]
    mm = pd.DataFrame(metric_rows)
    mm = mm[(mm.cohort_id == "cohort_A")]
    order = ["training_mean", "x_only", "thermal_v1", "anchor_only", "anchor_residual"]
    maes = [float(mm[mm.model_id == n].mae_mm.iloc[0]) for n in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(order, maes,
                  color=["tab:blue", "tab:gray", "tab:orange", "tab:cyan", "tab:green"])
    for b, v in zip(bars, maes):
        ax.text(b.get_x() + b.get_width() / 2, v + .004, f"{v:.3f}", ha="center")
    v = verdicts["cohort_A"]
    ax.set_ylabel("track-21 width MAE (mm)")
    ax.set_title(f"cohort A: dMAE(x_only - anchor_residual) = "
                 f"{v['delta_mae_mm']:.3f} mm ({v['relative_improvement']*100:.1f}%), "
                 f"CI [{v['ci95_mm'][0]:.3f}, {v['ci95_mm'][1]:.3f}]")
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "final_track21_v3_mae_ladder.png", dpi=150)

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.fill_between(P.x_actual_mm, P.anchor_residual_inflated_q05,
                    P.anchor_residual_inflated_q95, alpha=.28,
                    color="tab:green", label="90% interval (inflated)")
    ax.plot(P.x_actual_mm, P.x_only_mm, color="tab:gray", lw=1.1, label="x_only")
    ax.plot(P.x_actual_mm, P.anchor_residual_mm, color="tab:green", lw=1.5,
            label="anchor_residual median")
    ax.plot(P.x_actual_mm, P.target_width_mm, "k.", ms=3, label="frozen target")
    ax.set_xlabel("actual x (mm)")
    ax.set_ylabel("local width (mm)")
    ax.legend(ncol=4, fontsize=9)
    ax.set_title("track 21 (cohort A): v2-protocol width prediction")
    fig.tight_layout()
    fig.savefig(FIG / "final_track21_v3_predictions.png", dpi=150)

    qs = [5, 25, 50, 75, 95]
    fig, ax = plt.subplots(figsize=(5, 4.6))
    for mid, c in [("anchor_residual", "tab:red"),
                   ("anchor_residual_inflated", "tab:green")]:
        row = mm[mm.model_id == mid].iloc[0]
        emp = [json.loads(row.calibration.replace("'", '"'))[f"q{q:02d}"]
               if isinstance(row.calibration, str) else row.calibration[f"q{q:02d}"]
               for q in qs]
        ax.plot(np.array(qs) / 100, emp, "o-", color=c, label=mid)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("nominal quantile")
    ax.set_ylabel("empirical coverage")
    ax.legend(fontsize=8)
    ax.set_title("track-21 calibration")
    fig.tight_layout()
    fig.savefig(FIG / "final_track21_v3_calibration.png", dpi=150)
    print("wrote metrics, predictions, and figures under "
          "outputs/models/final_height_width_v2 and outputs/figures/final_track21_v3")


if __name__ == "__main__":
    main()
