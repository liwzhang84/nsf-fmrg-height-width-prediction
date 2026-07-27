#!/usr/bin/env python3
"""Archive development provenance, select locked alphas, and refit artifacts.

This script is development-only.  It must complete before the final-test
runner is allowed to inspect any held-out source.
"""
from __future__ import annotations
import hashlib, importlib.util, json, os, shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR","/private/tmp/nsf_mplconfig")
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DEV=(8,10,14)
ALPHAS=(.1,1.,10.,100.,1000.)
CONF=ROOT/"configs/sem_mapping_engineering_confirmation_v1.yaml"
ARCHIVE=ROOT/"configs/sem_mapping_engineering_confirmation_final_v1.yaml"
LOCK=ROOT/"configs/final_height_width_model_lock_v1.yaml"
ART=ROOT/"outputs/models/final_height_width_v1"
EXPECTED_CONF="90c6ae9ce1b0cfc094db197c117369df45f91d7ec9f257d798417428e3d9649c"
EXPECTED_HEIGHT={"configs/height_boundary_estimator_v1.yaml":"4b1094cf2084f10026cdd470dba7ba3bdf8b561bfef02ba7ef3413e8a555617d",
 "configs/height_label_policy_v1.yaml":"5b568d33b06761ffdce9c35fd75fa03307c6ec8dcdb4cd510e7513d39e96c994"}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_module():
 spec=importlib.util.spec_from_file_location("deadline_dev",ROOT/"scripts/run_sem_multimodal_deadline_models.py")
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def artifact(name,features,x,y,alpha,tracks,lock_hash,source_hashes):
 x=np.asarray(x,float);y=np.asarray(y,float)
 med=np.nanmedian(x,0);x=np.where(np.isfinite(x),x,med);mean=x.mean(0);std=x.std(0);std[std<1e-12]=1
 z=(x-mean)/std;design=np.column_stack([np.ones(len(z)),z])
 penalty=np.eye(design.shape[1]);penalty[0,0]=0
 coef=np.linalg.solve(design.T@design+alpha*penalty,design.T@y)
 row={"artifact_version":"final_height_width_v1","model_id":name,"lock_hash":lock_hash,
  "feature_names":features,"imputation_values":med.tolist(),"scaler_mean":mean.tolist(),
  "scaler_scale":std.tolist(),"intercept":float(coef[0]),"coefficients":coef[1:].tolist(),
  "ridge_alpha":alpha,"intercept_policy":"unpenalized","training_sample_count":len(y),
  "training_track_counts":{str(t):int(np.sum(np.asarray(tracks)==t)) for t in DEV},
  "source_hashes":source_hashes,"software_version":"numpy-transparent-ridge-v1"}
 (ART/f"{name}.json").write_text(json.dumps(row,indent=2)+"\n");return row
def predict_art(a,x):
 x=np.asarray(x,float);med=np.array(a["imputation_values"]);mean=np.array(a["scaler_mean"]);std=np.array(a["scaler_scale"])
 x=np.where(np.isfinite(x),x,med);return a["intercept"]+((x-mean)/std)@np.array(a["coefficients"])
def eval_alpha(m,data,key,extra=None):
 scores={}
 for alpha in ALPHAS:
  maes=[]
  for held in DEV:
   tr=[i for i,d in enumerate(data) if d["track_id"]!=held];va=[i for i,d in enumerate(data) if d["track_id"]==held]
   p,_=m.fit_model(data,key,tr,va,alpha,None,extra);maes.append(float(np.mean(abs(p-np.array([data[i]["y"] for i in va])))))
  scores[alpha]=maes
 # Stronger regularization wins an exact tie.
 best=min(ALPHAS,key=lambda a:(np.mean(scores[a]),-a))
 return best,scores
def corrected_qa(m,geo,tc,manifest):
 out=ROOT/"outputs/figures/sem_input_deadline_qa_corrected";out.mkdir(parents=True,exist_ok=True)
 mapping=pd.read_csv(ROOT/"outputs/sem_deadline/sem_mapping_deadline_v1.csv").set_index("sample_id")
 chosen=[]
 for t in DEV:
  g=manifest[(manifest.track_id==t)&(manifest.tile_boundary_flag==True)].copy()
  # Avoid the prior beginning/middle/end examples; prefer eligible+available.
  full=manifest[manifest.track_id==t].sort_values("x_actual_mm")
  excluded={full.iloc[0].sample_id,full.iloc[len(full)//2].sample_id,full.iloc[-1].sample_id}
  g=g[~g.sample_id.isin(excluded)]
  g["priority"]=(g.primary_label_eligible==True).astype(int)+(g.sem_input_available==True).astype(int)
  r=g.sort_values(["priority","x_actual_mm"],ascending=[False,True]).iloc[0]
  mp=mapping.loc[r.sample_id];q=geo[(t,int(mp.sem_tile_id))]
  axis=tc[t]["axis"];direction=tc[t]["direction"];u=float(mp.sem_local_coordinate_px)
  half=.4/q["px_mm"];up=1.0/q["px_mm"]
  fig,ax=plt.subplots(figsize=(10,4));ax.imshow(q["array"],cmap="gray")
  ax.axvspan(u-half,u+half,color="red",alpha=.4,label="central exclusion")
  ax.axvspan(u-half-up,u-half,color="cyan",alpha=.25,label="upstream")
  ax.axvspan(u+half,u+half+up,color="lime",alpha=.25,label="downstream")
  ax.axvline(0,color="orange");ax.axvline(q["array"].shape[1]-1,color="orange")
  ax.set_title(f"Track {t} true tile-boundary QA: {r.sample_id}, x={r.x_actual_mm:.1f} mm, tile {int(mp.sem_tile_id):02d}")
  ax.legend();fig.tight_layout();target=out/f"track_{t}_near_tile_boundary_corrected.png";fig.savefig(target,dpi=170);plt.close(fig)
  chosen.append({"track_id":t,"sample_id":r.sample_id,"x_actual_mm":r.x_actual_mm,
   "tile_boundary_flag":bool(r.tile_boundary_flag),"primary_label_eligible":bool(r.primary_label_eligible),
   "sem_input_available":bool(r.sem_input_available),"figure_path":str(target.relative_to(ROOT))})
 pd.DataFrame(chosen).to_csv(out/"corrected_qa_selection.csv",index=False)
 return chosen

def main():
 for rel,h in EXPECTED_HEIGHT.items():
  if sha(ROOT/rel)!=h:raise RuntimeError(f"Frozen Height file changed: {rel}")
 if sha(CONF)!=EXPECTED_CONF:raise RuntimeError("Development confirmation hash does not match recorded build provenance")
 cfg=json.loads(CONF.read_text())
 for t in DEV:
  c=cfg["tracks"][str(t)]
  required=(c["confirmed"] is True and c["along_track_axis"]=="horizontal" and
   c["increasing_x_direction"]=="left_to_right" and c["tile_order"]==list(range(13,0,-1)))
  if not required:raise RuntimeError(f"Confirmed mapping content mismatch: Track {t}")
 shutil.copyfile(CONF,ARCHIVE)
 archive_hash=sha(ARCHIVE)
 if archive_hash!=EXPECTED_CONF:raise RuntimeError("Archived confirmation is not byte-identical")
 mapping_cfg=json.loads((ROOT/"configs/sem_mapping_deadline_v1.yaml").read_text())
 if mapping_cfg["confirmation_hash"]!=archive_hash:raise RuntimeError("Development mapping was not built from archived confirmation")
 mapping=pd.read_csv(ROOT/"outputs/sem_deadline/sem_mapping_deadline_v1.csv")
 if len(mapping)!=1192 or set(mapping.track_id)!=set(DEV):raise RuntimeError("Development mapping row reproduction failed")
 if not (mapping.along_track_axis.eq("horizontal").all() and mapping.increasing_x_direction.eq("left_to_right").all()):
  raise RuntimeError("Development mapping direction reproduction failed")

 m=load_module();geo,tc,mapdf,mandf,data_b=m.prepare(cfg)
 corrected=corrected_qa(m,geo,tc,mandf)
 # Cohort B fixed hyperparameters: exact predeclared alpha grid, complete-Track LOTO.
 xb=[dict(d,xfeat=m.xmat([d["x"]])[0]) for d in data_b]
 alpha_b={};audit={}
 for name,key,extra in [("x_only","xfeat",None),("fixed_thermal_shape_ridge","thermal",None),
   ("sem_summary_ridge","summary",None),("thermal_plus_sem_summary_ridge","summary","thermal"),
   ("sem_summary_upstream_ridge","up_summary",None)]:
  source=xb if name=="x_only" else data_b
  alpha_b[name],audit[name]=eval_alpha(m,source,key,extra)
 # Cohort A: all frozen primary eligible labels with fixed x/Thermal representation.
 labels=pd.read_csv(ROOT/"outputs/height_labels/height_labels_v1_development.csv")
 labels=labels[(labels.track_id.isin(DEV))&(labels.primary_label_eligible==True)&labels.target_width_mm.notna()].copy()
 therm=m.thermal_shape(labels)
 data_a=[{"sample_id":r.sample_id,"track_id":int(r.track_id),"x":float(r.x_actual_mm),"y":float(r.target_width_mm),
          "thermal":therm[r.sample_id]} for r in labels.itertuples()]
 xa=[dict(d,xfeat=m.xmat([d["x"]])[0]) for d in data_a]
 alpha_a={}
 for name,key,source in [("x_only","xfeat",xa),("fixed_thermal_shape_ridge","thermal",data_a)]:
  alpha_a[name],audit[f"cohort_a_{name}"]=eval_alpha(m,source,key)

 lock={"protocol_version":"final_height_width_model_lock_v1","created_before_final_test_source_open":True,
  "development_tracks":list(DEV),"final_test_track":21,
  "target":{"height_estimator":"height_boundary_estimator_v1","height_estimator_hash":EXPECTED_HEIGHT["configs/height_boundary_estimator_v1.yaml"],
   "label_policy":"all_finite_height_estimator_v1","label_policy_config_hash":EXPECTED_HEIGHT["configs/height_label_policy_v1.yaml"]},
  "mapping":{"confirmation_file":str(ARCHIVE.relative_to(ROOT)),"confirmation_hash":archive_hash,
   "mapping_type":"full_mosaic_endpoint_linear_fallback","along_track_axis":"horizontal",
   "increasing_x_direction":"left_to_right","tile_order":list(range(13,0,-1)),"physical_interval_mm":[20.0,100.0]},
  "mask":{"protocol_id":"symmetric_two_strip_1mm_mask0p4mm_v1","central_exclusion_half_width_mm":.4,
   "upstream_context_mm":1.0,"downstream_context_mm":1.0,"mask_before_preprocessing":True},
  "representations":{"x_only":["x_actual_mm","x_actual_mm_squared"],
   "thermal_shape_features":"fixed_predevelopment_shape_features",
   "sem_context_summary":"fixed_label_free_retained_context_summary"},
  "models":{"required":["training_mean","x_only","fixed_thermal_shape_ridge","sem_summary_ridge","thermal_plus_sem_summary_ridge"],
   "sensitivity":["sem_summary_upstream_ridge"],"ridge_alpha_grid":list(ALPHAS),
   "cohort_a_alphas":alpha_a,"cohort_b_alphas":alpha_b,
   "missing_value_handling":"training-cohort feature median","scaling":"training-cohort mean/std",
   "intercept_policy":"unpenalized","seed":20260725,"software_version":"numpy-transparent-ridge-v1"},
  "cohorts":{"A":"all primary_label_eligible frozen-v1 development rows",
   "B":"primary_label_eligible and leakage-safe SEM input available; fixed Thermal available when required"},
  "success_rules":{"primary":"sem_summary_ridge versus cohort-matched x_only; MAE reduction >= max(0.003 mm, 3%) without material bias worsening or tiny-region concentration",
   "secondary":"thermal_plus_sem_summary_ridge versus sem_summary_ridge; MAE reduction >= 0.002 mm without material bias worsening"},
  "development_alpha_selection_audit":audit}
 LOCK.write_text(json.dumps(lock,indent=2)+"\n");lock_hash=sha(LOCK)
 lock["lock_hash"]=lock_hash
 # Lock is immutable from this point; hash is recorded externally/artifacts, not inserted into itself.
 ART.mkdir(parents=True,exist_ok=True)
 sources={"development_labels":sha(ROOT/"outputs/height_labels/height_labels_v1_development.csv"),
  "development_sem_manifest":sha(ROOT/"outputs/manifests/sem_height_manifest_deadline_v1_development.csv"),
  "confirmation":archive_hash,"model_lock":lock_hash}
 # Transparent artifacts, with cohort-matched baselines.
 ta=np.array([d["track_id"] for d in data_a]);ya=np.array([d["y"] for d in data_a])
 tb=np.array([d["track_id"] for d in data_b]);yb=np.array([d["y"] for d in data_b])
 mean_a={"artifact_version":"final_height_width_v1","model_id":"cohort_a_training_mean","lock_hash":lock_hash,
  "intercept":float(ya.mean()),"coefficients":[],"feature_names":[],"training_sample_count":len(ya),
  "training_track_counts":{str(t):int(np.sum(ta==t)) for t in DEV},"source_hashes":sources}
 mean_b={**mean_a,"model_id":"cohort_b_training_mean","intercept":float(yb.mean()),"training_sample_count":len(yb),
  "training_track_counts":{str(t):int(np.sum(tb==t)) for t in DEV}}
 (ART/"cohort_a_training_mean.json").write_text(json.dumps(mean_a,indent=2)+"\n")
 (ART/"cohort_b_training_mean.json").write_text(json.dumps(mean_b,indent=2)+"\n")
 artifact("cohort_a_x_only",["x_actual_mm","x_actual_mm_squared"],np.stack([d["xfeat"] for d in xa]),ya,alpha_a["x_only"],ta,lock_hash,sources)
 artifact("cohort_a_fixed_thermal_shape_ridge",[f"thermal_shape_{i:02d}" for i in range(len(data_a[0]["thermal"]))],
  np.stack([d["thermal"] for d in data_a]),ya,alpha_a["fixed_thermal_shape_ridge"],ta,lock_hash,sources)
 artifact("cohort_b_x_only",["x_actual_mm","x_actual_mm_squared"],np.stack([d["xfeat"] for d in xb]),yb,alpha_b["x_only"],tb,lock_hash,sources)
 artifact("cohort_b_fixed_thermal_shape_ridge",[f"thermal_shape_{i:02d}" for i in range(len(data_b[0]["thermal"]))],
  np.stack([d["thermal"] for d in data_b]),yb,alpha_b["fixed_thermal_shape_ridge"],tb,lock_hash,sources)
 artifact("cohort_b_sem_summary_ridge",[f"sem_summary_{i:02d}" for i in range(len(data_b[0]["summary"]))],
  np.stack([d["summary"] for d in data_b]),yb,alpha_b["sem_summary_ridge"],tb,lock_hash,sources)
 artifact("cohort_b_thermal_plus_sem_summary_ridge",
  [f"sem_summary_{i:02d}" for i in range(len(data_b[0]["summary"]))]+[f"thermal_shape_{i:02d}" for i in range(len(data_b[0]["thermal"]))],
  np.stack([np.r_[d["summary"],d["thermal"]] for d in data_b]),yb,alpha_b["thermal_plus_sem_summary_ridge"],tb,lock_hash,sources)
 artifact("cohort_b_sem_summary_upstream_ridge",[f"sem_upstream_summary_{i:02d}" for i in range(len(data_b[0]["up_summary"]))],
  np.stack([d["up_summary"] for d in data_b]),yb,alpha_b["sem_summary_upstream_ridge"],tb,lock_hash,sources)
 (ART/"lock_hash.txt").write_text(lock_hash+"\n")
 (ART/"feature_schema.json").write_text(json.dumps(lock["representations"],indent=2)+"\n")
 (ROOT/"outputs/reports/sem_mapping_provenance_reconciliation.md").write_text(
  f"""# SEM mapping provenance reconciliation

- Original development confirmation hash: `{EXPECTED_CONF}`
- Final archived confirmation hash: `{archive_hash}`
- Recovery status: **exactly recovered and byte-identical**
- Development mapping confirmation hash: `{mapping_cfg['confirmation_hash']}`
- Existing development mapping rows reproduced: **1192**
- Direction/order reproduced: horizontal, left-to-right, tiles 13 through 01.
- Discrepancy: none.

The 20–100 mm interval applies to the complete tile sequence, never to an individual tile.
""")
 (ROOT/"outputs/reports/final_height_width_model_lock_v1.md").write_text(
  f"""# Final Height-width model lock v1

- Lock hash: `{lock_hash}`
- Development Tracks: 8, 10, 14.
- Final held-out Track: 21.
- Cohort A development rows: {len(data_a)}.
- Cohort B development rows: {len(data_b)}.
- Required final models: training mean, x-only, fixed Thermal shape ridge, SEM summary ridge, Thermal+SEM summary ridge.
- Sensitivity only: upstream SEM summary ridge.
- No PCA, CNN, new feature search, or random position split.
- Locked cohort-A alphas: `{json.dumps(alpha_a,sort_keys=True)}`.
- Locked cohort-B alphas: `{json.dumps(alpha_b,sort_keys=True)}`.
- Mapping confirmation was archived before final-test source inspection.
""")
 print(f"confirmation={archive_hash} lock={lock_hash} cohortA={len(data_a)} cohortB={len(data_b)} artifacts={len(list(ART.glob('*.json')))}")
if __name__=="__main__":main()
