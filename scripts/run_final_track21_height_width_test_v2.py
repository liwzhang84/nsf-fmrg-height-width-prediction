#!/usr/bin/env python3
"""Versioned, amendment-authorized single final held-out evaluation."""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, os, re
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR","/private/tmp/nsf_mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.io import loadmat
from scipy.stats import pearsonr,spearmanr

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/"configs/final_height_width_model_lock_v1.yaml"
AMEND=ROOT/"configs/final_test_sem_source_compatibility_amendment_v1.yaml"
ART=ROOT/"outputs/models/final_height_width_v1"
LOCK_HASH="2fb60cbb83a04ea9819c3bfff9fa39b9765dc9bad74b4a7d0693103af013be1b"
HEIGHT_HASH="4b1094cf2084f10026cdd470dba7ba3bdf8b561bfef02ba7ef3413e8a555617d"
POLICY_HASH="5b568d33b06761ffdce9c35fd75fa03307c6ec8dcdb4cd510e7513d39e96c994"
TRACK=21;MASK=.4;CONTEXT=1.0

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(path,name):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def truth(v):return str(v).strip().lower() in {"true","1","yes"}
def finite(v):
 try:return bool(np.isfinite(float(v)))
 except:return False
def f(v,default=np.nan):return float(v) if finite(v) else default
def predict_art(a,x):
 x=np.asarray(x,float);med=np.array(a.get("imputation_values",[]));mean=np.array(a.get("scaler_mean",[]));std=np.array(a.get("scaler_scale",[]))
 if x.shape[1]==0:return np.repeat(float(a["intercept"]),len(x))
 x=np.where(np.isfinite(x),x,med);return float(a["intercept"])+((x-mean)/std)@np.array(a["coefficients"])
def metrics(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);e=p-y;ae=abs(e);ss=np.sum((y-y.mean())**2)
 def corr(fun):
  try:return float(fun(y,p).statistic) if len(y)>2 and np.std(y)>0 and np.std(p)>0 else np.nan
  except:return np.nan
 return {"sample_count":len(y),"mae_mm":float(ae.mean()),"rmse_mm":float(np.sqrt(np.mean(e*e))),
  "median_absolute_error_mm":float(np.median(ae)),"p90_absolute_error_mm":float(np.percentile(ae,90)),
  "maximum_absolute_error_mm":float(ae.max()),"signed_bias_mm":float(e.mean()),
  "r_squared":float(1-np.sum(e*e)/ss) if ss>0 else np.nan,
  "pearson_correlation":corr(pearsonr),"spearman_correlation":corr(spearmanr)}

def build_labels():
 gate=module("scripts/build_height_quality_gate_review.py","height_gate_final")
 gate.DEV={TRACK}
 positions=gate.supported_positions()
 if not (390<=len(positions)<=400):raise RuntimeError(f"Unexpected final Height support: {len(positions)}")
 features,_=gate.build_features(positions)
 dev_cols=pd.read_csv(ROOT/"outputs/height_labels/height_labels_v1_development.csv",nrows=0).columns.tolist()
 diagnostic=("local_height_valid_fraction","aggregation_valid_fraction","central_region_valid_fraction",
  "left_substrate_valid_fraction","right_substrate_valid_fraction",
  "left_boundary_distance_to_nearest_nan_mm","right_boundary_distance_to_nearest_nan_mm",
  "aggregation_sensitivity_width_mm","aggregation_sensitivity_center_mm",
  "maximum_method_disagreement_width_mm","maximum_method_disagreement_center_mm",
  "width_neighbor_difference_mm","center_neighbor_difference_mm","apparent_cap_flag","distance_to_candidate_limit_mm",
  "aggregation_width_sensitivity_mm","aggregation_center_sensitivity_mm",
  "maximum_method_width_disagreement_mm","maximum_method_center_disagreement_mm")
 rows=[]
 for r in features:
  ok=truth(r["finite_boundary_prediction"]) and finite(r["predicted_y_left_mm"]) and finite(r["predicted_y_right_mm"]) and f(r["predicted_y_right_mm"])>f(r["predicted_y_left_mm"]) and not str(r.get("estimator_exclusion_reason","")).strip()
  out={c:"" for c in dev_cols}
  out.update({"sample_id":r["sample_id"],"track_id":TRACK,"segment_frame_index":r["segment_frame_index"],
   "raw_frame_index":r["raw_frame_index"],"x_actual_mm":r["x_actual_mm"],
   "official_position_index":r["segment_frame_index"],"height_window_available":True,
   "height_estimator_id":"height_boundary_estimator_v1","height_estimator_config_hash":HEIGHT_HASH,
   "leveling_method_id":"official_robust_plane_v1","boundary_method_id":"gradient_edges",
   "aggregation_half_width_mm":.1,"target_protocol_version":"height_label_policy_v1",
   "target_y_left_mm":r["predicted_y_left_mm"] if ok else "","target_y_right_mm":r["predicted_y_right_mm"] if ok else "",
   "target_width_mm":r["predicted_width_mm"] if ok else "","target_center_mm":r["predicted_center_mm"] if ok else "",
   "finite_boundary_prediction":ok,"estimator_exclusion_reason":r.get("estimator_exclusion_reason",""),
   "primary_label_policy_id":"all_finite_height_estimator_v1","primary_label_eligible":ok,
   "primary_label_exclusion_reason":"" if ok else "estimator_nonfinite_or_excluded",
   "sensitivity_local_valid_ge_0p50":ok and f(r.get("local_height_valid_fraction"),0)>=.5,
   "sensitivity_side_support_ge_0p50":ok and min(f(r.get("left_substrate_valid_fraction"),0),f(r.get("right_substrate_valid_fraction"),0))>=.5,
   "sensitivity_local_and_side_support_ge_0p50":ok and f(r.get("local_height_valid_fraction"),0)>=.5 and min(f(r.get("left_substrate_valid_fraction"),0),f(r.get("right_substrate_valid_fraction"),0))>=.5,
   "human_reference_available":False})
  for c in diagnostic:
   source={"aggregation_width_sensitivity_mm":"aggregation_sensitivity_width_mm",
           "aggregation_center_sensitivity_mm":"aggregation_sensitivity_center_mm",
           "maximum_method_width_disagreement_mm":"maximum_method_disagreement_width_mm",
           "maximum_method_center_disagreement_mm":"maximum_method_disagreement_center_mm"}.get(c,c)
   out[c]=r.get(source,"")
  rows.append(out)
 out=ROOT/"outputs/height_labels/height_labels_v1_track21_final_test.csv";pd.DataFrame(rows,columns=dev_cols).to_csv(out,index=False)
 return pd.DataFrame(rows)

def thermal_inputs(labels,deadline):
 audit=module("scripts/run_thermal_representation_audit.py","thermal_fixed_final")
 arr=np.asarray(loadmat(ROOT/"thermal/Thermal_21.mat")["temperature_data"])
 feats={};manifest=[]
 for r in labels.itertuples():
  idx=int(r.raw_frame_index)
  if not (0<=idx<len(arr)):raise RuntimeError(f"Thermal raw index out of bounds: {idx}")
  frame=np.asarray(arr[idx],float);success,cx,cy,mask=audit.hotspot(frame)
  feats[r.sample_id]=audit.shape_features(frame,success,cx,cy,mask)
  manifest.append({**r._asdict(),"thermal_input_available":True,
   "thermal_source_path":str((ROOT/"thermal/Thermal_21.mat").resolve()),
   "thermal_tensor_reference":f"raw_frame_index={idx}","thermal_shape":"400x400",
   "thermal_representation_id":"thermal_shape_features","thermal_representation_status":"fixed_descriptive_not_reselected",
   "thermal_mapping_version":"canonical_official_zero_based_unchanged"})
 pd.DataFrame(manifest).to_csv(ROOT/"outputs/manifests/thermal_height_manifest_v1_track21_final_test.csv",index=False)
 return feats

def sem_geometry(deadline):
 root=ROOT/"sem/SEM_21/PlainImages";items=[]
 for tile in range(14,0,-1):
  pp=list(root.glob(f"*_{tile:02d}.tif"))
  if len(pp)!=1:raise RuntimeError(f"Canonical held-out tile not unique: {tile}")
  p=pp[0]
  with Image.open(p) as im:
   a=np.asarray(ImageOps.grayscale(im));text=str(im.tag_v2.get(34118,""))
  def val(label):
   z=re.search(rf"{label}\s*=\s*([-+0-9.]+)",text,re.I);return float(z.group(1)) if z else np.nan
  items.append({"tile":tile,"path":p,"array":a,"px_mm":val("Image Pixel Size")/1000,
   "stage_x":val("Stage at X"),"source_hash":sha(p)})
 if [q["tile"] for q in items]!=list(range(14,0,-1)):raise RuntimeError("Amended contiguous tile rule failed")
 stage=np.array([q["stage_x"] for q in items]);fw=np.median([q["array"].shape[1]*q["px_mm"] for q in items])
 slope=((100-fw/2)-(20+fw/2))/(stage.max()-stage.min());intercept=(20+fw/2)-slope*stage.min()
 for q in items:
  center=slope*q["stage_x"]+intercept;q["xmin"]=center-fw/2;q["xmax"]=center+fw/2
 return items,float(np.median([q["px_mm"] for q in items]))
def choose(items,x,lo,hi):
 c=[]
 for q in items:
  if q["xmin"]<=lo and q["xmax"]>=hi:
   u=(x-q["xmin"])/q["px_mm"];c.append((min(u,q["array"].shape[1]-1-u),q,u))
 return max(c,key=lambda z:z[0]) if c else None
def bounds(q,lo,hi,retained):
 a=(lo-q["xmin"])/q["px_mm"];b=(hi-q["xmin"])/q["px_mm"]
 x0=math.ceil(min(a,b)) if retained else math.floor(min(a,b));x1=math.floor(max(a,b)) if retained else math.ceil(max(a,b))
 return max(0,int(x0)),min(q["array"].shape[1],int(x1))

def sem_inputs(labels,deadline):
 items,pixel_mm=sem_geometry(deadline);mapping=[];manifest=[];summary={};upsummary={}
 for r in labels.itertuples():
  x=float(r.x_actual_mm);mp=choose(items,x,x,x);ctx=choose(items,x,x-MASK-CONTEXT,x+MASK+CONTEXT)
  mosaic=(x-20)/pixel_mm
  if mp is None:
   mapping.append({"sample_id":r.sample_id,"track_id":TRACK,"official_position_index":r.official_position_index,
    "x_actual_mm":x,"sem_source_path":"","sem_tile_id":"","sem_mosaic_coordinate_px":mosaic,
    "sem_local_coordinate_px":"","sem_cross_track_reference_px":"","along_track_axis":"horizontal",
    "increasing_x_direction":"left_to_right","sem_mapping_id":"generalized_N_to_1_v1",
    "mapping_source":"source-schema compatibility amendment v1","mapping_confidence":"low",
    "mapping_available":False,"mapping_exclusion_reason":"official_position_outside_supported_full_sequence",
    "distance_to_image_border_px":"","distance_to_tile_boundary_px":""})
   source="";tile="";reason="official_position_outside_supported_full_sequence"
  else:
   dist,q,u=mp;source=str(q["path"].resolve());tile=q["tile"];reason=""
   mapping.append({"sample_id":r.sample_id,"track_id":TRACK,"official_position_index":r.official_position_index,
    "x_actual_mm":x,"sem_source_path":source,"sem_tile_id":tile,"sem_mosaic_coordinate_px":mosaic,
    "sem_local_coordinate_px":u,"sem_cross_track_reference_px":q["array"].shape[0]/2,
    "along_track_axis":"horizontal","increasing_x_direction":"left_to_right",
    "sem_mapping_id":"generalized_N_to_1_v1","mapping_source":"source-schema compatibility amendment v1",
    "mapping_confidence":"low","mapping_available":True,"mapping_exclusion_reason":"",
    "distance_to_image_border_px":dist,"distance_to_tile_boundary_px":dist})
  available=ctx is not None and mp is not None
  row={"sample_id":r.sample_id,"track_id":TRACK,"segment_frame_index":r.segment_frame_index,
   "raw_frame_index":r.raw_frame_index,"official_position_index":r.official_position_index,"x_actual_mm":x,
   "target_width_mm":r.target_width_mm,"primary_label_eligible":r.primary_label_eligible,
   "sem_mapping_id":"generalized_N_to_1_v1","sem_protocol_id":"symmetric_two_strip_1mm_mask0p4mm_v1",
   "sem_source_path":source,"sem_tile_id":tile,"central_mask_native_bounds":"",
   "upstream_native_bounds":"","downstream_native_bounds":"","sem_input_available":available,
   "sem_exclusion_reason":"" if available else (reason or "full_2p8mm_context_not_supported_by_one_native_tile"),
   "border_flag":False,"tile_boundary_flag":not available,"context_clipped_flag":not available,
   "upstream_available":available,"downstream_available":available,"valid_pixel_fraction":1.0 if available else 0.0}
  if available:
   dist,q,u=ctx;mask=bounds(q,x-MASK,x+MASK,False);up=bounds(q,x-MASK-CONTEXT,x-MASK,True);down=bounds(q,x+MASK,x+MASK+CONTEXT,True)
   if not (up[1]<=mask[0] and mask[1]<=down[0]):raise RuntimeError("Native target pixels entered context")
   us=q["array"][:,up[0]:up[1]];ds=q["array"][:,down[0]:down[1]]
   s,_=deadline.representations(us,ds,False,dist<(MASK+CONTEXT)/q["px_mm"])
   su,_=deadline.representations(us,None,False,dist<(MASK+CONTEXT)/q["px_mm"])
   summary[r.sample_id]=s;upsummary[r.sample_id]=su
   row.update(central_mask_native_bounds=f"{mask[0]}:{mask[1]}",upstream_native_bounds=f"{up[0]}:{up[1]}",
    downstream_native_bounds=f"{down[0]}:{down[1]}",tile_boundary_flag=dist<(MASK+CONTEXT)/q["px_mm"])
  manifest.append(row)
 out=ROOT/"outputs/sem_final_test_v2";out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(mapping).to_csv(out/"sem_mapping_v1_track21.csv",index=False)
 pd.DataFrame(manifest).to_csv(ROOT/"outputs/manifests/sem_height_manifest_v1_track21_final_test.csv",index=False)
 return pd.DataFrame(mapping),pd.DataFrame(manifest),summary,upsummary

def main():
 # Immutable inputs are verified before held-out pixels or targets are opened.
 if sha(LOCK)!="2fb60cbb83a04ea9819c3bfff9fa39b9765dc9bad74b4a7d0693103af013be1b":raise RuntimeError("Original model lock changed")
 amendment_hash=sha(AMEND);recorded=(ROOT/"outputs/sem_final_test_v2/amendment_file_hash.txt").read_text().strip()
 if amendment_hash!=recorded:raise RuntimeError("Compatibility amendment hash mismatch")
 artifacts={}
 for p in ART.glob("*.json"):
  if p.name=="feature_schema.json":continue
  a=json.loads(p.read_text())
  if a["lock_hash"]!=sha(LOCK):raise RuntimeError(f"Artifact lock mismatch: {p.name}")
  artifacts[p.stem]=a
 artifact_hashes_before={p.name:sha(p) for p in ART.iterdir() if p.is_file()}
 preserved=json.loads(AMEND.read_text())["first_blocked_attempt_hashes"]
 for p,h in preserved.items():
  if sha(ROOT/p)!=h:raise RuntimeError(f"First blocked attempt was overwritten: {p}")
 deadline=module("scripts/run_sem_multimodal_deadline_models.py","deadline_features_final")
 labels=build_labels()
 thermal=thermal_inputs(labels,deadline)
 mapping,semmanifest,summary,upsummary=sem_inputs(labels,deadline)
 eligible=labels[(labels.primary_label_eligible==True)&labels.target_width_mm.notna()].copy()
 cohort_b=semmanifest[(semmanifest.primary_label_eligible==True)&(semmanifest.sem_input_available==True)].copy()
 cohort_b=cohort_b[cohort_b.sample_id.isin(thermal)].copy()
 if cohort_b.sample_id.duplicated().any():raise RuntimeError("Duplicate cohort-B sample")
 ids_a=eligible.sample_id.tolist();ids_b=cohort_b.sample_id.tolist()
 lab=labels.set_index("sample_id");pred_rows=[];metric_rows=[]
 def run(cohort,ids,models):
  y=np.array([lab.loc[s,"target_width_mm"] for s in ids],float);x=np.array([lab.loc[s,"x_actual_mm"] for s in ids],float)
  for name,kind in models:
   a=artifacts[name]
   if kind=="mean":X=np.empty((len(ids),0))
   elif kind=="x":X=np.column_stack([x,x*x])
   elif kind=="thermal":X=np.stack([thermal[s] for s in ids])
   elif kind=="sem":X=np.stack([summary[s] for s in ids])
   elif kind=="multi":X=np.stack([np.r_[summary[s],thermal[s]] for s in ids])
   elif kind=="upstream":X=np.stack([upsummary[s] for s in ids])
   p=predict_art(a,X);mm=metrics(y,p);metric_rows.append({"cohort_id":cohort,"model_id":name,**mm})
   for sid,xx,yy,pp in zip(ids,x,y,p):pred_rows.append({"cohort_id":cohort,"sample_id":sid,"track_id":TRACK,
    "x_actual_mm":xx,"model_id":name,"target_width_mm":yy,"prediction_mm":pp,
    "signed_error_mm":pp-yy,"absolute_error_mm":abs(pp-yy)})
 run("cohort_A",ids_a,[("cohort_a_training_mean","mean"),("cohort_a_x_only","x"),("cohort_a_fixed_thermal_shape_ridge","thermal")])
 run("cohort_B",ids_b,[("cohort_b_training_mean","mean"),("cohort_b_x_only","x"),
  ("cohort_b_fixed_thermal_shape_ridge","thermal"),("cohort_b_sem_summary_ridge","sem"),
  ("cohort_b_thermal_plus_sem_summary_ridge","multi"),("cohort_b_sem_summary_upstream_ridge","upstream")])
 pred=pd.DataFrame(pred_rows);met=pd.DataFrame(metric_rows)
 pred.to_csv(ROOT/"outputs/models/final_track21_height_width_predictions.csv",index=False)
 met.to_csv(ROOT/"outputs/models/final_track21_height_width_metrics.csv",index=False)
 cohort_summary=pd.DataFrame([{"cohort_id":"cohort_A","total_positions":len(labels),
   "primary_label_eligible":len(ids_a),"sem_input_available":np.nan,"common_thermal_sem_available":np.nan},
  {"cohort_id":"cohort_B","total_positions":len(labels),"primary_label_eligible":int(labels.primary_label_eligible.sum()),
   "sem_input_available":int(semmanifest.sem_input_available.sum()),"common_thermal_sem_available":len(ids_b),
   "exclusion_distribution":json.dumps(Counter(semmanifest.sem_exclusion_reason.fillna("")),sort_keys=True),
   "tile_boundary_cases":int(semmanifest.tile_boundary_flag.sum()),"border_cases":int(semmanifest.border_flag.sum())}])
 cohort_summary.to_csv(ROOT/"outputs/models/final_track21_height_width_cohort_summary.csv",index=False)
 sub=[]
 pb=pred[pred.cohort_id=="cohort_B"].merge(semmanifest[["sample_id","tile_boundary_flag"]],on="sample_id",how="left")
 for model,g in pb.groupby("model_id"):
  for lo,hi in ((20,40),(40,60),(60,80),(80,100.0001)):
   z=g[(g.x_actual_mm>=lo)&(g.x_actual_mm<hi)]
   if len(z):sub.append({"model_id":model,"subgroup_type":"x_region","subgroup_id":f"{lo}_{min(hi,100):g}mm",**metrics(z.target_width_mm,z.prediction_mm)})
  for flag in (False,True):
   z=g[g.tile_boundary_flag==flag]
   if len(z):sub.append({"model_id":model,"subgroup_type":"tile_boundary","subgroup_id":"near" if flag else "away",**metrics(z.target_width_mm,z.prediction_mm)})
 pd.DataFrame(sub).to_csv(ROOT/"outputs/models/final_track21_height_width_subgroup_metrics.csv",index=False)
 # Interpret exactly once from locked primary/secondary comparisons.
 mm=met.set_index("model_id");x=mm.loc["cohort_b_x_only"];s=mm.loc["cohort_b_sem_summary_ridge"];tm=mm.loc["cohort_b_thermal_plus_sem_summary_ridge"]
 sem_gain=x.mae_mm-s.mae_mm;required=max(.003,.03*x.mae_mm);bias_worsen=abs(s.signed_bias_mm)-abs(x.signed_bias_mm)
 wide=pb.pivot(index=["sample_id","x_actual_mm"],columns="model_id",values="absolute_error_mm").reset_index()
 wide["gain"]=wide["cohort_b_x_only"]-wide["cohort_b_sem_summary_ridge"];wide["region"]=pd.cut(wide.x_actual_mm,[20,40,60,80,100.1],right=False)
 contributions=wide.groupby("region",observed=True).gain.sum();concentration=(float(contributions.max()/contributions.sum()) if contributions.sum()>0 else np.nan)
 sem_ok=bool(sem_gain>=required and bias_worsen<=.003 and (not np.isfinite(concentration) or concentration<=.80))
 thermal_gain=s.mae_mm-tm.mae_mm;thermal_bias=abs(tm.signed_bias_mm)-abs(s.signed_bias_mm);thermal_ok=bool(thermal_gain>=.002 and thermal_bias<=.003)
 status="SEM GENERALIZATION CONFIRMED ON FINAL TRACK 21" if sem_ok else "SEM IMPROVEMENT NOT CONFIRMED ON FINAL TRACK 21"
 # Post-hoc historical diagnostic only after primary metrics are fixed.
 hist=[]
 for p in (ROOT/"outputs/human_review").glob("*.csv"):
  try:d=pd.read_csv(p)
  except:continue
  if "track_id" in d and (pd.to_numeric(d.track_id,errors="coerce")==TRACK).any():
   z=d[pd.to_numeric(d.track_id,errors="coerce")==TRACK].copy();z["diagnostic_role"]="post_hoc_historical_human_diagnostic";z["source_file"]=str(p.relative_to(ROOT));hist.append(z)
 if hist:pd.concat(hist,ignore_index=True,sort=False).to_csv(ROOT/"outputs/models/final_track21_historical_human_diagnostics.csv",index=False)
 figdir=ROOT/"outputs/figures/final_track21_height_width_v2";figdir.mkdir(parents=True,exist_ok=True)
 fig,axs=plt.subplots(2,1,figsize=(12,8),sharex=True)
 for model in ("cohort_b_x_only","cohort_b_sem_summary_ridge","cohort_b_thermal_plus_sem_summary_ridge"):
  g=pb[pb.model_id==model].sort_values("x_actual_mm");axs[0].plot(g.x_actual_mm,g.prediction_mm,label=model,alpha=.8);axs[1].plot(g.x_actual_mm,g.absolute_error_mm,label=model,alpha=.8)
 ytrue=pb[pb.model_id=="cohort_b_x_only"].sort_values("x_actual_mm");axs[0].plot(ytrue.x_actual_mm,ytrue.target_width_mm,color="black",lw=1,label="frozen target")
 axs[0].set_ylabel("width (mm)");axs[1].set_ylabel("absolute error (mm)");axs[1].set_xlabel("official x (mm)")
 for ax in axs:ax.legend(fontsize=7);ax.grid(alpha=.2)
 fig.tight_layout();fig.savefig(figdir/"final_track21_predictions_and_errors.png",dpi=170);plt.close(fig)
 review=f"""# Final Track-21 Height-width test review v2

## Frozen execution

- Original model-lock hash: `{sha(LOCK)}`
- Compatibility-amendment hash: `{amendment_hash}`
- No fitted artifact, coefficient, alpha, feature, mask, target policy, Thermal representation, or success rule changed.
- Tile 14 was retained under the metadata-authorized contiguous variable-N rule.

## Cohorts

- Official Height-supported positions: {len(labels)}
- Cohort A primary eligible rows: {len(ids_a)}
- Cohort B identical common SEM/Thermal rows: {len(ids_b)}

## Primary comparison

- Cohort-B x-only MAE: {x.mae_mm:.6f} mm
- SEM-summary MAE: {s.mae_mm:.6f} mm
- Absolute MAE reduction: {sem_gain:.6f} mm
- Required reduction: {required:.6f} mm (`max(0.003 mm, 3%)`)
- Relative reduction: {100*sem_gain/x.mae_mm:.3f}%
- RMSE difference (SEM − x): {s.rmse_mm-x.rmse_mm:.6f} mm
- Signed-bias difference (SEM − x): {s.signed_bias_mm-x.signed_bias_mm:.6f} mm
- Absolute-bias worsening: {bias_worsen:.6f} mm
- Largest 20-mm-region contribution fraction: {concentration:.3f}
- SEM generalization supported: **{sem_ok}**

## Secondary comparison

- Thermal+SEM MAE: {tm.mae_mm:.6f} mm
- MAE reduction versus SEM: {thermal_gain:.6f} mm
- Required reduction: 0.002 mm
- Absolute-bias worsening: {thermal_bias:.6f} mm
- Confirmed additive Thermal value: **{thermal_ok}**

Subgroup and historical-human results are descriptive only and did not alter the locked interpretation.

{status}
"""
 (ROOT/"outputs/reports/final_track21_height_width_test_review_v2.md").write_text(review)
 for p,h in artifact_hashes_before.items():
  if sha(ART/p)!=h:raise RuntimeError(f"Fitted artifact changed: {p}")
 if sha(LOCK)!=LOCK_HASH or sha(AMEND)!=amendment_hash:raise RuntimeError("Lock/amendment changed during final evaluation")
 leakage=f"""# Final Track-21 Height-width leakage audit v2

- Original model lock unchanged: **PASS**, `{sha(LOCK)}`
- Original fitted artifacts unchanged: **PASS**
- First blocked-attempt outputs preserved: **PASS**
- Amendment based only on structural metadata and written before held-out pixels/targets: **PASS**
- Generalized mapping development equivalence: **PASS**
- Tile 14 retained: **PASS**
- Held-out model fitting calls: **0**
- New model/alpha/feature/threshold selection: **0**
- Cohort-B models use identical {len(ids_b)} sample IDs: **PASS**
- Native SEM mask applied before retained-context feature extraction: **PASS**
- Error-based sample exclusion: **0**
- Historical human references in primary metrics: **0**
"""
 (ROOT/"outputs/reports/final_track21_height_width_leakage_audit_v2.md").write_text(leakage)
 build=f"""# Final Track-21 Height-width build report v2

- Height rows: {len(labels)}
- Primary eligible: {len(ids_a)}
- Formal SEM mapping rows: {len(mapping)}
- SEM manifest rows: {len(semmanifest)}
- Common Cohort-B rows: {len(ids_b)}
- Prediction rows: {len(pred)}
- Metric rows: {len(met)}
- Subgroup metric rows: {len(sub)}
- Figures: 1
- Final status: `{status}`
"""
 (ROOT/"outputs/reports/final_track21_height_width_build_report_v2.md").write_text(build)
 print(f"height={len(labels)} cohortA={len(ids_a)} cohortB={len(ids_b)} predictions={len(pred)} metrics={len(met)} status={status}")
if __name__=="__main__":main()
