#!/usr/bin/env python3
"""Leakage-safe complete-Track Thermal-only baselines for frozen Height-v1 width."""
from __future__ import annotations
import csv,hashlib,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr,spearmanr

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from nsf_fmrg_data import extract_final_thermal_frames
from scripts.build_height_quality_gate_review import truth,write

DEV=(8,10,14);SEED=20260725
HL=PROJECT/"outputs/height_labels";MP=PROJECT/"outputs/mappings";MN=PROJECT/"outputs/manifests"
MO=PROJECT/"outputs/models";RP=PROJECT/"outputs/reports";FIGP=PROJECT/"outputs/figures/height_label_policy_v1"
FIGM=PROJECT/"outputs/figures/thermal_height_baselines"
PRED=HL/"height_boundary_predictions_v1_development.csv"
SUPPORT=MP/"thermal_height_support.csv";THERMAL_MAP=MP/"thermal_frame_mapping_official.csv"
REFS=PROJECT/"outputs/human_review/height_human_reference_v3_development.csv"
V1=PROJECT/"configs/height_boundary_estimator_v1.yaml"
POLICY=PROJECT/"configs/height_label_policy_v1.yaml"
V1_HASH="4b1094cf2084f10026cdd470dba7ba3bdf8b561bfef02ba7ef3413e8a555617d"
POLICIES=("all_finite_height_estimator_v1","local_valid_ge_0p50",
 "side_support_ge_0p50","local_and_side_support_ge_0p50")
FAMILIES=("training_mean","x_only","thermal_summary_ridge","thermal_profile_ridge",
          "thermal_summary_plus_x_ridge")
RIDGE=(.1,1.,10.,100.)

def rows(path):
 with Path(path).open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))
def finite(x):
 try:return bool(np.isfinite(float(x)))
 except:return False
def fv(row,key,default=np.nan):return float(row[key]) if finite(row.get(key)) else default
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def safe_corr(fun,y,p):
 try:return float(fun(y,p).statistic) if len(y)>2 and np.std(y)>0 and np.std(p)>0 else np.nan
 except:return np.nan
def metrics(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);e=p-y;ae=abs(e)
 ss=np.sum((y-y.mean())**2)
 return {"number_of_samples":len(y),"mae_mm":float(ae.mean()),"rmse_mm":float(np.sqrt(np.mean(e**2))),
  "median_absolute_error_mm":float(np.median(ae)),"p90_absolute_error_mm":float(np.percentile(ae,90)),
  "maximum_absolute_error_mm":float(ae.max()),"signed_bias_mm":float(e.mean()),
  "r_squared":float(1-np.sum(e**2)/ss) if ss>0 else np.nan,
  "pearson_correlation":safe_corr(pearsonr,y,p),"spearman_correlation":safe_corr(spearmanr,y,p)}
def ridge_fit(x,y,alpha):
 x=np.asarray(x,float);y=np.asarray(y,float)
 return np.linalg.solve(x.T@x+alpha*np.eye(x.shape[1]),x.T@y)
def fit_scaler(x):
 mean=np.mean(x,axis=0);scale=np.std(x,axis=0);scale[scale<1e-8]=1
 return mean,scale
def summary_features(frame):
 a=np.asarray(frame,float);baseline=np.percentile(a,10);hot=a>=np.percentile(a,95)
 yy,xx=np.indices(a.shape);weights=np.maximum(a-baseline,0);total=weights.sum()
 gx=np.diff(a,axis=1);gy=np.diff(a,axis=0);h,w=a.shape
 return np.array([a.mean(),a.std(),a.min(),a.max(),np.percentile(a,10),np.percentile(a,50),
  np.percentile(a,90),np.percentile(a,99),a.max()-baseline,hot.mean(),
  (weights*xx).sum()/max(total,1)/(w-1),(weights*yy).sum()/max(total,1)/(h-1),
  a[:,:w//2].mean()-a[:,w//2:].mean(),a[:h//2].mean()-a[h//2:].mean(),
  np.mean(abs(gx)),np.mean(abs(gy)),np.mean(np.maximum(a-baseline,0))],float)
def profile_features(frame):
 # Transparent sample-local 20x20 block-average representation; no cross-sample fitting.
 return np.asarray(frame,float).reshape(20,20,20,20).mean(axis=(1,3)).ravel()
def policy_flag(row,policy):
 if not truth(row["primary_label_eligible"]):return False
 local=fv(row,"local_height_valid_fraction",0)>=.5
 sides=fv(row,"left_substrate_valid_fraction",0)>=.5 and fv(row,"right_substrate_valid_fraction",0)>=.5
 return {"all_finite_height_estimator_v1":True,"local_valid_ge_0p50":local,
  "side_support_ge_0p50":sides,"local_and_side_support_ge_0p50":local and sides}[policy]

def design(family):
 if family=="training_mean":return [{"complexity":0,"parameter_id":"mean"}]
 if family=="x_only":
  return [{"complexity":d,"parameter_id":f"poly_degree_{d}_ridge_{a:g}","degree":d,"alpha":a}
          for d in (1,2,3) for a in RIDGE]
 return [{"complexity":i,"parameter_id":f"ridge_{a:g}","alpha":a} for i,a in enumerate(RIDGE)]
def matrix(family,indices,x,summary,profile,degree=None):
 if family=="training_mean":return np.ones((len(indices),1))
 if family=="x_only":return np.column_stack([x[indices]**d for d in range(1,degree+1)])
 if family=="thermal_summary_ridge":return summary[indices]
 if family=="thermal_profile_ridge":return profile[indices]
 if family=="thermal_summary_plus_x_ridge":return np.column_stack([summary[indices],x[indices]])
 raise ValueError(family)
def fit_predict(family,param,train,val,y,x,summary,profile):
 if family=="training_mean":return np.full(len(val),np.mean(y[train])),{"preprocessing":"training_mean"}
 xt=matrix(family,train,x,summary,profile,param.get("degree"));xv=matrix(family,val,x,summary,profile,param.get("degree"))
 mean,scale=fit_scaler(xt);xt=(xt-mean)/scale;xv=(xv-mean)/scale
 xt=np.column_stack([np.ones(len(xt)),xt]);xv=np.column_stack([np.ones(len(xv)),xv])
 coef=ridge_fit(xt,y[train],param["alpha"])
 return xv@coef,{"feature_mean":mean,"feature_scale":scale,"coefficients":coef,
                 "preprocessing":"training_fold_only_standardization"}
def choose(family,train_tracks,eligible,track,y,x,summary,profile):
 candidates=[]
 for param in design(family):
  fold_mae=[]
  for held in train_tracks:
   tr=np.flatnonzero(eligible & np.isin(track,[t for t in train_tracks if t!=held]))
   va=np.flatnonzero(eligible & (track==held))
   pred,_=fit_predict(family,param,tr,va,y,x,summary,profile)
   fold_mae.append(np.mean(abs(pred-y[va])))
  candidates.append((np.mean(fold_mae),param.get("complexity",0),param["parameter_id"],param))
 return min(candidates,key=lambda z:(z[0],z[1],z[2]))[3],candidates

def main():
 if sha(V1)!=V1_HASH:raise RuntimeError("Frozen estimator changed.")
 predictions=rows(PRED)
 if len(predictions)!=1192 or Counter(int(r["track_id"]) for r in predictions)!=Counter({8:399,10:400,14:393}):
  raise RuntimeError("Development support reconciliation failed.")
 support={(int(r["track_id"]),int(r["segment_frame_index"])):r for r in rows(SUPPORT)
          if int(r["track_id"]) in DEV and truth(r["inside_height_coverage"])}
 thermal={(int(r["track_id"]),int(r["segment_frame_index"])):r for r in rows(THERMAL_MAP)
          if int(r["track_id"]) in DEV}
 refs={r["sample_id"]:r for r in rows(REFS) if int(r["track_id"]) in DEV}
 if len(support)!=1192:raise RuntimeError("Expected 1192 Height-supported official positions.")

 labels=[]
 diagnostic_fields=("local_height_valid_fraction","aggregation_valid_fraction","central_region_valid_fraction",
  "left_substrate_valid_fraction","right_substrate_valid_fraction",
  "left_boundary_distance_to_nearest_nan_mm","right_boundary_distance_to_nearest_nan_mm",
  "aggregation_sensitivity_width_mm","aggregation_sensitivity_center_mm",
  "maximum_method_disagreement_width_mm","maximum_method_disagreement_center_mm",
  "width_neighbor_difference_mm","center_neighbor_difference_mm","apparent_cap_flag",
  "distance_to_candidate_limit_mm")
 for row in sorted(predictions,key=lambda r:(int(r["track_id"]),int(r["segment_frame_index"]))):
  track=int(row["track_id"]);seg=int(row["segment_frame_index"]);key=(track,seg)
  if key not in support or key not in thermal:raise RuntimeError(f"Missing exact mapping: {key}")
  mapping=thermal[key]
  if int(mapping["raw_frame_index"])!=int(row["raw_frame_index"]) or abs(float(mapping["x_mm_center"])-float(row["x_actual_mm"]))>1e-9:
   raise RuntimeError(f"Official mapping mismatch: {key}")
  finite_pred=(truth(row["finite_boundary_prediction"]) and finite(row["predicted_y_left_mm"]) and
   finite(row["predicted_y_right_mm"]) and finite(row["predicted_width_mm"]))
  ordered=finite_pred and fv(row,"predicted_y_right_mm")>fv(row,"predicted_y_left_mm")
  explicit=bool(row.get("estimator_exclusion_reason","").strip()) and row["estimator_exclusion_reason"] not in ("","none")
  eligible=finite_pred and ordered and not explicit
  reason="" if eligible else ("estimator_nonfinite" if not finite_pred else
    "invalid_boundary_order" if not ordered else "estimator_exclusion")
  ref=refs.get(row["sample_id"],{})
  human=truth(ref.get("primary_boundary_available","False"))
  out={"sample_id":row["sample_id"],"track_id":track,"segment_frame_index":seg,
   "raw_frame_index":int(row["raw_frame_index"]),"x_actual_mm":float(row["x_actual_mm"]),
   "official_position_index":seg,"height_window_available":truth(row["height_window_available"]),
   "height_estimator_id":"height_boundary_estimator_v1","height_estimator_config_hash":V1_HASH,
   "leveling_method_id":"official_robust_plane_v1","boundary_method_id":"gradient_edges",
   "aggregation_half_width_mm":.10,"target_protocol_version":"height_label_policy_v1",
   "target_y_left_mm":row["predicted_y_left_mm"],"target_y_right_mm":row["predicted_y_right_mm"],
   "target_width_mm":row["predicted_width_mm"],"target_center_mm":row["predicted_center_mm"],
   "finite_boundary_prediction":finite_pred,"estimator_exclusion_reason":row["estimator_exclusion_reason"],
   "primary_label_policy_id":"all_finite_height_estimator_v1",
   "primary_label_eligible":eligible,"primary_label_exclusion_reason":reason,
   "sensitivity_local_valid_ge_0p50":eligible and fv(row,"local_height_valid_fraction",0)>=.5,
   "sensitivity_side_support_ge_0p50":eligible and fv(row,"left_substrate_valid_fraction",0)>=.5 and fv(row,"right_substrate_valid_fraction",0)>=.5,
   "sensitivity_local_and_side_support_ge_0p50":eligible and fv(row,"local_height_valid_fraction",0)>=.5 and fv(row,"left_substrate_valid_fraction",0)>=.5 and fv(row,"right_substrate_valid_fraction",0)>=.5,
   "human_reference_available":human,"human_reference_source":ref.get("primary_reference_source",""),
   "human_measurability":ref.get("primary_human_measurability",""),
   "human_width_mm":ref.get("audit_reviewed_width_mm","") if human else "",
   "human_center_mm":ref.get("audit_reviewed_center_mm","") if human else ""}
  if human and finite_pred:
   out["frozen_v1_width_error_vs_human_mm"]=fv(row,"predicted_width_mm")-fv(ref,"audit_reviewed_width_mm")
   out["frozen_v1_center_error_vs_human_mm"]=fv(row,"predicted_center_mm")-fv(ref,"audit_reviewed_center_mm")
  else:out["frozen_v1_width_error_vs_human_mm"]=out["frozen_v1_center_error_vs_human_mm"]=""
  out.update({k:row.get(k,"") for k in diagnostic_fields});labels.append(out)
  out["aggregation_width_sensitivity_mm"]=row.get("aggregation_sensitivity_width_mm","")
  out["aggregation_center_sensitivity_mm"]=row.get("aggregation_sensitivity_center_mm","")
  out["maximum_method_width_disagreement_mm"]=row.get("maximum_method_disagreement_width_mm","")
  out["maximum_method_center_disagreement_mm"]=row.get("maximum_method_disagreement_center_mm","")
 write(HL/"height_labels_v1_development.csv",labels)
 primary=sum(truth(r["primary_label_eligible"]) for r in labels)
 finite_count=sum(truth(r["finite_boundary_prediction"]) for r in labels)
 if primary!=finite_count or primary!=1155:raise RuntimeError(f"Eligibility reconciliation failed: {primary}/{finite_count}")

 # Policy summary and descriptive figures.
 summary=[]
 for policy in POLICIES:
  for scope in ("aggregate",8,10,14):
   pool=[r for r in labels if scope=="aggregate" or int(r["track_id"])==scope]
   chosen=[r for r in pool if policy_flag(r,policy)];excluded=[r for r in pool if not policy_flag(r,policy)]
   widths=np.array([fv(r,"target_width_mm") for r in chosen]);xs=sorted(fv(r,"x_actual_mm") for r in chosen)
   reasons=Counter(r["primary_label_exclusion_reason"] or "sensitivity_support_threshold" for r in excluded)
   summary.append({"policy_id":policy,"track_scope":scope,"eligible_count":len(chosen),
    "excluded_count":len(excluded),"eligible_fraction":len(chosen)/len(pool),
    "percentage_official_positions_retained":100*len(chosen)/len(pool),
    "exclusion_reason_distribution":json.dumps(reasons,sort_keys=True),
    "target_width_mean_mm":np.mean(widths) if len(widths) else "",
    "target_width_std_mm":np.std(widths) if len(widths) else "",
    "target_width_median_mm":np.median(widths) if len(widths) else "",
    "target_width_minimum_mm":np.min(widths) if len(widths) else "",
    "target_width_maximum_mm":np.max(widths) if len(widths) else "",
    "x_minimum_mm":min(xs) if xs else "","x_maximum_mm":max(xs) if xs else "",
    "largest_x_gap_mm":max(np.diff(xs)) if len(xs)>1 else "",
    "major_track_imbalance_flag":False,"large_missing_x_interval_flag":bool(len(xs)>1 and max(np.diff(xs))>.6)})
 write(HL/"height_label_policy_v1_summary.csv",summary)
 FIGP.mkdir(parents=True,exist_ok=True)
 for track in DEV:
  pool=[r for r in labels if int(r["track_id"])==track]
  fig,axes=plt.subplots(2,2,figsize=(13,8))
  for policy,color in zip(POLICIES,("black","blue","orange","red")):
   ok=np.array([policy_flag(r,policy) for r in pool]);x=np.array([fv(r,"x_actual_mm") for r in pool])
   width=np.array([fv(r,"target_width_mm") for r in pool])
   axes[0,0].scatter(x[ok],width[ok],s=4,label=policy,color=color)
   axes[0,1].scatter(x[~ok],np.full((~ok).sum(),POLICIES.index(policy)),s=5,color=color)
   axes[1,0].hist(width[ok],bins=25,histtype="step",label=policy,color=color)
   axes[1,1].plot(x,ok.astype(int)+POLICIES.index(policy)*1.2,color=color)
  for ax in axes.ravel():ax.legend(fontsize=6);ax.grid(alpha=.2)
  axes[0,0].set_title("target width vs x");axes[0,1].set_title("excluded positions")
  axes[1,0].set_title("target-width distributions");axes[1,1].set_title("spatial support gaps")
  fig.tight_layout();fig.savefig(FIGP/f"Track_{track}_height_label_policy_v1.png",dpi=150);plt.close(fig)

 # Exact Thermal-to-Height manifest.
 manifest=[]
 for row in labels:
  key=(int(row["track_id"]),int(row["segment_frame_index"]));m=thermal[key]
  manifest.append({**row,"thermal_input_available":True,
   "thermal_source_path":m["thermal_file"],"thermal_tensor_reference":f"raw_frame_index={m['raw_frame_index']}",
   "thermal_shape":"400x400","thermal_representation_metadata":"single official raw thermal frame",
   "thermal_mapping_version":m["mapping_source"],"height_target_available":truth(row["finite_boundary_prediction"])})
 MN.mkdir(parents=True,exist_ok=True);write(MN/"thermal_height_manifest_v1_development.csv",manifest)
 if len({r["sample_id"] for r in manifest})!=1192 or len({(r["track_id"],r["x_actual_mm"]) for r in manifest})!=1192:
  raise RuntimeError("Duplicate manifest mapping.")
 splits=[]
 for held in DEV:
  fold=f"holdout_track_{held}"
  for r in manifest:splits.append({"outer_fold_id":fold,"sample_id":r["sample_id"],"track_id":r["track_id"],
    "role":"validation" if int(r["track_id"])==held else "training",
    "inner_validation_protocol":"two_way_complete_track_swap","random_position_split":False})
 write(MN/"thermal_height_loto_splits_v1.csv",splits)

 # Load canonical Thermal arrays for development Tracks only; compute sample-local representations.
 feature_by_key={}
 for track in DEV:
  thermal_data=extract_final_thermal_frames(PROJECT/"thermal",track)
  frames=thermal_data["raw_frames"]
  for row in [r for r in manifest if int(r["track_id"])==track]:
   frame=frames[int(row["raw_frame_index"])]
   if frame.shape!=(400,400):raise RuntimeError(f"Unexpected Thermal shape: {frame.shape}")
   feature_by_key[row["sample_id"]]=(summary_features(frame),profile_features(frame))
 del frames,thermal_data
 n=len(manifest);track=np.array([int(r["track_id"]) for r in manifest]);x=np.array([fv(r,"x_actual_mm") for r in manifest])
 y=np.array([fv(r,"target_width_mm") for r in manifest])
 summary_x=np.vstack([feature_by_key[r["sample_id"]][0] for r in manifest])
 profile_x=np.vstack([feature_by_key[r["sample_id"]][1] for r in manifest])

 model_grid=[];fold_metrics=[];predictions_out=[];sensitivity_metrics=[]
 for policy in POLICIES:
  eligible=np.array([policy_flag(r,policy) for r in manifest])
  for family in FAMILIES:
   for held in DEV:
    train_tracks=tuple(t for t in DEV if t!=held)
    param,candidates=choose(family,train_tracks,eligible,track,y,x,summary_x,profile_x)
    tr=np.flatnonzero(eligible & np.isin(track,train_tracks));va=np.flatnonzero(eligible & (track==held))
    pred,fit=fit_predict(family,param,tr,va,y,x,summary_x,profile_x)
    met=metrics(y[va],pred);met.update(policy_id=policy,model_family=family,
      outer_fold_id=f"holdout_track_{held}",training_tracks=";".join(map(str,train_tracks)),
      validation_track=held,training_sample_count=len(tr),validation_sample_count=len(va),
      selected_parameter_id=param["parameter_id"],preprocessing_fit_tracks=";".join(map(str,train_tracks)),
      inner_validation_tracks="complete_track_swap",random_position_split=False,
      human_reference_used_for_fit=False,height_feature_used_as_input=False,track_id_used_as_feature=False)
    if policy==POLICIES[0]:
     fold_metrics.append(met)
     for idx,prediction in zip(va,pred):
      r=manifest[idx];predictions_out.append({"policy_id":policy,"model_family":family,
       "outer_fold_id":f"holdout_track_{held}","sample_id":r["sample_id"],"track_id":r["track_id"],
       "x_actual_mm":r["x_actual_mm"],"target_width_mm":y[idx],"predicted_width_mm":prediction,
       "residual_mm":prediction-y[idx],"absolute_error_mm":abs(prediction-y[idx]),
       "local_height_valid_fraction":r["local_height_valid_fraction"],
       "left_substrate_valid_fraction":r["left_substrate_valid_fraction"],
       "right_substrate_valid_fraction":r["right_substrate_valid_fraction"],
       "human_width_mm":r["human_width_mm"],"human_reference_available":r["human_reference_available"]})
    else:sensitivity_metrics.append(met)
    for score,complexity,pid,candidate in candidates:
     model_grid.append({"policy_id":policy,"model_family":family,"outer_fold_id":f"holdout_track_{held}",
      "parameter_id":pid,"inner_complete_track_mean_mae_mm":score,"complexity_rank":complexity,
      "selected":pid==param["parameter_id"],"training_tracks":";".join(map(str,train_tracks)),
      "seed":SEED})
 write(MO/"thermal_height_baseline_model_grid.csv",model_grid)
 write(MO/"thermal_height_baseline_fold_metrics.csv",fold_metrics)
 write(MO/"thermal_height_baseline_predictions.csv",predictions_out)
 write(MO/"thermal_height_baseline_sensitivity_metrics.csv",sensitivity_metrics)

 aggregate=[]
 for family in FAMILIES:
  pp=[r for r in predictions_out if r["model_family"]==family]
  pooled=metrics([fv(r,"target_width_mm") for r in pp],[fv(r,"predicted_width_mm") for r in pp])
  fold=[r for r in fold_metrics if r["model_family"]==family]
  aggregate.append({"policy_id":POLICIES[0],"model_family":family,"aggregation":"pooled_rows",**pooled})
  aggregate.append({"policy_id":POLICIES[0],"model_family":family,"aggregation":"unweighted_track_mean",
   **{k:np.nanmean([fv(r,k) for r in fold]) for k in
   ("mae_mm","rmse_mm","median_absolute_error_mm","p90_absolute_error_mm","maximum_absolute_error_mm",
    "signed_bias_mm","r_squared","pearson_correlation","spearman_correlation")}})
 write(MO/"thermal_height_baseline_aggregate_metrics.csv",aggregate)

 human_diag=[]
 for r in predictions_out:
  if truth(r["human_reference_available"]) and finite(r["human_width_mm"]):
   human=fv(r,"human_width_mm");auto=fv(r,"target_width_mm");pred=fv(r,"predicted_width_mm")
   human_diag.append({**r,"automatic_target_error_vs_human_mm":auto-human,
    "model_error_vs_human_mm":pred-human,"model_error_vs_automatic_mm":pred-auto,
    "diagnostic_only":True,"used_for_fit_or_selection":False})
 write(MO/"thermal_height_baseline_human_reference_diagnostics.csv",human_diag)

 # Review figures.
 FIGM.mkdir(parents=True,exist_ok=True)
 for track_id in DEV:
  fig,ax=plt.subplots(figsize=(13,5))
  for family in FAMILIES:
   pp=sorted([r for r in predictions_out if int(r["track_id"])==track_id and r["model_family"]==family],
             key=lambda r:fv(r,"x_actual_mm"))
   ax.plot([fv(r,"x_actual_mm") for r in pp],[fv(r,"predicted_width_mm") for r in pp],label=family,lw=1)
  target=sorted([r for r in manifest if int(r["track_id"])==track_id and truth(r["primary_label_eligible"])],
                key=lambda r:fv(r,"x_actual_mm"))
  ax.scatter([fv(r,"x_actual_mm") for r in target],[fv(r,"target_width_mm") for r in target],
             s=4,c="black",label="frozen-v1 target")
  ax.legend(fontsize=7,ncol=3);ax.set(xlabel="x (mm)",ylabel="width (mm)",title=f"Track {track_id} LOTO predictions")
  fig.tight_layout();fig.savefig(FIGM/f"Track_{track_id}_thermal_height_baselines.png",dpi=150);plt.close(fig)

 agg={(r["model_family"],r["aggregation"]):r for r in aggregate}
 xmae=fv(agg[("x_only","unweighted_track_mean")],"mae_mm")
 thermal_choices=["thermal_summary_ridge","thermal_profile_ridge","thermal_summary_plus_x_ridge"]
 best=min(thermal_choices,key=lambda f:fv(agg[(f,"unweighted_track_mean")],"mae_mm"))
 bestmae=fv(agg[(best,"unweighted_track_mean")],"mae_mm")
 per_track_consistent=all(next(fv(r,"mae_mm") for r in fold_metrics if r["model_family"]==best and int(r["validation_track"])==t)
  <=next(fv(r,"mae_mm") for r in fold_metrics if r["model_family"]=="x_only" and int(r["validation_track"])==t) for t in DEV)
 status=("THERMAL-ONLY BASELINE PIPELINE READY FOR NEXT-STAGE MODELING"
         if bestmae<xmae and per_track_consistent else "THERMAL SIGNAL NOT ESTABLISHED BEYOND X-ONLY BASELINE")
 comparisons=[]
 mean_mae=fv(agg[("training_mean","unweighted_track_mean")],"mae_mm")
 summary_mae=fv(agg[("thermal_summary_ridge","unweighted_track_mean")],"mae_mm")
 profile_mae=fv(agg[("thermal_profile_ridge","unweighted_track_mean")],"mae_mm")
 plus_x_mae=fv(agg[("thermal_summary_plus_x_ridge","unweighted_track_mean")],"mae_mm")
 for label,a,b in (("x_only minus training_mean",xmae,mean_mae),
  ("thermal_summary minus x_only",summary_mae,xmae),
  ("thermal_profile minus x_only",profile_mae,xmae),
  ("summary_plus_x minus summary",plus_x_mae,summary_mae),
  ("best_thermal minus training_mean",bestmae,mean_mae),
  ("best_thermal minus x_only",bestmae,xmae)):
  comparisons.append(f"| {label} | {a-b:.6f} | {(a-b)/b:.3%} |")
 (RP/"height_label_policy_v1_freeze_decision.md").write_text(
  "# Height label policy v1 freeze decision\n\nEstimator v1 is retained because v2 through dense-v2r4 "
  "did not provide a stable replacement. All 1,155 finite frozen-v1 predictions are the primary target; "
  "the prior objective quality gates did not reliably identify geometry errors and caused unstable "
  "Track-dependent selection. No fitted quality gate is used. Human references remain separated audit "
  "metadata and never substitute for the automatic target. Three fixed support subsets are retained only "
  "for descriptive robustness checks. Track 21 is reserved for final testing and is excluded from this "
  "development policy and all outputs.\n",encoding="utf-8")
 (RP/"thermal_height_baseline_leakage_audit.md").write_text(
  "# Thermal–Height baseline leakage audit\n\n"
  f"- Development manifest rows: {len(manifest)}; primary eligible: {primary}; folds: 3.\n"
  "- Only development Tracks 8, 10, and 14 were loaded. The reserved final Track was not loaded or emitted.\n"
  "- No SEM data or paths were loaded. No Height map, profile, validity mask, diagnostic, or human field "
  "entered a model feature matrix; Height supplied target_width_mm only.\n"
  "- Track ID was used only to define complete-Track folds, never as a feature. No random x split was used.\n"
  "- Scaling, coefficients, and all hyperparameter choices were fit using outer-training Tracks only. "
  "Inner validation used the two complete training Tracks in both directions.\n"
  "- Human references were post-hoc diagnostics only. Every primary-eligible row has exactly one outer "
  "validation prediction per family.\n",encoding="utf-8")
 (RP/"thermal_height_baseline_review.md").write_text(
  "# Initial Thermal-only Height-width baseline review\n\n"
  f"Primary frozen-v1 eligible samples: {primary}/1192. The best Thermal family by unweighted Track MAE "
  f"is `{best}` at {bestmae:.6f} mm; x-only is {xmae:.6f} mm. Per-Track improvement over x-only: "
  f"{per_track_consistent}. Thermal is considered additive only when improvement is directionally "
  "consistent across all three complete-Track holdouts.\n\n"
  "## Unweighted Track-MAE comparisons\n\n| Comparison | Absolute MAE difference (mm) | Relative change |\n"
  "|---|---:|---:|\n"+"\n".join(comparisons)+
  "\n\nNo single pooled result overrides per-Track consistency; the fold table identifies whether one Track "
  "drives a difference.\n\nSensitivity policies remain descriptive and "
  "do not replace the primary policy. Human-reference comparisons are post-hoc and were not used for "
  "fitting, selection, or ranking.\n\n"+status+"\n",encoding="utf-8")
 (RP/"thermal_height_baseline_build_report.md").write_text(
  "# Thermal–Height baseline build report\n\n"
  f"- Labels/manifest: {len(labels)} rows; finite/primary eligible: {primary}; Thermal features: "
  f"{summary_x.shape[1]} summary and {profile_x.shape[1]} block-average values.\n"
  f"- Primary predictions: {len(predictions_out)}; fold metrics: {len(fold_metrics)}; sensitivity metrics: "
  f"{len(sensitivity_metrics)}; human diagnostics: {len(human_diag)}.\n"
  "- Five deterministic model families, three complete-Track outer folds, two-way complete-Track inner "
  "validation, seed 20260725. No final-test predictions, SEM processing, or multimodal modeling.\n",encoding="utf-8")
 print(f"labels={len(labels)} eligible={primary} predictions={len(predictions_out)} best={best} status={status}")

if __name__=="__main__":main()
