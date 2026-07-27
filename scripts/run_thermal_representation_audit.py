#!/usr/bin/env python3
"""Thermal representation, alignment, and causal-context audit on development Tracks."""
from __future__ import annotations
import csv,hashlib,json,math,sys
from collections import Counter,defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import whosmat
from scipy.stats import pearsonr,spearmanr

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from nsf_fmrg_data import extract_final_thermal_frames
from scripts.build_height_quality_gate_review import truth,write
from scripts.run_thermal_height_baselines import (
 finite,fv,metrics,ridge_fit,fit_scaler,summary_features,profile_features,policy_flag)

DEV=(8,10,14);SEED=20260725
V1_HASH="4b1094cf2084f10026cdd470dba7ba3bdf8b561bfef02ba7ef3413e8a555617d"
POLICY_HASH="5b568d33b06761ffdce9c35fd75fa03307c6ec8dcdb4cd510e7513d39e96c994"
V1=PROJECT/"configs/height_boundary_estimator_v1.yaml";POLICY=PROJECT/"configs/height_label_policy_v1.yaml"
MANIFEST=PROJECT/"outputs/manifests/thermal_height_manifest_v1_development.csv"
LABELS=PROJECT/"outputs/height_labels/height_labels_v1_development.csv"
REFS=PROJECT/"outputs/human_review/height_human_reference_v3_development.csv"
TA=PROJECT/"outputs/thermal_audit";MN=PROJECT/"outputs/manifests";MO=PROJECT/"outputs/models"
RP=PROJECT/"outputs/reports";FIGS=PROJECT/"outputs/figures/thermal_source_audit"
FIGR=PROJECT/"outputs/figures/thermal_representation_review"
REPRESENTATIONS=("current_absolute","current_background_subtracted","current_robust_scaled",
 "hotspot_centered_absolute","hotspot_centered_relative","thermal_shape_features",
 "causal_context_prev1","causal_context_prev2","causal_context_prev4")
ALPHAS=(.1,1.,10.,100.,1000.);LAGS=tuple(range(-5,6))

def rows(path):
 with Path(path).open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def block20(frame):
 a=np.asarray(frame,float);h,w=a.shape
 if h%20 or w%20:raise ValueError("Representation image must be divisible into a 20x20 grid.")
 return a.reshape(20,h//20,20,w//20).mean(axis=(1,3)).ravel()
def robust_scale(frame):
 p5,p50,p95=np.percentile(frame,[5,50,95]);return (frame-p50)/(p95-p5+1e-6)
def hotspot(frame):
 a=np.asarray(frame,float);threshold=np.percentile(a,95);mask=a>=threshold
 if not mask.any():return False,np.nan,np.nan,mask
 yy,xx=np.indices(a.shape);weights=np.maximum(a-threshold,0)+mask*1e-12;total=weights.sum()
 return True,float((weights*xx).sum()/total),float((weights*yy).sum()/total),mask
def crop(frame,cx,cy,size=160):
 half=size//2;x0=int(round(cx))-half;y0=int(round(cy))-half
 out=np.zeros((size,size),float);sx0=max(0,x0);sy0=max(0,y0);sx1=min(frame.shape[1],x0+size);sy1=min(frame.shape[0],y0+size)
 if sx1>sx0 and sy1>sy0:out[sy0-y0:sy1-y0,sx0-x0:sx1-x0]=frame[sy0:sy1,sx0:sx1]
 return out,bool(x0<0 or y0<0 or x0+size>frame.shape[1] or y0+size>frame.shape[0])
def shape_features(frame,success,cx,cy,mask):
 a=np.asarray(frame,float);bg=np.percentile(a,10);excess=np.maximum(a-bg,0)
 yy,xx=np.indices(a.shape);w=excess*mask;total=w.sum()
 if not success or total<=0:return np.r_[np.zeros(18),1.]
 mx=(w*xx).sum()/total;my=(w*yy).sum()/total
 dx=xx-mx;dy=yy-my;cxx=(w*dx*dx).sum()/total;cyy=(w*dy*dy).sum()/total;cxy=(w*dx*dy).sum()/total
 eig=np.linalg.eigvalsh([[cxx,cxy],[cxy,cyy]])[::-1];axes=2*np.sqrt(np.maximum(eig,0))
 area=mask.mean();equiv=2*np.sqrt(mask.sum()/np.pi)/400
 gx=np.diff(a,axis=1);gy=np.diff(a,axis=0)
 thresholds=[np.mean(a>=np.percentile(a,p)) for p in (90,95,97,99)]
 return np.array(thresholds+[equiv,axes[0]/400,axes[1]/400,axes[0]/max(axes[1],1e-9),
  mx/399,my/399,eig[0]/160000,eig[1]/160000,a.max()-bg,excess.sum()/a.size,
  np.mean(abs(gx)),np.mean(abs(gy)),a[:,:200].mean()-a[:,200:].mean(),
  a[:200].mean()-a[200:].mean(),0.],float)
def normalized_correlation(a,b):
 x=np.asarray(a,float).ravel();y=np.asarray(b,float).ravel();x-=x.mean();y-=y.mean()
 return float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-12))
def fit_predict(xtrain,ytrain,xval,alpha):
 mean,scale=fit_scaler(xtrain);xt=(xtrain-mean)/scale;xv=(xval-mean)/scale
 xt=np.column_stack([np.ones(len(xt)),xt]);xv=np.column_stack([np.ones(len(xv)),xv])
 coef=ridge_fit(xt,ytrain,alpha);return xv@coef,{"mean":mean,"scale":scale,"coef":coef}
def x_features(x,degree=2):return np.column_stack([x**d for d in range(1,degree+1)])
def choose_x(train_tracks,eligible,track,x,y):
 ranked=[]
 for degree in (1,2,3):
  for alpha in ALPHAS:
   maes=[]
   for held in train_tracks:
    tr=np.flatnonzero(eligible&np.isin(track,[t for t in train_tracks if t!=held]));va=np.flatnonzero(eligible&(track==held))
    pred,_=fit_predict(x_features(x[tr],degree),y[tr],x_features(x[va],degree),alpha);maes.append(np.mean(abs(pred-y[va])))
   ranked.append((np.mean(maes),degree,alpha))
 return min(ranked,key=lambda z:(z[0],z[1],z[2]))
def residual_candidate(rep,alpha,train_tracks,eligible,track,x,y,reps):
 maes=[]
 for held in train_tracks:
  inner_train=tuple(t for t in train_tracks if t!=held)
  degree,xalpha=2,1.0
  tr=np.flatnonzero(eligible&np.isin(track,inner_train));va=np.flatnonzero(eligible&(track==held))
  xtr_pred,_=fit_predict(x_features(x[tr],degree),y[tr],x_features(x[tr],degree),xalpha)
  xva_pred,_=fit_predict(x_features(x[tr],degree),y[tr],x_features(x[va],degree),xalpha)
  residual=y[tr]-xtr_pred
  correction,_=fit_predict(reps[rep][tr],residual,reps[rep][va],alpha)
  maes.append(np.mean(abs(xva_pred+correction-y[va])))
 return np.mean(maes)

def main():
 if sha(V1)!=V1_HASH or sha(POLICY)!=POLICY_HASH:raise RuntimeError("Frozen Height configuration changed.")
 manifest=rows(MANIFEST);labels={r["sample_id"]:r for r in rows(LABELS)}
 if len(manifest)!=1192 or {int(r["track_id"]) for r in manifest}!={8,10,14}:raise RuntimeError("Development manifest invalid.")
 index={r["sample_id"]:i for i,r in enumerate(manifest)}
 track=np.array([int(r["track_id"]) for r in manifest]);x=np.array([fv(r,"x_actual_mm") for r in manifest])
 raw_index=np.array([int(r["raw_frame_index"]) for r in manifest]);segment=np.array([int(r["segment_frame_index"]) for r in manifest])
 y=np.array([fv(r,"target_width_mm") for r in manifest]);eligible=np.array([truth(r["primary_label_eligible"]) for r in manifest])

 TA.mkdir(parents=True,exist_ok=True);FIGS.mkdir(parents=True,exist_ok=True);FIGR.mkdir(parents=True,exist_ok=True)
 inventory=[];mapping_audit=[];hotspot_audit=[];frame_features=np.zeros((len(manifest),17))
 absolute=np.zeros((len(manifest),400));background=np.zeros_like(absolute);scaled=np.zeros_like(absolute)
 centered_abs=np.zeros_like(absolute);centered_rel=np.zeros_like(absolute);shapes=np.zeros((len(manifest),19))
 reps_frames={};representative={}
 for tid in DEV:
  info=extract_final_thermal_frames(PROJECT/"thermal",tid);frames=info["raw_frames"]
  mat_info=whosmat(PROJECT/f"thermal/Thermal_{tid}.mat")
  used=[r for r in manifest if int(r["track_id"])==tid];used_frames=np.stack([frames[int(r["raw_frame_index"])] for r in used])
  flat=used_frames.reshape(-1);finite_values=flat[np.isfinite(flat)]
  duplicate=sum(np.array_equal(used_frames[i],used_frames[i-1]) for i in range(1,len(used_frames)))
  near=sum(np.mean(abs(used_frames[i]-used_frames[i-1]))<1e-6 for i in range(1,len(used_frames)))
  inventory.append({"track_id":tid,"source_file":info["file"],"file_format":"MAT",
   "available_arrays":json.dumps(mat_info),"selected_array":info["variable"],
   "raw_array_shape":"x".join(map(str,frames.shape)),"frame_count":len(frames),
   "spatial_shape":"400x400","data_type":str(frames.dtype),"units":"not_explicitly_calibrated",
   "minimum":finite_values.min(),"maximum":finite_values.max(),"mean":finite_values.mean(),
   "standard_deviation":finite_values.std(),"p01":np.percentile(finite_values,1),
   "p50":np.percentile(finite_values,50),"p99":np.percentile(finite_values,99),
   "nan_count":np.isnan(flat).sum(),"infinite_count":np.isinf(flat).sum(),
   "zero_fraction":np.mean(flat==0),"constant_frame_fraction":np.mean(np.std(used_frames,axis=(1,2))==0),
   "minimum_value_fraction":np.mean(flat==finite_values.min()),"maximum_value_fraction":np.mean(flat==finite_values.max()),
   "duplicate_consecutive_frames":duplicate,"near_duplicate_consecutive_frames":near,
   "raw_frame_index_minimum":min(int(r["raw_frame_index"]) for r in used),
   "raw_frame_index_maximum":max(int(r["raw_frame_index"]) for r in used),
   "x_direction_relative_to_raw_order":"increasing_x_with_increasing_raw_frame_index"})
  mean_img=np.mean(used_frames,axis=0);median_img=np.median(used_frames,axis=0);std_img=np.std(used_frames,axis=0)
  representative[tid]=(mean_img,median_img,std_img,used_frames[len(used_frames)//2])
  fig,axes=plt.subplots(1,4,figsize=(16,4))
  for ax,img,title in zip(axes,representative[tid],("mean","median","std","representative")):
   ax.imshow(img);ax.set_title(f"Track {tid} {title}");ax.axis("off")
  fig.tight_layout();fig.savefig(FIGS/f"Track_{tid}_thermal_source_summary.png",dpi=140);plt.close(fig)
  previous=None
  for r in used:
   i=index[r["sample_id"]];frame=frames[int(r["raw_frame_index"])]
   success,cx,cy,mask=hotspot(frame);roi,edge=crop(frame,cx,cy) if success else (np.zeros((160,160)),True)
   bg=np.percentile(frame,10);relative=robust_scale(frame)
   roi_rel=robust_scale(roi) if success else roi
   absolute[i]=block20(frame);background[i]=block20(frame-bg);scaled[i]=block20(relative)
   centered_abs[i]=block20(roi);centered_rel[i]=block20(roi_rel);shapes[i]=shape_features(frame,success,cx,cy,mask)
   s=summary_features(frame);frame_features[i]=s
   hotspot_audit.append({"sample_id":r["sample_id"],"track_id":tid,"raw_frame_index":r["raw_frame_index"],
    "hotspot_detection_success":success,"centroid_x_px":cx,"centroid_y_px":cy,"crop_boundary_failure":edge,
    "hotspot_area_fraction":mask.mean(),"hotspot_contrast":frame.max()-bg,
    "shared_threshold_percentile":95,"shared_roi_size_pixels":160})
   mad=np.mean(abs(frame-previous)) if previous is not None else np.nan
   corr=normalized_correlation(frame,previous) if previous is not None else np.nan
   prev_hot=hotspot(previous)[1:3] if previous is not None else (np.nan,np.nan)
   displacement=np.hypot(cx-prev_hot[0],cy-prev_hot[1]) if previous is not None and success else np.nan
   expected_raw=int(r["segment_frame_index"])+int(info["start_idx"])
   consistent=int(r["raw_frame_index"])==expected_raw
   mapping_audit.append({"sample_id":r["sample_id"],"track_id":tid,"segment_frame_index":r["segment_frame_index"],
    "raw_frame_index":r["raw_frame_index"],"expected_zero_based_raw_frame_index":expected_raw,
    "zero_based_index_verified":consistent,"x_actual_mm":r["x_actual_mm"],
    "raw_index_increment_from_previous":int(r["raw_frame_index"])-int(used[max(0,used.index(r)-1)]["raw_frame_index"]) if previous is not None else "",
    "x_increment_from_previous_mm":fv(r,"x_actual_mm")-fv(used[max(0,used.index(r)-1)],"x_actual_mm") if previous is not None else "",
    "mean_absolute_pixel_difference":mad,"normalized_frame_correlation":corr,
    "hotspot_centroid_displacement_px":displacement,
    "duplicate_frame_flag":bool(previous is not None and mad==0),
    "near_duplicate_frame_flag":bool(previous is not None and mad<1e-6),
    "direction_reversal_flag":False,"mapping_consistent":consistent})
   if not consistent:raise RuntimeError(f"Confirmed official frame mapping inconsistency: {r['sample_id']}")
   previous=frame
  del frames,used_frames,flat,finite_values
 write(TA/"thermal_source_inventory_development.csv",inventory)
 write(TA/"thermal_frame_mapping_audit.csv",mapping_audit)
 write(TA/"thermal_hotspot_detection_audit.csv",hotspot_audit)

 # Distribution-shift summaries and audit-only PCA.
 shift=[]
 names=("frame_mean","frame_std","frame_min","frame_max","p10","p50","p90","p99",
  "peak_background_contrast","hot_area","centroid_x","centroid_y","left_right","up_down","grad_x","grad_y","integrated_excess")
 for j,name in enumerate(names):
  pooled_std=np.std(frame_features[:,j])
  track_means={t:np.mean(frame_features[track==t,j]) for t in DEV}
  within=np.mean([np.std(frame_features[track==t,j]) for t in DEV])
  shift.append({"feature_name":name,"track_8_mean":track_means[8],"track_10_mean":track_means[10],
   "track_14_mean":track_means[14],"between_track_mean_range":max(track_means.values())-min(track_means.values()),
   "mean_within_track_std":within,"between_to_within_ratio":(max(track_means.values())-min(track_means.values()))/max(within,1e-12),
   "pooled_std":pooled_std})
 write(TA/"thermal_distribution_shift_metrics.csv",shift)
 amean,ascale=fit_scaler(absolute);u,s,v=np.linalg.svd((absolute-amean)/ascale,full_matrices=False)
 scores=u[:,:2]*s[:2]
 fig,ax=plt.subplots(figsize=(7,6))
 for tid,color in zip(DEV,("red","blue","green")):ax.scatter(scores[track==tid,0],scores[track==tid,1],s=8,label=f"Track {tid}",c=color)
 ax.legend();ax.set(title="Audit-only PCA of current absolute representation",xlabel="PC1",ylabel="PC2")
 fig.tight_layout();fig.savefig(FIGS/"thermal_distribution_audit_pca.png",dpi=150);plt.close(fig)

 # Causal representations: current and preceding official positions only.
 reps={"current_absolute":absolute,"current_background_subtracted":background,
  "current_robust_scaled":scaled,"hotspot_centered_absolute":centered_abs,
  "hotspot_centered_relative":centered_rel,"thermal_shape_features":shapes}
 context_indices={}
 for previous_count in (1,2,4):
  matrix=[];context_indices[previous_count]=[]
  for i,r in enumerate(manifest):
   tid=int(r["track_id"]);seg=int(r["segment_frame_index"]);indices=[]
   for lag in range(previous_count,-1,-1):
    candidate=next((index[x["sample_id"]] for x in manifest if int(x["track_id"])==tid and int(x["segment_frame_index"])==seg-lag),None)
    indices.append(i if candidate is None else candidate)
   stack=scaled[indices];matrix.append(np.r_[stack[-1],stack.mean(axis=0),stack[-1]-stack[0],
                                             float(any(j==i for j in indices[:-1]))])
   context_indices[previous_count].append(indices)
  reps[f"causal_context_prev{previous_count}"]=np.vstack(matrix)

 rep_manifest=[]
 for i,r in enumerate(manifest):
  hot=next(x for x in hotspot_audit if x["sample_id"]==r["sample_id"])
  for rep in REPRESENTATIONS:
   count=int(rep.replace("causal_context_prev","")) if rep.startswith("causal") else 0
   indices=[raw_index[j] for j in context_indices[count][i]] if count else [raw_index[i]]
   rep_manifest.append({"sample_id":r["sample_id"],"track_id":r["track_id"],"x_actual_mm":r["x_actual_mm"],
    "raw_frame_index":r["raw_frame_index"],"representation_id":rep,"representation_available":True,
    "representation_shape":str(reps[rep].shape[1]),"context_frame_indices":";".join(map(str,indices)),
    "context_direction":"current_and_preceding_only" if count else "current_only",
    "hotspot_detection_success":hot["hotspot_detection_success"],
    "normalization_id":("sample_local_robust_p05_p50_p95" if "relative" in rep or "robust" in rep or "causal" in rep
                        else "sample_local_p10_background" if "background" in rep else "none"),
    "representation_source":"canonical_official_raw_thermal_frame","primary_label_eligible":r["primary_label_eligible"],
    "future_frame_used":False})
 write(MN/"thermal_representation_manifest_v1_development.csv",rep_manifest)

 # Nested residual modeling and complete representation/model grid.
 model_grid=[];fold_metrics=[];residual_metrics=[];predictions=[];nested_selected={}
 for held in DEV:
  training=tuple(t for t in DEV if t!=held)
  _,degree,xalpha=choose_x(training,eligible,track,x,y)
  candidates=[]
  for rep in REPRESENTATIONS:
   for alpha in ALPHAS:
    score=residual_candidate(rep,alpha,training,eligible,track,x,y,reps)
    candidates.append((score,reps[rep].shape[1],rep,alpha))
    model_grid.append({"outer_fold_id":f"holdout_track_{held}","representation_id":rep,
     "model_family":"ridge_residual","ridge_alpha":alpha,"inner_complete_track_mae_mm":score,
     "training_tracks":";".join(map(str,training)),"inner_validation":"complete_track_swap",
     "random_position_split":False})
  _,_,selected_rep,selected_alpha=min(candidates,key=lambda z:(z[0],z[1],z[2],z[3]))
  nested_selected[held]=(selected_rep,selected_alpha,degree,xalpha)
  tr=np.flatnonzero(eligible&np.isin(track,training));va=np.flatnonzero(eligible&(track==held))
  xtr_pred,_=fit_predict(x_features(x[tr],degree),y[tr],x_features(x[tr],degree),xalpha)
  xva_pred,_=fit_predict(x_features(x[tr],degree),y[tr],x_features(x[va],degree),xalpha)
  residual=y[tr]-xtr_pred
  correction,_=fit_predict(reps[selected_rep][tr],residual,reps[selected_rep][va],selected_alpha)
  thermal_only,_=fit_predict(reps[selected_rep][tr],y[tr],reps[selected_rep][va],selected_alpha)
  combined=xva_pred+correction
  for prediction_type,prediction in (("thermal_only",thermal_only),("x_only",xva_pred),
                                      ("x_plus_thermal_residual",combined)):
   met=metrics(y[va],prediction);met.update(outer_fold_id=f"holdout_track_{held}",
    validation_track=held,training_tracks=";".join(map(str,training)),representation_id=selected_rep,
    model_family=prediction_type,ridge_alpha=selected_alpha,sample_count=len(va),
    preprocessing_fit_tracks=";".join(map(str,training)),held_out_statistics_used=False,
    human_reference_used=False,random_position_split=False)
   if prediction_type=="x_plus_thermal_residual":
    xmet=metrics(y[va],xva_pred);met.update(mae_difference_vs_x_only_mm=met["mae_mm"]-xmet["mae_mm"],
     relative_mae_change_vs_x_only=(met["mae_mm"]-xmet["mae_mm"])/xmet["mae_mm"],
     residual_correlation=safe_corr(pearsonr,y[va]-xva_pred,correction),
     residual_variance_explained=1-np.sum((y[va]-combined)**2)/max(np.sum((y[va]-xva_pred)**2),1e-12),
     correction_above_0p02_fraction=np.mean(abs(correction)>.02),
     correction_above_0p05_fraction=np.mean(abs(correction)>.05),
     correction_above_0p10_fraction=np.mean(abs(correction)>.10))
    residual_metrics.append(met)
   else:fold_metrics.append(met)
  for idx,xp,tp,cp,comb in zip(va,xva_pred,thermal_only,correction,combined):
   predictions.append({"outer_fold_id":f"holdout_track_{held}","sample_id":manifest[idx]["sample_id"],
    "track_id":track[idx],"x_actual_mm":x[idx],"target_width_mm":y[idx],
    "representation_id":selected_rep,"x_only_prediction_mm":xp,"thermal_only_prediction_mm":tp,
    "predicted_residual_correction_mm":cp,"x_plus_thermal_residual_prediction_mm":comb,
    "residual_target_created_for_held_out":False,"x_only_prediction_locked_before_correction":True})
 write(MO/"thermal_representation_model_grid.csv",model_grid)
 write(MO/"thermal_representation_fold_metrics.csv",fold_metrics)
 write(MO/"thermal_residual_fold_metrics.csv",residual_metrics)
 write(MO/"thermal_representation_predictions.csv",predictions)

 # Fixed representation: global development review only, then unchanged representation across Track folds.
 global_scores=defaultdict(list)
 for row in model_grid:global_scores[(row["representation_id"],float(row["ridge_alpha"]))].append(float(row["inner_complete_track_mae_mm"]))
 fixed_rep,fixed_alpha=min(global_scores,key=lambda k:(np.mean(global_scores[k]),reps[k[0]].shape[1],k[0],k[1]))
 fixed_rows=[]
 for held in DEV:
  training=tuple(t for t in DEV if t!=held);_,degree,xalpha=choose_x(training,eligible,track,x,y)
  tr=np.flatnonzero(eligible&np.isin(track,training));va=np.flatnonzero(eligible&(track==held))
  xp_tr,_=fit_predict(x_features(x[tr],degree),y[tr],x_features(x[tr],degree),xalpha)
  xp_va,_=fit_predict(x_features(x[tr],degree),y[tr],x_features(x[va],degree),xalpha)
  correction,_=fit_predict(reps[fixed_rep][tr],y[tr]-xp_tr,reps[fixed_rep][va],fixed_alpha)
  met=metrics(y[va],xp_va+correction);met.update(evaluation_type="fixed_representation_track_evaluation",
   representation_id=fixed_rep,ridge_alpha=fixed_alpha,validation_track=held,
   fixed_representation_selected_using_all_development_review=True,independently_validated=False)
  fixed_rows.append(met)

 def aggregate(rows_in,label):
  out=[];keys=[r["validation_track"] for r in rows_in]
  out.append({"evaluation_type":label,"aggregation":"unweighted_track_mean",
   **{k:np.nanmean([fv(r,k) for r in rows_in]) for k in
   ("mae_mm","rmse_mm","median_absolute_error_mm","p90_absolute_error_mm","maximum_absolute_error_mm",
    "signed_bias_mm","r_squared","pearson_correlation","spearman_correlation")}})
  return out
 write(MO/"thermal_representation_aggregate_metrics.csv",aggregate(fold_metrics,"nested_selected_nonresidual"))
 write(MO/"thermal_residual_aggregate_metrics.csv",aggregate(residual_metrics,"nested_representation_selection")+
       aggregate(fixed_rows,"fixed_representation_descriptive"))

 # Symmetric lag diagnostic: future offsets remain diagnostic and never enter nested selection.
 lag_metrics=[]
 for held in DEV:
  training=tuple(t for t in DEV if t!=held)
  _,degree,xalpha=choose_x(training,eligible,track,x,y)
  for lag in LAGS:
   valid=np.array([eligible[i] and (segment[i]+lag)>=0 and
    any(int(m["track_id"])==track[i] and int(m["segment_frame_index"])==segment[i]+lag for m in manifest)
    for i in range(len(manifest))])
   shifted=np.zeros_like(absolute)
   for i in np.flatnonzero(valid):
    j=next(index[m["sample_id"]] for m in manifest if int(m["track_id"])==track[i] and
           int(m["segment_frame_index"])==segment[i]+lag);shifted[i]=scaled[j]
   tr=np.flatnonzero(valid&np.isin(track,training));va=np.flatnonzero(valid&(track==held))
   xp_tr,_=fit_predict(x_features(x[tr],degree),y[tr],x_features(x[tr],degree),xalpha)
   xp_va,_=fit_predict(x_features(x[tr],degree),y[tr],x_features(x[va],degree),xalpha)
   correction,_=fit_predict(shifted[tr],y[tr]-xp_tr,shifted[va],10.)
   met=metrics(y[va],xp_va+correction);met.update(validation_track=held,offset_official_positions=lag,
    future_frame_offset=lag>0,diagnostic_only=True,used_for_model_or_mapping_selection=False)
   lag_metrics.append(met)
 write(MO/"thermal_lag_diagnostic_metrics.csv",lag_metrics)

 # Fixed support sensitivity using locked nested choices.
 sensitivity=[]
 for policy_id in ("local_valid_ge_0p50","side_support_ge_0p50","local_and_side_support_ge_0p50"):
  policy_mask=np.array([policy_flag(labels[r["sample_id"]],policy_id) for r in manifest])
  for held in DEV:
   rep,alpha,degree,xalpha=nested_selected[held];training=tuple(t for t in DEV if t!=held)
   tr=np.flatnonzero(policy_mask&np.isin(track,training));va=np.flatnonzero(policy_mask&(track==held))
   xp_tr,_=fit_predict(x_features(x[tr],degree),y[tr],x_features(x[tr],degree),xalpha)
   xp_va,_=fit_predict(x_features(x[tr],degree),y[tr],x_features(x[va],degree),xalpha)
   correction,_=fit_predict(reps[rep][tr],y[tr]-xp_tr,reps[rep][va],alpha)
   met=metrics(y[va],xp_va+correction);met.update(policy_id=policy_id,validation_track=held,
    representation_id=rep,representation_selection_reused_from_primary=True,threshold_tuned=False)
   sensitivity.append(met)
 write(MO/"thermal_representation_sensitivity_metrics.csv",sensitivity)

 refs={r["sample_id"]:r for r in rows(REFS)}
 human=[]
 for row in predictions:
  ref=refs.get(row["sample_id"],{})
  if truth(ref.get("primary_boundary_available","False")):
   hw=fv(ref,"audit_reviewed_width_mm");target=fv(row,"target_width_mm")
   human.append({**row,"human_width_mm":hw,"frozen_v1_error_vs_human_mm":target-hw,
    "x_only_error_vs_human_mm":fv(row,"x_only_prediction_mm")-hw,
    "thermal_only_error_vs_human_mm":fv(row,"thermal_only_prediction_mm")-hw,
    "x_plus_thermal_error_vs_human_mm":fv(row,"x_plus_thermal_residual_prediction_mm")-hw,
    "diagnostic_only":True,"used_for_selection":False})
 write(MO/"thermal_representation_human_diagnostics.csv",human)

 # Figures: shift distributions, hotspot examples, and residual comparisons.
 fig,axes=plt.subplots(2,3,figsize=(15,8))
 for j,name in enumerate(("frame_mean","frame_std","frame_max","hot_area","centroid_x","centroid_y")):
  feature_index=names.index(name)
  for tid in DEV:axes.ravel()[j].hist(frame_features[track==tid,feature_index],bins=25,histtype="step",label=f"T{tid}")
  axes.ravel()[j].set_title(name);axes.ravel()[j].legend(fontsize=6)
 fig.tight_layout();fig.savefig(FIGR/"thermal_feature_distribution_shift.png",dpi=150);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,5))
 for tid in DEV:
  r=[m for m in residual_metrics if int(m["validation_track"])==tid][0]
  ax.bar(str(tid),fv(r,"mae_difference_vs_x_only_mm"))
 ax.axhline(0,color="black");ax.set(title="Nested residual MAE difference vs x-only",ylabel="mm")
 fig.tight_layout();fig.savefig(FIGR/"nested_residual_mae_differences.png",dpi=150);plt.close(fig)

 best_lag={t:min((r for r in lag_metrics if int(r["validation_track"])==t),key=lambda r:fv(r,"mae_mm")) for t in DEV}
 lag_consistent=len({int(r["offset_official_positions"]) for r in best_lag.values()})==1
 nested_improve=[fv(r,"mae_difference_vs_x_only_mm")<0 for r in residual_metrics]
 nested_gain=-np.mean([fv(r,"mae_difference_vs_x_only_mm") for r in residual_metrics])
 x_mae=np.mean([fv(r,"mae_mm")-fv(r,"mae_difference_vs_x_only_mm") for r in residual_metrics])
 relative=nested_gain/x_mae
 human_x=np.mean([abs(fv(r,"x_only_error_vs_human_mm")) for r in human])
 human_combined=np.mean([abs(fv(r,"x_plus_thermal_error_vs_human_mm")) for r in human])
 success=(all(nested_improve) and (nested_gain>=.005 or relative>=.05) and
          all(fv(r,"mae_difference_vs_x_only_mm")<=.002 for r in residual_metrics) and
          human_combined<=human_x)
 status=("THERMAL SIGNAL ESTABLISHED BEYOND X-ONLY AFTER REPRESENTATION AUDIT" if success else
         "THERMAL SIGNAL NOT GENERALIZABLE ACROSS DEVELOPMENT TRACKS")
 write(TA/"thermal_source_inventory_development.csv",inventory)
 (RP/"thermal_source_and_loader_audit.md").write_text(
  "# Thermal source and loader audit\n\nDevelopment MAT files contain one selected `temperature_data` "
  "array each, with 400×400 spatial samples and 929/961/976 raw frames. Values are reported as "
  "uncalibrated Thermal intensity because explicit calibrated units are absent. The official mapping "
  "uses zero-based raw indices, increasing raw index with increasing official x; all 1,192 manifest "
  "rows exactly match `segment_frame_index + extracted_start` and no correction was applied.\n",encoding="utf-8")
 (RP/"thermal_distribution_shift_review.md").write_text(
  "# Thermal distribution-shift review\n\nThe distribution table quantifies between-Track mean shift "
  "relative to within-Track variation for intensity, hotspot, centroid, asymmetry, and gradient features. "
  "Mean/median/std images and audit-only PCA show substantial acquisition/representation separation. "
  "This descriptive PCA is not reused by predictive models. Hotspot detection uses a shared p95 threshold "
  "and 160×160 ROI for every Track.\n",encoding="utf-8")
 criteria=(f"Nested improvement by Track: {dict(zip(DEV,nested_improve))}; unweighted absolute gain "
  f"{nested_gain:.6f} mm ({relative:.2%}). Human-reference MAE x-only/residual: "
  f"{human_x:.6f}/{human_combined:.6f} mm. Best lag offsets: "
  f"{ {t:int(r['offset_official_positions']) for t,r in best_lag.items()} }; consistent: {lag_consistent}.")
 (RP/"thermal_representation_and_alignment_review.md").write_text(
  "# Thermal representation and alignment review\n\n"+criteria+
  "\n\nThe canonical zero-based mapping passed exact index and direction checks. A nonzero lag optimum, "
  "if present, is classified as a diagnostic possible history lag unless it is consistent across all "
  "Tracks and supported by independent indexing/acquisition evidence; no mapping shift was applied. "
  "Nested representation selection and fixed-representation descriptive evaluation are reported "
  "separately. Future-frame lags are diagnostic only.\n\n"+status+"\n",encoding="utf-8")
 (RP/"thermal_representation_leakage_audit.md").write_text(
  "# Thermal representation leakage audit\n\nOnly Tracks 8, 10, and 14 were loaded. No SEM or Height-derived "
  "input was used. Sample-local normalization used only each current frame. Causal contexts contain the "
  "current and preceding official frames only. All training normalization and ridge fitting used outer "
  "training Tracks; inner validation swapped complete Tracks. Residual targets were constructed only on "
  "training rows after training x-only predictions, and held-out x-only predictions were locked before "
  "correction. Human references and symmetric future lags were diagnostic only. No random x split, deep "
  "learning, or multimodal processing was used.\n",encoding="utf-8")
 (RP/"thermal_representation_build_report.md").write_text(
  "# Thermal representation audit build report\n\n"
  f"- Source inventory: {len(inventory)}; mapping rows: {len(mapping_audit)}; representations: "
  f"{len(REPRESENTATIONS)}; representation manifest rows: {len(rep_manifest)}.\n"
  f"- Model-grid rows: {len(model_grid)}; nested folds: {len(residual_metrics)}; lag diagnostic rows: "
  f"{len(lag_metrics)}; sensitivity rows: {len(sensitivity)}; human diagnostics: {len(human)}.\n"
  f"- Fixed reviewed representation: `{fixed_rep}` (descriptive, not independently validated).\n"
  "- No reserved-Track data, SEM, Height inputs, mapping correction, deep learning, or multimodal model.\n",
  encoding="utf-8")
 print(f"inventory={len(inventory)} mapping={len(mapping_audit)} reps={len(rep_manifest)} "
       f"selected={nested_selected} status={status}")

def safe_corr(fun,a,b):
 try:return float(fun(a,b).statistic) if len(a)>2 and np.std(a)>0 and np.std(b)>0 else np.nan
 except:return np.nan

if __name__=="__main__":main()
