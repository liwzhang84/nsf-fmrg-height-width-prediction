#!/usr/bin/env python3
"""Freeze Height estimator v1 and build an objective development-only gate review."""
from __future__ import annotations
import csv,hashlib,math,sys
from collections import Counter,defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from src.height_boundary_estimator_v1 import (
    AGGREGATION_HALF_WIDTH_MM,BOUNDARY_METHOD_ID,CENTRAL_REGION,ESTIMATOR_ID,
    LEVELING_METHOD_ID,PARAMETER_SET_ID,SUBSTRATE_LEFT,SUBSTRATE_RIGHT,predict,width_center)
from src.human_review.data_access import height_data,profile
from src.official_coordinates import get_height_indices_in_window

DEV={8,10,14}
OUT=PROJECT/"outputs/height_labels"
HUMAN_OUT=PROJECT/"outputs/human_review/height_human_reference_v1_development.csv"
QA=PROJECT/"outputs/figures/height_quality_gate_qa"
FREEZE_REPORT=PROJECT/"outputs/reports/height_boundary_estimator_v1_freeze_decision.md"
GATE_REPORT=PROJECT/"outputs/reports/height_quality_gate_freeze_review.md"
BUILD_REPORT=PROJECT/"outputs/reports/height_quality_gate_build_report.md"
PILOT=PROJECT/"outputs/human_review/height_manual_review_pilot.csv"
REPEAT=PROJECT/"outputs/human_review/height_manual_review_repeat_1.csv"
ADJUD=PROJECT/"outputs/human_review/height_manual_review_adjudication_1.csv"
SUPPORT=PROJECT/"outputs/mappings/thermal_height_support.csv"
ADJUD_IDS={"T10_F105","T10_F300","T14_F079","T8_F009"}

def finite(v):
    try:return math.isfinite(float(v))
    except (TypeError,ValueError):return False
def truth(v):return str(v).strip().lower() in {"true","1","yes"}
def read_dev(path):
    rows=[]
    with Path(path).open(newline="",encoding="utf-8") as h:
        for row in csv.DictReader(h):
            if int(row["track_id"]) in DEV:rows.append(row)
    return rows
def write(path,rows):
    if not rows:raise RuntimeError(f"Required output is empty: {path}")
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
def fvalue(row,key,default=np.nan):
    return float(row[key]) if finite(row.get(key)) else default

def human_references():
    pilot={r["sample_id"]:r for r in read_dev(PILOT)}
    repeat={r["sample_id"]:r for r in read_dev(REPEAT)}
    adjud={r["sample_id"]:r for r in read_dev(ADJUD)}
    if set(adjud)!=ADJUD_IDS:raise RuntimeError(f"Unexpected adjudication override IDs: {sorted(adjud)}")
    rows=[]
    for sid,p in pilot.items():
        a=adjud.get(sid);r=repeat.get(sid)
        source="adjudication_1" if a else "pilot_1"
        status=(a["adjudicated_measurability"] if a else p["human_measurability"]).strip()
        left=fvalue(a,"adjudicated_y_left_mm") if a else fvalue(p,"manual_y_left_mm")
        right=fvalue(a,"adjudicated_y_right_mm") if a else fvalue(p,"manual_y_right_mm")
        available=status in {"measurable","uncertain"}
        if available and not (finite(left) and finite(right)):raise RuntimeError(f"Missing primary boundaries: {sid}")
        row={"sample_id":sid,"track_id":int(p["track_id"]),"x_actual_mm":float(p["x_actual_mm"]),
          "primary_reference_source":source,"primary_human_measurability":status,
          "primary_boundary_available":available,
          "primary_y_left_mm":left if available else "","primary_y_right_mm":right if available else "",
          "primary_width_mm":right-left if available else "","primary_center_mm":(left+right)/2 if available else "",
          "pilot_human_measurability":p["human_measurability"],
          "pilot_y_left_mm":p["manual_y_left_mm"],"pilot_y_right_mm":p["manual_y_right_mm"],
          "pilot_width_mm":p["manual_width_mm"],"pilot_review_notes":p.get("review_notes",""),
          "pilot_unmeasurable_reason":p.get("primary_unmeasurable_reason",""),
          "repeat_human_measurability":r.get("human_measurability","") if r else "",
          "repeat_y_left_mm":r.get("manual_y_left_mm","") if r else "",
          "repeat_y_right_mm":r.get("manual_y_right_mm","") if r else "",
          "repeat_width_mm":r.get("manual_width_mm","") if r else "",
          "repeat_review_notes":r.get("review_notes","") if r else "",
          "adjudicated_measurability":a.get("adjudicated_measurability","") if a else "",
          "adjudicated_y_left_mm":a.get("adjudicated_y_left_mm","") if a else "",
          "adjudicated_y_right_mm":a.get("adjudicated_y_right_mm","") if a else "",
          "adjudication_rationale":a.get("adjudication_rationale","") if a else ""}
        if r and finite(p.get("manual_width_mm")) and finite(r.get("manual_width_mm")):
            diff=abs(float(p["manual_width_mm"])-float(r["manual_width_mm"]))
            row.update(repeated_width_difference_mm=diff,
              midpoint_y_left_mm=(float(p["manual_y_left_mm"])+float(r["manual_y_left_mm"]))/2,
              midpoint_y_right_mm=(float(p["manual_y_right_mm"])+float(r["manual_y_right_mm"]))/2,
              midpoint_width_mm=(float(p["manual_width_mm"])+float(r["manual_width_mm"]))/2,
              reference_stability_category=("stable_le_0.05mm" if diff<=.05 else "ambiguous_gt_0.05mm"))
        else:row["reference_stability_category"]="not_repeated_or_boundary_unavailable"
        rows.append(row)
    return sorted(rows,key=lambda x:(x["track_id"],x["sample_id"]))

def supported_positions():
    rows=[]
    with SUPPORT.open(newline="",encoding="utf-8") as h:
        for r in csv.DictReader(h):
            track=int(r["track_id"])
            if track not in DEV:continue
            if truth(r["inside_height_coverage"]) and r["support_status"]=="official_thermal_height_supported":
                rows.append({"sample_id":f"T{track}_F{int(r['segment_frame_index']):03d}",
                  "track_id":track,"segment_frame_index":int(r["segment_frame_index"]),
                  "raw_frame_index":int(r["raw_frame_index"]),"x_actual_mm":float(r["thermal_x_center_mm"])})
    return rows

def longest_run(mask):
    best=cur=0
    for value in mask:
        cur=cur+1 if value else 0;best=max(best,cur)
    return best
def largest_gap(mask):
    return longest_run(~np.asarray(mask,dtype=bool))
def near_stats(valid,y,boundary,radius=.10):
    mask=(y>=boundary-radius)&(y<=boundary+radius)
    dy=float(np.nanmedian(np.diff(y)))
    return longest_run(valid[mask])*dy,largest_gap(valid[mask])*dy,int(valid[mask].sum())
def nearest_nan(valid,y,boundary):
    nan_y=y[~valid]
    return float(np.min(abs(nan_y-boundary))) if len(nan_y) else float("inf")
def baseline_stats(z,y,limits):
    vals=z[(y>=limits[0])&(y<=limits[1])]
    valid=np.isfinite(vals);clean=vals[valid]
    if not len(clean):return np.nan,np.nan,float(valid.mean()),0
    med=float(np.median(clean));mad=float(1.4826*np.median(abs(clean-med)))
    return med,mad,float(valid.mean()),int(valid.sum())
def prediction_at(track,x,agg,method=LEVELING_METHOD_ID):
    y,z,q=profile(track,x,agg,method);left,right,d=predict(z,y);width,center=width_center(left,right)
    return {"y":y,"z":z,"left":left,"right":right,"width":width,"center":center,"diag":d,"query":q}

def build_features(positions):
    rows=[];cache={}
    for pos in positions:
        track,x=pos["track_id"],pos["x_actual_mm"]
        preds={a:prediction_at(track,x,a) for a in (.025,.05,.10)}
        base=preds[.10];y,z=base["y"],base["z"];valid=np.isfinite(z)
        raw=height_data(track,LEVELING_METHOD_ID);q=base["query"]
        slab=raw["Z_leveled_mm"][:,q.index_start:q.index_end]
        lb,ln,lv,lc=baseline_stats(z,y,SUBSTRATE_LEFT);rb,rn,rv,rc=baseline_stats(z,y,SUBSTRATE_RIGHT)
        left,right=base["left"],base["right"];ok=finite(left) and finite(right)
        control_c=prediction_at(track,x,.05,"outside_quadratic_v1")
        control_d=prediction_at(track,x,.025,"local_side_linear_v1")
        comparisons=[preds[.025],control_c,control_d]
        row={**pos,"height_window_available":True,"frozen_estimator_id":ESTIMATOR_ID,
          "predicted_y_left_mm":left if ok else "","predicted_y_right_mm":right if ok else "",
          "predicted_width_mm":base["width"] if ok else "","predicted_center_mm":base["center"] if ok else "",
          "finite_boundary_prediction":ok,"estimator_exclusion_reason":"" if ok else base["diag"]["status"],
          "label_ready_for_training":"unresolved",
          "local_height_valid_fraction":float(valid.mean()),
          "aggregation_valid_fraction":float(np.isfinite(slab).mean()),
          "central_region_valid_fraction":float(valid[(y>=CENTRAL_REGION[0])&(y<=CENTRAL_REGION[1])].mean()),
          "left_substrate_valid_fraction":lv,"right_substrate_valid_fraction":rv,
          "left_substrate_valid_count":lc,"right_substrate_valid_count":rc,
          "left_baseline_mm":lb,"right_baseline_mm":rb,"left_local_noise_mm":ln,"right_local_noise_mm":rn}
        for a,p in preds.items():
            token=str(a).replace(".","p")
            row[f"predicted_left_agg_{token}_mm"]=p["left"] if finite(p["left"]) else ""
            row[f"predicted_right_agg_{token}_mm"]=p["right"] if finite(p["right"]) else ""
            row[f"predicted_width_agg_{token}_mm"]=p["width"] if finite(p["width"]) else ""
            row[f"predicted_center_agg_{token}_mm"]=p["center"] if finite(p["center"]) else ""
        if ok:
            lrun,lgap,lsupport=near_stats(valid,y,left);rrun,rgap,rsupport=near_stats(valid,y,right)
            row.update(left_boundary_distance_to_nearest_nan_mm=nearest_nan(valid,y,left),
              right_boundary_distance_to_nearest_nan_mm=nearest_nan(valid,y,right),
              left_longest_valid_run_near_boundary_mm=lrun,right_longest_valid_run_near_boundary_mm=rrun,
              left_largest_nan_gap_near_boundary_mm=lgap,right_largest_nan_gap_near_boundary_mm=rgap,
              left_transition_valid_support_count=lsupport,right_transition_valid_support_count=rsupport,
              left_gradient_magnitude=base["diag"]["left_gradient_magnitude"],
              right_gradient_magnitude=base["diag"]["right_gradient_magnitude"],
              left_gradient_prominence=base["diag"]["left_gradient_magnitude"]/max(ln,1e-9),
              right_gradient_prominence=base["diag"]["right_gradient_magnitude"]/max(rn,1e-9),
              left_profile_baseline_contrast_mm=abs(float(z[base["diag"]["left_index"]])-lb),
              right_profile_baseline_contrast_mm=abs(float(z[base["diag"]["right_index"]])-rb),
              left_distance_to_candidate_limit_mm=left-CENTRAL_REGION[0],
              right_distance_to_candidate_limit_mm=CENTRAL_REGION[1]-right,
              distance_to_candidate_limit_mm=min(left-CENTRAL_REGION[0],CENTRAL_REGION[1]-right))
            for key in ("left","right","width","center"):
                vals=[p[key] for p in preds.values() if finite(p[key])]
                row[f"aggregation_sensitivity_{key}_mm"]=max(vals)-min(vals) if len(vals)>=2 else ""
                ds=[abs(base[key]-p[key]) for p in comparisons if finite(p[key])]
                row[f"maximum_method_disagreement_{key}_mm"]=max(ds) if ds else ""
                for name,p in zip(("control_B","control_C","control_D"),comparisons):
                    row[f"{name}_{key}_disagreement_mm"]=abs(base[key]-p[key]) if finite(p[key]) else ""
        rows.append(row);cache[pos["sample_id"]]=base
    # Neighbor continuity uses official adjacent positions only.
    index={(r["track_id"],r["segment_frame_index"]):r for r in rows}
    for row in rows:
        neighbors=[index[k] for k in ((row["track_id"],row["segment_frame_index"]-1),
                                     (row["track_id"],row["segment_frame_index"]+1)) if k in index]
        valid_neighbors=[n for n in neighbors if truth(n["finite_boundary_prediction"])]
        row["neighbor_availability"]="two_sided" if len(valid_neighbors)==2 else "one_sided" if len(valid_neighbors)==1 else "none"
        for key in ("left","right","width","center"):
            current=fvalue(row,f"predicted_{'y_' if key in ('left','right') else ''}{key}_mm")
            vals=[fvalue(n,f"predicted_{'y_' if key in ('left','right') else ''}{key}_mm") for n in valid_neighbors]
            vals=[v for v in vals if finite(v)]
            row[f"{key}_boundary_neighbor_difference_mm" if key in ("left","right") else f"{key}_neighbor_difference_mm"]=(
                float(np.mean([abs(current-v) for v in vals])) if finite(current) and vals else "")
            trio=[v for v in [current]+vals if finite(v)]
            row[f"neighborhood_{key}_mad_mm"]=float(np.median(abs(np.array(trio)-np.median(trio)))) if len(trio)>=2 else ""
    # Exact repeated-width/cap diagnostics are descriptive objective features.
    counts=Counter(round(fvalue(r,"predicted_width_mm"),6) for r in rows if finite(r.get("predicted_width_mm")))
    apparent=max(counts) if counts else np.nan
    for r in rows:
        w=round(fvalue(r,"predicted_width_mm"),6)
        r["exact_repeated_width_count"]=counts.get(w,0) if finite(w) else 0
        r["apparent_cap_flag"]=bool(finite(w) and w==apparent and counts[w]>=3)
        r["proximity_to_apparent_candidate_width_cap_mm"]=apparent-w if finite(w) else ""
    return rows,cache

FEATURES=("aggregation_valid_fraction","central_region_valid_fraction",
 "left_substrate_valid_fraction","right_substrate_valid_fraction",
 "left_boundary_distance_to_nearest_nan_mm","right_boundary_distance_to_nearest_nan_mm",
 "left_gradient_prominence","right_gradient_prominence",
 "aggregation_sensitivity_width_mm","aggregation_sensitivity_center_mm",
 "maximum_method_disagreement_width_mm","maximum_method_disagreement_center_mm",
 "distance_to_candidate_limit_mm")

def gates():
    return {
      "accept_all_finite":lambda r:truth(r["finite_boundary_prediction"]),
      "valid_ge_0p50":lambda r:truth(r["finite_boundary_prediction"]) and fvalue(r,"aggregation_valid_fraction")>=.50,
      "valid_ge_0p70":lambda r:truth(r["finite_boundary_prediction"]) and fvalue(r,"aggregation_valid_fraction")>=.70,
      "side_support_ge_0p50":lambda r:truth(r["finite_boundary_prediction"]) and min(fvalue(r,"left_substrate_valid_fraction"),fvalue(r,"right_substrate_valid_fraction"))>=.50,
      "agg_width_le_0p10":lambda r:truth(r["finite_boundary_prediction"]) and fvalue(r,"aggregation_sensitivity_width_mm")<=.10,
      "method_width_le_0p15":lambda r:truth(r["finite_boundary_prediction"]) and fvalue(r,"maximum_method_disagreement_width_mm")<=.15,
      "nan_distance_ge_0p02":lambda r:truth(r["finite_boundary_prediction"]) and min(fvalue(r,"left_boundary_distance_to_nearest_nan_mm"),fvalue(r,"right_boundary_distance_to_nearest_nan_mm"))>=.02,
      "balanced_rule_v1":lambda r:truth(r["finite_boundary_prediction"]) and fvalue(r,"aggregation_valid_fraction")>=.50 and fvalue(r,"aggregation_sensitivity_width_mm")<=.15 and fvalue(r,"maximum_method_disagreement_width_mm")<=.20,
      "low_false_accept_rule_v1":lambda r:truth(r["finite_boundary_prediction"]) and min(fvalue(r,"left_substrate_valid_fraction"),fvalue(r,"right_substrate_valid_fraction"))>=.70 and fvalue(r,"aggregation_sensitivity_width_mm")<=.10 and fvalue(r,"maximum_method_disagreement_width_mm")<=.15,
      "high_yield_rule_v1":lambda r:truth(r["finite_boundary_prediction"]) and (fvalue(r,"aggregation_valid_fraction")>=.40 or fvalue(r,"maximum_method_disagreement_width_mm")<=.25)}

def gate_metric(gate_id,decision,refs,features,scope="aggregate"):
    joined=[(r,features[r["sample_id"]]) for r in refs if r["sample_id"] in features and
            (scope=="aggregate" or int(r["track_id"])==int(scope))]
    tp=fp=fn=tn=0;meas_total=meas_hit=unc_total=unc_hit=0;errors=[]
    for ref,feat in joined:
        target=truth(ref["primary_boundary_available"]);accepted=bool(decision(feat))
        tp+=target and accepted;fp+=(not target) and accepted;fn+=target and not accepted;tn+=(not target) and not accepted
        tier=ref["primary_human_measurability"]
        if tier=="measurable":meas_total+=1;meas_hit+=accepted
        if tier=="uncertain":unc_total+=1;unc_hit+=accepted
        if target and accepted and finite(feat.get("predicted_y_left_mm")):
            pl,pr=fvalue(feat,"predicted_y_left_mm"),fvalue(feat,"predicted_y_right_mm")
            l,r=fvalue(ref,"primary_y_left_mm"),fvalue(ref,"primary_y_right_mm")
            errors.append((abs(pl-l),abs(pr-r),abs((pr-pl)-(r-l)),(pr-pl)-(r-l),
                           abs((pl+pr-l-r)/2),(pl+pr-l-r)/2))
    arr=np.array(errors) if errors else np.empty((0,6))
    result={"gate_id":gate_id,"track_scope":scope,"human_boundary_available_count":tp+fn,
      "human_unmeasurable_count":fp+tn,"accepted_boundary_available_count":tp,
      "false_accepted_unmeasurable_count":fp,"false_rejected_boundary_available_count":fn,
      "accepted_precision":tp/(tp+fp) if tp+fp else "not_estimable",
      "boundary_available_recall":tp/(tp+fn) if tp+fn else "not_estimable",
      "measurable_only_recall":meas_hit/meas_total if meas_total else "not_estimable",
      "uncertain_only_recall":unc_hit/unc_total if unc_total else "not_estimable",
      "unmeasurable_specificity":tn/(tn+fp) if tn+fp else "not_estimable",
      "conditional_geometry_sample_count":len(errors)}
    for name,col,func in (("left_boundary_mae_mm",0,np.mean),("right_boundary_mae_mm",1,np.mean),
      ("width_mae_mm",2,np.mean),("signed_width_bias_mm",3,np.mean),("center_mae_mm",4,np.mean),
      ("signed_center_bias_mm",5,np.mean),("maximum_width_error_mm",2,np.max),("maximum_center_error_mm",4,np.max)):
        result[name]=float(func(arr[:,col])) if len(arr) else "not_estimable"
    result["combined_boundary_mae_mm"]=float(np.mean(arr[:,:2])) if len(arr) else "not_estimable"
    return result

def model_decision(train,feature_map,kind):
    usable=[(r,feature_map[r["sample_id"]]) for r in train if r["sample_id"] in feature_map]
    names=("aggregation_valid_fraction","aggregation_sensitivity_width_mm",
           "maximum_method_disagreement_width_mm","distance_to_candidate_limit_mm")
    X=np.array([[fvalue(x,n,0) for n in names] for _,x in usable]);y=np.array([truth(r["primary_boundary_available"]) for r,_ in usable],float)
    if kind=="tree":
        best=None
        for j in range(X.shape[1]):
            for threshold in np.unique(X[:,j]):
                for direction in (1,-1):
                    pred=(direction*X[:,j]>=direction*threshold)
                    loss=np.mean(pred!=y)
                    if best is None or loss<best[0]:best=(loss,j,threshold,direction)
        _,j,t,d=best
        return lambda r:truth(r["finite_boundary_prediction"]) and d*fvalue(r,names[j],0)>=d*t
    mean=X.mean(0);scale=X.std(0);scale[scale==0]=1;Z=(X-mean)/scale
    w=np.zeros(Z.shape[1]+1)
    for _ in range(1000):
        score=np.clip(w[0]+Z@w[1:],-20,20);prob=1/(1+np.exp(-score))
        grad=np.r_[np.mean(prob-y),Z.T@(prob-y)/len(y)]+.05*np.r_[0,w[1:]]
        w-=.1*grad
    return lambda r:truth(r["finite_boundary_prediction"]) and (w[0]+((np.array([fvalue(r,n,0) for n in names])-mean)/scale)@w[1:]>=0)

def qa_figure(sid,ref,feat,cache,feature_map,gate_id,decision,decision_details=None,
              output_dir=None):
    track=int(feat["track_id"]);x=float(feat["x_actual_mm"]);base=height_data(track,LEVELING_METHOD_ID)
    mask=(base["x_actual_mm"]>=x-.4)&(base["x_actual_mm"]<=x+.4);p=cache[sid]
    fig,axes=plt.subplots(1,3,figsize=(17,4.7))
    axes[0].imshow(base["Z_raw_mm"][:,mask]*1000,origin="lower",aspect="auto",
      extent=[base["x_actual_mm"][mask][0],base["x_actual_mm"][mask][-1],base["y_mm"][0],base["y_mm"][-1]])
    axes[0].axvline(x,color="red");axes[0].set(title=f"{sid} raw Height",xlabel="x mm",ylabel="y mm")
    axes[1].plot(p["y"],p["z"]*1000,color="black",label="frozen profile")
    for v,c,label in ((p["left"],"cyan","frozen boundaries"),(p["right"],"cyan",None),
      (fvalue(ref,"primary_y_left_mm"),"purple","human primary"),(fvalue(ref,"primary_y_right_mm"),"purple",None)):
        if finite(v):axes[1].axvline(v,color=c,ls="--",label=label)
    bad=~np.isfinite(p["z"])
    if bad.any():axes[1].plot(p["y"][bad],np.zeros(bad.sum()),"rx",ms=3,label="NaN")
    axes[1].axvspan(*SUBSTRATE_LEFT,color="green",alpha=.08);axes[1].axvspan(*SUBSTRATE_RIGHT,color="green",alpha=.08)
    axes[1].legend(fontsize=7);axes[1].set(title="Frozen estimator and reference",xlabel="y mm",ylabel="µm")
    clauses=[
      f"gate: {gate_id}",f"decision: {'accept' if decision(feat) else 'reject'}",
      f"clause finite: {truth(feat['finite_boundary_prediction'])}",
      f"clause left side>=0.50: {fvalue(feat,'left_substrate_valid_fraction')>=.50}",
      f"clause right side>=0.50: {fvalue(feat,'right_substrate_valid_fraction')>=.50}",
      f"valid={fvalue(feat,'aggregation_valid_fraction'):.3f}",
      f"side support L/R={fvalue(feat,'left_substrate_valid_fraction'):.3f}/{fvalue(feat,'right_substrate_valid_fraction'):.3f}",
      f"agg Δwidth={fvalue(feat,'aggregation_sensitivity_width_mm'):.3f} mm",
      f"method Δwidth={fvalue(feat,'maximum_method_disagreement_width_mm'):.3f} mm",
      f"neighbor={feat['neighbor_availability']}",
      f"NaN distance L/R={fvalue(feat,'left_boundary_distance_to_nearest_nan_mm'):.3f}/{fvalue(feat,'right_boundary_distance_to_nearest_nan_mm'):.3f} mm"]
    if decision_details:
        clauses[1:1]=decision_details
    by_key={(int(r["track_id"]),int(r["segment_frame_index"])):r for r in feature_map.values()}
    for delta in (-1,1):
        neighbor=by_key.get((track,int(feat["segment_frame_index"])+delta))
        if neighbor:
            clauses.append(f"x{delta:+d} L/R/W/C="
              f"{neighbor.get('predicted_y_left_mm','')}/{neighbor.get('predicted_y_right_mm','')}/"
              f"{neighbor.get('predicted_width_mm','')}/{neighbor.get('predicted_center_mm','')}")
    axes[2].axis("off");axes[2].text(.02,.98,"\n".join(clauses),va="top",family="monospace")
    target_dir=Path(output_dir) if output_dir else QA
    fig.tight_layout();target_dir.mkdir(parents=True,exist_ok=True)
    fig.savefig(target_dir/f"{sid}_quality_gate_QA.png",dpi=150);plt.close(fig)

def main():
    refs=human_references();write(HUMAN_OUT,refs)
    positions=supported_positions()
    expected={8:399,10:400,14:393};actual=Counter(r["track_id"] for r in positions)
    if dict(actual)!=expected:raise RuntimeError(f"Development support reconciliation failed: {dict(actual)}")
    feature_rows,cache=build_features(positions);feature_map={r["sample_id"]:r for r in feature_rows}
    write(OUT/"height_boundary_predictions_v1_development.csv",feature_rows)
    write(OUT/"height_quality_features_v1_development.csv",feature_rows)
    gate_map=gates()
    diagnostic_map={
      "diagnostic_depth1_tree":model_decision(refs,feature_map,"tree"),
      "diagnostic_regularized_logistic":model_decision(refs,feature_map,"logistic")}
    evaluation_map={**gate_map,**diagnostic_map}
    metrics=[];by_track=[];pilot_predictions=[];yield_rows=[]
    for gid,decision in evaluation_map.items():
        aggregate=gate_metric(gid,decision,refs,feature_map)
        aggregate["model_role"]="diagnostic_only" if gid.startswith("diagnostic_") else "deterministic_candidate"
        metrics.append(aggregate)
        for track in sorted(DEV):by_track.append(gate_metric(gid,decision,refs,feature_map,track))
        for ref in refs:
            feat=feature_map[ref["sample_id"]];accepted=decision(feat)
            pilot_predictions.append({"gate_id":gid,"sample_id":ref["sample_id"],"track_id":ref["track_id"],
              "gate_accept":accepted,"primary_boundary_available":ref["primary_boundary_available"],
              "primary_human_measurability":ref["primary_human_measurability"],
              "predicted_y_left_mm":feat["predicted_y_left_mm"],"predicted_y_right_mm":feat["predicted_y_right_mm"]})
        accepted=[r for r in feature_rows if decision(r)]
        reasons=Counter("accepted" if decision(r) else ("nonfinite_boundary" if not truth(r["finite_boundary_prediction"]) else "objective_rule_failed") for r in feature_rows)
        yield_rows.append({"gate_id":gid,"development_position_count":len(feature_rows),
          "accepted_position_count":len(accepted),"accepted_fraction":len(accepted)/len(feature_rows),
          "accepted_track_8":sum(r["track_id"]==8 for r in accepted),
          "accepted_track_10":sum(r["track_id"]==10 for r in accepted),
          "accepted_track_14":sum(r["track_id"]==14 for r in accepted),
          "rejection_reason_distribution":";".join(f"{k}:{v}" for k,v in sorted(reasons.items()))})
    # Complete-Track leave-one-out: fixed rules plus train-fitted diagnostics.
    loto=[]
    for held in sorted(DEV):
        train=[r for r in refs if int(r["track_id"])!=held];test=[r for r in refs if int(r["track_id"])==held]
        for gid,decision in gate_map.items():
            row=gate_metric(gid,decision,test,feature_map);row.update(held_out_track=held,model_role="shared_fixed_rule");loto.append(row)
        training_scores=[gate_metric(gid,decision,train,feature_map) for gid,decision in gate_map.items()]
        selected=max(training_scores,key=lambda r:(
          int(r["false_accepted_unmeasurable_count"])<=1,
          fvalue(r,"boundary_available_recall",0)>=.70,
          fvalue(r,"accepted_precision",0)+fvalue(r,"boundary_available_recall",0),
          -fvalue(r,"width_mae_mm",99)))
        selected_id=selected["gate_id"]
        row=gate_metric("selected_shared_rule",gate_map[selected_id],test,feature_map)
        row.update(held_out_track=held,model_role="selected_on_other_two_tracks",
                   selected_rule_id=selected_id);loto.append(row)
        for kind in ("tree","logistic"):
            decision=model_decision(train,feature_map,kind)
            row=gate_metric(f"diagnostic_{kind}",decision,test,feature_map)
            row.update(held_out_track=held,model_role="diagnostic_train_tracks_only");loto.append(row)
    # Pareto categories and transparent criteria.
    ungated=metrics[0]
    for row in metrics:
        fp=int(row["false_accepted_unmeasurable_count"]);rec=fvalue(row,"boundary_available_recall",0)
        row["pareto_category"]="low_false_accept" if fp<=1 else "balanced_precision_recall" if rec>=.7 else "high_yield_riskier"
        row["uses_only_objective_features"]=True;row["shared_parameters"]=True;row["complete_track_loto"]=True
        row["criterion_false_accept_le_1"]=fp<=1;row["criterion_recall_ge_0p70"]=rec>=.70
        row["criterion_width_improved"]=finite(row["width_mae_mm"]) and fvalue(row,"width_mae_mm")<.0997296
        row["criterion_center_not_materially_worse"]=finite(row["center_mae_mm"]) and fvalue(row,"center_mae_mm")<=.0642573
        tracks=[x for x in by_track if x["gate_id"]==row["gate_id"]]
        row["severe_track_failure"]=any(fvalue(x,"boundary_available_recall",0)<.50 for x in tracks)
    deterministic_metrics=[r for r in metrics if r["model_role"]=="deterministic_candidate"]
    eligible=[r for r in deterministic_metrics if truth(r["criterion_false_accept_le_1"]) and truth(r["criterion_recall_ge_0p70"])]
    recommended=max(eligible,key=lambda r:(fvalue(r,"accepted_precision",0),fvalue(r,"boundary_available_recall",0),
                                           -fvalue(r,"width_mae_mm",99))) if eligible else max(deterministic_metrics,key=lambda r:fvalue(r,"accepted_precision",0)+fvalue(r,"boundary_available_recall",0))
    decision=gate_map[recommended["gate_id"]]
    # QA union of all required error types plus explicit scientific cases.
    qa_ids={"T8_F013","T8_F106","T10_F042","T10_F105","T14_F079","T14_F207"}
    ref_map={r["sample_id"]:r for r in refs}
    for ref in refs:
        feat=feature_map[ref["sample_id"]];target=truth(ref["primary_boundary_available"]);accepted=decision(feat)
        if (accepted and not target) or (not accepted and target):
            qa_ids.add(ref["sample_id"])
        if accepted and target:
            we=abs(fvalue(feat,"predicted_width_mm")-fvalue(ref,"primary_width_mm"))
            ce=abs(fvalue(feat,"predicted_center_mm")-fvalue(ref,"primary_center_mm"))
            if we>.15 or ce>.10:qa_ids.add(ref["sample_id"])
    QA.mkdir(parents=True,exist_ok=True)
    for old in QA.glob("*.png"):old.unlink()
    for sid in sorted(qa_ids):qa_figure(sid,ref_map[sid],feature_map[sid],cache,feature_map,recommended["gate_id"],decision)
    write(OUT/"height_quality_gate_candidate_metrics.csv",metrics)
    write(OUT/"height_quality_gate_metrics_by_track.csv",by_track)
    write(OUT/"height_quality_gate_loto_metrics.csv",loto)
    write(OUT/"height_quality_gate_predictions_on_pilot.csv",pilot_predictions)
    write(OUT/"height_quality_gate_development_yield.csv",yield_rows)
    available=sum(truth(r["primary_boundary_available"]) for r in refs)
    tiers=Counter(r["primary_human_measurability"] for r in refs)
    FREEZE_REPORT.parent.mkdir(parents=True,exist_ok=True)
    FREEZE_REPORT.write_text(
      "# Height boundary estimator v1 freeze decision\n\n"
      f"`{ESTIMATOR_ID}` freezes Candidate A: `{LEVELING_METHOD_ID}` + `{BOUNDARY_METHOD_ID}` + "
      f"`{PARAMETER_SET_ID}`, aggregation half-width 0.10 mm. It retained 18/18 finite development "
      "predictions with width MAE 0.0997296 mm, combined boundary MAE 0.0806171 mm, center MAE "
      "0.0542573 mm, signed width bias -0.047841 mm, maximum width error 0.227986 mm, and maximum "
      "center error 0.166861 mm. Controls B–D, a label-fitted global offset, and the 54-combination "
      "outer-transition experiment were compared. The refinement was rejected because its reviewed "
      "best coverage was 0.5556 with worse aggregate errors and severe per-Track degradation. The "
      "global offset was rejected because it was label-fitted and did not improve center position.\n\n"
      "Known limitations include systematic underestimation, notable negative signed-width bias on "
      "one development Track, repeatable narrow predictions at T8_F013 and T10_F042, the center shift "
      "at T14_F207, and no improvement on the declared stable-repeat criterion.\n\n"
      "Only development Tracks 8, 10, and 14 were loaded. Reserved data were not loaded or inspected. "
      "Quality gating remains unresolved and cannot silently modify this frozen estimator.\n",encoding="utf-8")
    feature_docs=[
      ("valid fractions/counts","fraction or count","nullable only when boundary unavailable","build_features","local/aggregation/central and independent side support"),
      ("boundary NaN distance/run/gap","mm or count","yes","near_stats/nearest_nan","support and missing-data proximity"),
      ("gradient magnitude/prominence","mm/mm and ratio","yes","predict/build_features","edge strength relative to robust side MAD"),
      ("profile-baseline contrast","mm","yes","build_features","boundary height departure from independent substrate baseline"),
      ("aggregation sensitivity","mm","yes","prediction_at/build_features","spread over 0.025/0.05/0.10 mm aggregation"),
      ("method disagreement","mm","yes","prediction_at/build_features","pairwise and maximum disagreement with Controls B–D"),
      ("neighbor continuity/MAD","mm","yes at endpoints/missing neighbor","build_features second pass","official ±0.2 mm continuity without replacement"),
      ("candidate-limit/cap diagnostics","mm/count/boolean","yes","build_features","limit proximity and repeated apparent width")]
    table="\n".join("| "+" | ".join(x)+" |" for x in feature_docs)
    review_lines=["| Gate | Role | FA | Recall | Precision | Width MAE | Center MAE | Development accepted | Review category |",
      "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    yields={r["gate_id"]:r for r in yield_rows}
    for row in metrics:
        review_lines.append(f"| `{row['gate_id']}` | {row['model_role']} | "
          f"{row['false_accepted_unmeasurable_count']} | {row['boundary_available_recall']} | "
          f"{row['accepted_precision']} | {row['width_mae_mm']} | {row['center_mae_mm']} | "
          f"{yields[row['gate_id']]['accepted_position_count']} | {row['pareto_category']} |")
    final_status=("QUALITY GATE READY FOR HUMAN FREEZE DECISION" if eligible
                  else "QUALITY GATE REQUIRES REVISION")
    GATE_REPORT.write_text(
      "# Height objective quality-gate freeze review\n\n"
      f"Human primary references: {len(refs)} total; {available} boundary-available and "
      f"{len(refs)-available} unmeasurable after exactly four adjudication overrides "
      f"(measurable={tiers['measurable']}, uncertain={tiers['uncertain']}). Human confidence is not a feature.\n\n"
      "## Objective feature dictionary\n\n| Feature group | Units | Nullability | Code path | Interpretation |\n"
      "|---|---|---|---|---|\n"+table+"\n\n"
      f"Evaluated {len(gate_map)} shared deterministic gates and two diagnostic model families using "
      "three complete-Track leave-one-out folds. No random row split, sample exceptions, or "
      "Track-specific thresholds were used.\n\n"
      "## Gate review table\n\n"+"\n".join(review_lines)+"\n\n"
      f"Candidate highlighted for human review: `{recommended['gate_id']}`. It has "
      f"false accepts={recommended['false_accepted_unmeasurable_count']}, recall="
      f"{recommended['boundary_available_recall']}, precision={recommended['accepted_precision']}, "
      f"conditional width MAE={recommended['width_mae_mm']} mm, center MAE="
      f"{recommended['center_mae_mm']} mm. Development yield is reported separately and is not a "
      "full-dataset yield estimate. Low-false-accept, balanced, and higher-yield trade-offs remain "
      "visible in the candidate table. This recommendation does not freeze a gate.\n\n"
      "Reserved data were not loaded, inspected, plotted, or used. Boundary coordinates remain the "
      "frozen estimator outputs regardless of gate decisions.\n\n"
      f"{final_status}\n",encoding="utf-8")
    BUILD_REPORT.write_text(
      "# Height quality-gate build report\n\n"
      f"- Command: `python3 scripts/build_height_quality_gate_review.py`\n"
      f"- Frozen estimator: {LEVELING_METHOD_ID} / {BOUNDARY_METHOD_ID} / {PARAMETER_SET_ID} / 0.10 mm.\n"
      f"- Development support reconciled: {len(positions)} positions ({dict(actual)}).\n"
      f"- Human references: {len(refs)}; boundary-available={available}; unmeasurable={len(refs)-available}.\n"
      f"- Objective feature rows: {len(feature_rows)}; deterministic gate candidates: {len(gate_map)}; "
      f"diagnostic models: {len(diagnostic_map)}; "
      f"LOTO rows: {len(loto)}; QA figures: {len(list(QA.glob('*.png')))}.\n"
      "- Inputs inspected: official support/mapping conventions, frozen candidate generator, pilot, "
      "repeat, adjudication, agreement, Candidate A/control tables, and outer-transition outputs.\n"
      "- Created the frozen configuration, estimator implementation, consolidated reference, predictions, "
      "features, gate metrics, yield, LOTO, QA, and review reports. Source reviews and candidates were not modified.\n"
      "- Verification command: `python3 -m unittest tests.test_height_manual_review "
      "tests.test_height_outer_transition tests.test_height_quality_gate_review` (53 tests expected).\n"
      "- Reserved data were not loaded. No final label file, gate config, or model training was created.\n",encoding="utf-8")
    print(f"development={len(positions)}, human={len(refs)} ({available} available), gates={len(gate_map)}, QA={len(qa_ids)}, recommended={recommended['gate_id']}")

if __name__=="__main__":main()
