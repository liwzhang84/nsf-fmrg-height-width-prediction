#!/usr/bin/env python3
"""Strict, read-only production-implementation equivalence audit."""
from __future__ import annotations
import hashlib, importlib.util, json, os
from collections import Counter
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR","/private/tmp/nsf_mplconfig")
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[1]
LOCK_HASH="2fb60cbb83a04ea9819c3bfff9fa39b9765dc9bad74b4a7d0693103af013be1b"
AMEND_HASH="3c0de4b78b5417cd60df27bbb9e059bcf27986aaadd8964d14e51dbf831a847a"
EXPECTED_DEV_FEATURE="4f8473e69b6c05d0049c920676b6cc0891b708bca176c86fd6acc51815033203"
DEV=(8,10,14)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def mod(path,name):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def norm_path(v):
 if pd.isna(v) or str(v).strip()=="":return ""
 return str(Path(str(v)).resolve())
def f64hash(a):return hashlib.sha256(np.asarray(a,dtype="<f8").tobytes()).hexdigest()
def table_hash(df):
 records=df.sort_values("sample_id").replace({np.nan:None}).to_dict("records")
 return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()
def immutable_paths():
 amendment=json.loads((ROOT/"configs/final_test_sem_source_compatibility_amendment_v1.yaml").read_text())
 p=[ROOT/"configs/final_height_width_model_lock_v1.yaml",
  ROOT/"configs/final_test_sem_source_compatibility_amendment_v1.yaml"]
 p+=sorted((ROOT/"outputs/models/final_height_width_v1").glob("*"))
 
 def blocked_path(x):
  direct=ROOT/x
  if direct.exists():return direct
  name=Path(x).name
  archive=(ROOT/"outputs/reports/archive/first_blocked_attempt"/name if x.startswith("outputs/reports/") else
           ROOT/"outputs/sem_final_test/archive/first_blocked_attempt"/name)
  return archive
 p += [blocked_path(x) for x in amendment["first_blocked_attempt_paths"]]
 p += [ROOT/x for x in [
  "outputs/sem_final_test_v2/sem_mapping_v1_track21.csv",
  "outputs/height_labels/height_labels_v1_track21_final_test.csv",
  "outputs/manifests/thermal_height_manifest_v1_track21_final_test.csv",
  "outputs/manifests/sem_height_manifest_v1_track21_final_test.csv",
  "outputs/models/final_track21_height_width_predictions.csv",
  "outputs/models/final_track21_height_width_metrics.csv",
  "outputs/models/final_track21_height_width_cohort_summary.csv",
  "outputs/models/final_track21_height_width_subgroup_metrics.csv",
  "outputs/reports/final_track21_height_width_test_review_v2.md",
  "outputs/reports/final_track21_height_width_leakage_audit_v2.md",
  "outputs/reports/final_track21_height_width_build_report_v2.md"]]
 return p
def add(rows,section,track,req,expected,observed,tol,passed,path,notes=""):
 rows.append({"audit_section":section,"track_id":track,"requirement_id":req,
  "expected_value":expected,"observed_value":observed,"tolerance":tol,"pass":bool(passed),
  "evidence_path":path,"notes":notes})
def compare_mapping(old,new,track,mismatches):
 old=old.sort_values("sample_id").reset_index(drop=True);new=new.sort_values("sample_id").reset_index(drop=True)
 fields=[("track_id",0),("official_position_index",0),("x_actual_mm",1e-12),
  ("sem_tile_id",0),("sem_local_coordinate_px",1e-9),("sem_mosaic_coordinate_px",1e-9),
  ("along_track_axis",0),("increasing_x_direction",0),("mapping_available",0),
  ("mapping_exclusion_reason",0),("sem_source_path","path")]
 counts=Counter()
 for i,(a,b) in enumerate(zip(old.itertuples(),new.itertuples())):
  if a.sample_id!=b.sample_id:
   counts["cohort_membership_difference"]+=1;continue
  for field,tol in fields:
   av=getattr(a,field);bv=getattr(b,field)
   if tol=="path":same=norm_path(av)==norm_path(bv);diff=np.nan;cat="path_format_only"
   elif isinstance(tol,float) and tol>0:
    if pd.isna(av) and pd.isna(bv):same=True;diff=0
    else:diff=abs(float(av)-float(bv));same=diff<=tol
    cat="floating_point_only" if same else ("local_coordinate_difference" if "coordinate" in field else "unknown")
   elif field in ("track_id","official_position_index","sem_tile_id"):
    amiss=pd.isna(av) or str(av).strip()==""
    bmiss=pd.isna(bv) or str(bv).strip()==""
    if amiss and bmiss:same=True
    elif amiss or bmiss:same=False
    else:same=float(av)==float(bv)
    diff=0 if same else (abs(float(av)-float(bv)) if not pd.isna(av) and not pd.isna(bv) else np.nan)
    cat="tile_assignment_difference" if field=="sem_tile_id" else "unknown"
   else:
    av="" if pd.isna(av) else av;bv="" if pd.isna(bv) else bv;same=str(av)==str(bv);diff=np.nan
    cat="tile_assignment_difference" if field=="sem_tile_id" else "unknown"
   if not same:
    counts[cat]+=1
    if len(mismatches)<50:mismatches.append({"track_id":track,"sample_id":a.sample_id,
     "official_x":a.x_actual_mm,"field":field,"old_value":av,"reconstructed_value":bv,
     "absolute_difference":diff,"category":cat})
 return counts
def predict_art(a,x):
 x=np.asarray(x,float);med=np.array(a["imputation_values"]);mean=np.array(a["scaler_mean"]);std=np.array(a["scaler_scale"])
 x=np.where(np.isfinite(x),x,med)
 return float(a["intercept"])+((x-mean)/std)@np.array(a["coefficients"])

def main():
 paths=immutable_paths()
 before={str(p.relative_to(ROOT)):sha(p) for p in paths}
 if before["configs/final_height_width_model_lock_v1.yaml"]!=LOCK_HASH:raise RuntimeError("Lock hash mismatch")
 if before["configs/final_test_sem_source_compatibility_amendment_v1.yaml"]!=AMEND_HASH:raise RuntimeError("Amendment hash mismatch")
 # Only now may SEM pixels be read.
 gen=mod("scripts/generalized_sem_mapping_v1.py","generalized_production")
 featmod=mod("scripts/run_sem_multimodal_deadline_models.py","locked_sem_features")
 rows=[];mismatches=[];mismatch_counts=Counter();dev_maps=[];dev_mans=[];dev_summary={};dev_up={};dev_track_hashes={};dev_mapping_mismatches=0
 frozen_map=pd.read_csv(ROOT/"outputs/sem_deadline/sem_mapping_deadline_v1.csv")
 frozen_man=pd.read_csv(ROOT/"outputs/manifests/sem_height_manifest_deadline_v1_development.csv")
 labels=pd.read_csv(ROOT/"outputs/height_labels/height_labels_v1_development.csv")
 for track in DEV:
  pos=labels[labels.track_id==track][["sample_id","track_id","official_position_index","x_actual_mm","primary_label_eligible"]]
  mp,mn,s,u,meta=gen.reconstruct(track,pos,featmod.representations)
  old=frozen_map[frozen_map.track_id==track]
  c=compare_mapping(old,mp,track,mismatches);mismatch_counts.update(c)
  dev_mapping_mismatches+=sum(c.values())
  dev_maps.append(mp);dev_mans.append(mn);dev_summary.update(s);dev_up.update(u)
  available=mn[(mn.primary_label_eligible==True)&(mn.sem_input_available==True)].sample_id.tolist()
  mat=np.stack([s[x] for x in sorted(available)])
  dev_track_hashes[str(track)]=f64hash(mat)
  add(rows,"development_mapping",track,"mapping_row_count",len(old),len(mp),"exact",len(old)==len(mp),
   "outputs/sem_deadline/sem_mapping_deadline_v1.csv")
  add(rows,"development_mapping",track,"mapping_field_equivalence",0,sum(c.values()),"specified field tolerances",sum(c.values())==0,
   "in-memory generalized production reconstruction")
 devmap=pd.concat(dev_maps,ignore_index=True);devman=pd.concat(dev_mans,ignore_index=True)
 dev_ids=sorted(devman[(devman.primary_label_eligible==True)&(devman.sem_input_available==True)].sample_id)
 old_ids=sorted(frozen_man[(frozen_man.primary_label_eligible==True)&(frozen_man.sem_input_available==True)].sample_id)
 dev_matrix=np.stack([dev_summary[x] for x in dev_ids]);dev_up_matrix=np.stack([dev_up[x] for x in dev_ids])
 dev_hash=f64hash(dev_matrix);dev_up_hash=f64hash(dev_up_matrix)
 bounds_cols=["sample_id","central_mask_native_bounds","upstream_native_bounds","downstream_native_bounds"]
 oldb=frozen_man[bounds_cols].sort_values("sample_id").fillna("").reset_index(drop=True)
 newb=devman[bounds_cols].sort_values("sample_id").fillna("").reset_index(drop=True)
 bound_equal=oldb.equals(newb)
 if not bound_equal:
  mismatch_counts["native_bound_difference"]+=int((oldb!=newb).any(axis=1).sum())
 add(rows,"development_mapping","all","total_mapping_rows",1192,len(devmap),"exact",len(devmap)==1192,"in-memory")
 add(rows,"development_cohort","all","cohort_b_sample_ids",678,len(dev_ids),"exact",dev_ids==old_ids,
  "outputs/manifests/sem_height_manifest_deadline_v1_development.csv")
 add(rows,"development_features","all","sem_summary_hash",EXPECTED_DEV_FEATURE,dev_hash,"exact SHA-256",dev_hash==EXPECTED_DEV_FEATURE,"in-memory float64 little-endian")
 add(rows,"development_bounds","all","native_bounds","identical",str(bound_equal),"exact",bound_equal,
  "outputs/manifests/sem_height_manifest_deadline_v1_development.csv")

 # Reconstruct N=14 Track-21 mapping/features.
 labels21=pd.read_csv(ROOT/"outputs/height_labels/height_labels_v1_track21_final_test.csv")
 pos21=labels21[["sample_id","track_id","official_position_index","x_actual_mm","primary_label_eligible"]]
 mp21,mn21,s21,u21,meta21=gen.reconstruct(21,pos21,featmod.representations)
 oldmp21=pd.read_csv(ROOT/"outputs/sem_final_test_v2/sem_mapping_v1_track21.csv")
 c=compare_mapping(oldmp21,mp21,21,mismatches);mismatch_counts.update(c)
 oldmn21=pd.read_csv(ROOT/"outputs/manifests/sem_height_manifest_v1_track21_final_test.csv")
 cmpcols=["sample_id","central_mask_native_bounds","upstream_native_bounds","downstream_native_bounds",
  "sem_input_available","tile_boundary_flag","context_clipped_flag","sem_exclusion_reason"]
 x=oldmn21[cmpcols].sort_values("sample_id").fillna("").reset_index(drop=True)
 y=mn21[cmpcols].sort_values("sample_id").fillna("").reset_index(drop=True)
 man_equal=x.equals(y)
 if not man_equal:
  count=int((x!=y).any(axis=1).sum());mismatch_counts["native_bound_difference"]+=count
  for i in np.flatnonzero((x!=y).any(axis=1).to_numpy())[:max(0,50-len(mismatches))]:
   mismatches.append({"track_id":21,"sample_id":x.iloc[i].sample_id,"official_x":"",
    "field":"manifest_or_native_bounds","old_value":x.iloc[i].to_dict(),"reconstructed_value":y.iloc[i].to_dict(),
    "absolute_difference":"","category":"native_bound_difference"})
 ids21=sorted(mn21[(mn21.primary_label_eligible==True)&(mn21.sem_input_available==True)].sample_id)
 oldids21=sorted(oldmn21[(oldmn21.primary_label_eligible==True)&(oldmn21.sem_input_available==True)].sample_id)
 mat21=np.stack([s21[z] for z in ids21]);upmat21=np.stack([u21[z] for z in ids21])
 hash21=f64hash(mat21);uphash21=f64hash(upmat21)
 add(rows,"track21_mapping",21,"mapping_rows",396,len(mp21),"exact",len(mp21)==396,"outputs/sem_final_test_v2/sem_mapping_v1_track21.csv")
 add(rows,"track21_mapping",21,"mapping_field_equivalence",0,sum(c.values()),"specified field tolerances",sum(c.values())==0,"in-memory")
 add(rows,"track21_manifest",21,"manifest_equivalence","identical",str(man_equal),"exact",man_equal,
  "outputs/manifests/sem_height_manifest_v1_track21_final_test.csv")
 add(rows,"track21_cohort",21,"cohort_b_sample_ids",233,len(ids21),"exact",ids21==oldids21,
  "outputs/manifests/sem_height_manifest_v1_track21_final_test.csv")

 # Recompute fixed Thermal shapes and the three specified predictions in memory.
 thermalmod=mod("scripts/run_thermal_representation_audit.py","fixed_thermal_final_audit")
 raw=np.asarray(loadmat(ROOT/"thermal/Thermal_21.mat")["temperature_data"])
 label_index=labels21.set_index("sample_id");thermal={}
 for sid in ids21:
  frame=np.asarray(raw[int(label_index.loc[sid,"raw_frame_index"])],float)
  success,cx,cy,mask=thermalmod.hotspot(frame)
  thermal[sid]=thermalmod.shape_features(frame,success,cx,cy,mask)
 arts={p.stem:json.loads(p.read_text()) for p in (ROOT/"outputs/models/final_height_width_v1").glob("*.json") if p.name!="feature_schema.json"}
 existing=pd.read_csv(ROOT/"outputs/models/final_track21_height_width_predictions.csv")
 pred_diffs={}
 for name,kind in [("cohort_b_sem_summary_ridge","sem"),("cohort_b_thermal_plus_sem_summary_ridge","multi"),
                   ("cohort_b_sem_summary_upstream_ridge","up")]:
  X=(np.stack([s21[z] for z in ids21]) if kind=="sem" else
     np.stack([u21[z] for z in ids21]) if kind=="up" else
     np.stack([np.r_[s21[z],thermal[z]] for z in ids21]))
  p=predict_art(arts[name],X)
  old=existing[existing.model_id==name].set_index("sample_id").loc[ids21].prediction_mm.to_numpy()
  diff=float(np.max(abs(p-old)));pred_diffs[name]=diff
  if diff>1e-10:
   mismatch_counts["prediction_reproduction_difference"]+=int(np.sum(abs(p-old)>1e-10))
  add(rows,"prediction_reproduction",21,name,"<=1e-10",diff,"1e-10 mm",diff<=1e-10,
   "outputs/models/final_track21_height_width_predictions.csv")

 after={str(p.relative_to(ROOT)):sha(p) for p in paths}
 immutable_ok=before==after
 add(rows,"immutable_integrity","all","before_after_hashes","identical",str(immutable_ok),"exact",immutable_ok,
  "outputs/audits/generalized_sem_mapping_production_equivalence_hashes_v1.json")
 required=(len(devmap)==1192 and dev_ids==old_ids and dev_hash==EXPECTED_DEV_FEATURE and bound_equal and
  sum(mismatch_counts.values())==0 and len(mp21)==396 and ids21==oldids21 and man_equal and
  max(pred_diffs.values())<=1e-10 and immutable_ok)
 status=("GENERALIZED SEM PRODUCTION IMPLEMENTATION EQUIVALENCE PASS" if required else
         "GENERALIZED SEM PRODUCTION IMPLEMENTATION EQUIVALENCE FAIL")
 out=ROOT/"outputs/audits";out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_csv(out/"generalized_sem_mapping_production_equivalence_v1.csv",index=False)
 hashes={"immutable_hashes_before_audit":before,"immutable_hashes_after_audit":after,
  "development_mapping_hashes":{"reconstructed_table_hash":table_hash(devmap),
   "native_mask_bound_hash":table_hash(devman[bounds_cols])},
  "development_feature_hashes":{"sem_summary":dev_hash,"sem_summary_upstream":dev_up_hash,"per_track":dev_track_hashes},
  "track21_reconstructed_mapping_hashes":{"mapping_table":table_hash(mp21),
   "native_mask_bound_hash":table_hash(mn21[bounds_cols])},
  "track21_reconstructed_feature_hashes":{"sem_summary":hash21,"sem_summary_upstream":uphash21},
  "prediction_reproduction_maximum_absolute_differences":pred_diffs,
  "mismatch_counts_by_category":dict(mismatch_counts),"first_50_mismatches":mismatches[:50],
  "overall_status":status}
 (out/"generalized_sem_mapping_production_equivalence_hashes_v1.json").write_text(json.dumps(hashes,indent=2,default=str)+"\n")
 report=f"""# Generalized SEM mapping production-implementation equivalence v1

## Scope distinction

The previous report demonstrated old-pipeline self-reproduction through the development `prepare` path. This audit does not use that path as the generalized implementation. It independently executes the exact production geometry, tile selection, native bounds, masking, and summary extraction used by the completed final runner.

## Development N=13

- Reconstructed mapping rows: {len(devmap)} / 1192
- Cohort-B IDs: {len(dev_ids)} / 678; identical: {dev_ids==old_ids}
- Mapping/bounds mismatch count: {dev_mapping_mismatches + (0 if bound_equal else 1)}
- SEM-summary hash: `{dev_hash}`
- Required frozen hash: `{EXPECTED_DEV_FEATURE}`
- Upstream-summary hash: `{dev_up_hash}`
- Native-bound hash: `{table_hash(devman[bounds_cols])}`
- Per-Track summary hashes: `{json.dumps(dev_track_hashes,sort_keys=True)}`

## Track-21 N=14

- Reconstructed mapping rows: {len(mp21)} / 396
- Cohort-B IDs: {len(ids21)} / 233; identical: {ids21==oldids21}
- Mapping mismatch count: {sum(c.values())}
- Manifest/native bounds identical: {man_equal}
- SEM-summary hash: `{hash21}`
- Upstream-summary hash: `{uphash21}`

## Prediction reproduction

Maximum absolute differences: `{json.dumps(pred_diffs,sort_keys=True)}`. Required tolerance: `1e-10 mm`.

## Immutable integrity and mismatches

- All immutable hashes unchanged: {immutable_ok}
- Mismatch counts: `{json.dumps(dict(mismatch_counts),sort_keys=True)}`
- Existing final interpretation remains unchanged: **SEM IMPROVEMENT NOT CONFIRMED ON FINAL TRACK 21**

{status}
"""
 (ROOT/"outputs/reports/generalized_sem_mapping_production_equivalence_v1.md").write_text(report)
 print(f"development_rows={len(devmap)} development_cohort={len(dev_ids)} development_hash={dev_hash} "
  f"track21_rows={len(mp21)} track21_cohort={len(ids21)} max_prediction_diff={max(pred_diffs.values()):.3g} "
  f"immutable={'PASS' if immutable_ok else 'FAIL'} status={'PASS' if required else 'FAIL'}")
if __name__=="__main__":main()
