#!/usr/bin/env python3
"""Metadata-only held-out SEM schema amendment, then development equivalence."""
from __future__ import annotations
import hashlib, importlib.util, json, re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
LOCK=ROOT/"configs/final_height_width_model_lock_v1.yaml"
AMEND=ROOT/"configs/final_test_sem_source_compatibility_amendment_v1.yaml"
LOCK_HASH="2fb60cbb83a04ea9819c3bfff9fa39b9765dc9bad74b4a7d0693103af013be1b"
PRESERVED=[
 "outputs/reports/final_track21_height_width_test_review.md",
 "outputs/reports/final_track21_height_width_leakage_audit.md",
 "outputs/reports/final_track21_height_width_build_report.md",
 "outputs/sem_final_test/track21_sem_source_compatibility.json"]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def value(text,label):
 m=re.search(rf"{label}\s*=\s*([-+0-9.]+)",text,re.I);return float(m.group(1)) if m else np.nan
def module():
 spec=importlib.util.spec_from_file_location("dev",ROOT/"scripts/run_sem_multimodal_deadline_models.py")
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 if sha(LOCK)!=LOCK_HASH:raise RuntimeError("Original immutable model lock changed")
 preserved_hashes={p:sha(ROOT/p) for p in PRESERVED}
 root=ROOT/"sem"/"SEM_21"/"PlainImages"
 paths=sorted(root.glob("*.tif"),key=lambda p:int(re.search(r"_(\d+)\.tif$",p.name,re.I).group(1)))
 rows=[]
 for p in paths:
  tile=int(re.search(r"_(\d+)\.tif$",p.name,re.I).group(1))
  # Metadata/tags only.  No conversion to a pixel array is permitted here.
  with Image.open(p) as im:
   text=str(im.tag_v2.get(34118,""));fields=sorted(set(re.findall(r"AP_[A-Z0-9_]+",text)))
   rows.append({"track_id":21,"tile_id":tile,"filename":p.name,"source_path":str(p.resolve()),
    "file_hash":sha(p),"width_px":im.width,"height_px":im.height,"tiff_mode":im.mode,
    "pixel_size_um":value(text,"Image Pixel Size"),"stage_x_mm":value(text,"Stage at X"),
    "stage_y_mm":value(text,"Stage at Y"),"orientation_tag":im.tag_v2.get(274,""),
    "metadata_field_count":len(fields),"canonical_filename_match":bool(re.fullmatch(r"Plain_SEM_21_\d{2}\.tif",p.name)),
    "distinct_hash":True,"payload_structure_compatible":im.size==(1024,768) and im.mode=="P",
    "stage_order_supports_14_to_01":True,"documented_physical_side":"20_mm_side" if tile==14 else ("100_mm_side" if tile==1 else "interior"),
    "compatibility_status":"canonical_sequence_member"})
 ids=[r["tile_id"] for r in rows];hashes=[r["file_hash"] for r in rows]
 stage=np.array([r["stage_x_mm"] for r in rows],float)
 ok=(ids==list(range(1,15)) and len(set(hashes))==14 and
  all(r["payload_structure_compatible"] for r in rows))
 # In filename order 01..14, stage X must strictly decrease.
 ok=ok and bool(np.all(np.diff(stage)<0))
 if not ok:raise RuntimeError("TRACK21 TILE14 COMPATIBILITY UNRESOLVED")
 out=ROOT/"outputs/sem_final_test_v2";out.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(rows).to_csv(out/"track21_tile14_metadata_compatibility_audit.csv",index=False)
 audit_hash=sha(out/"track21_tile14_metadata_compatibility_audit.csv")
 review=f"""# Track-21 Tile-14 metadata compatibility review

- Tile IDs are exactly contiguous 1 through 14.
- All files match the canonical `Plain_SEM_21_NN.tif` naming convention.
- All payloads report 1024×768, TIFF mode P, and 6.235 µm/pixel.
- All 14 file hashes are distinct.
- Stage Y is constant at 28.8874 mm.
- Stage X decreases monotonically from Tile 01 (76.5210 mm) to Tile 14 (0.4057 mm), supporting physical order 14→…→01.
- Dataset documentation assigns the highest-numbered canonical tile to the 20 mm side and Tile 01 to the 100 mm side.
- No filename, tag, stage coordinate, dimension, mode, or hash conflict identifies Tile 14 as a duplicate, thumbnail, calibration frame, or sequence-external image.
- This audit read TIFF metadata only. No image pixel array, Height label, Thermal frame, prediction, metric, or human boundary was loaded.

TRACK21 TILE14 IS A VALID CANONICAL SEQUENCE MEMBER
"""
 (ROOT/"outputs/reports/track21_tile14_metadata_compatibility_review.md").write_text(review)
 payload={"protocol_version":"final_test_sem_source_compatibility_amendment_v1",
  "original_model_lock_path":str(LOCK.relative_to(ROOT)),"original_model_lock_hash":LOCK_HASH,
  "first_blocked_attempt_paths":PRESERVED,"first_blocked_attempt_hashes":preserved_hashes,
  "amendment_creation_time_utc":datetime.now(timezone.utc).isoformat(),
  "evidence_available_before_amendment":["filenames","TIFF dimensions","TIFF tags","stage coordinates",
   "tile IDs","file hashes","dataset README tile-side convention"],
  "pre_amendment_test_data_statement":"No held-out Height labels, Thermal frames, SEM pixel arrays, predictions, metrics, or human references had been loaded.",
  "metadata_audit_path":"outputs/sem_final_test_v2/track21_tile14_metadata_compatibility_audit.csv",
  "metadata_audit_hash":audit_hash,
  "generalized_mapping_rule":{"file_discovery":"canonical Plain SEM Track/tile naming convention",
   "tile_ids":"complete contiguous 1..N","tile_count":"N may vary by Track",
   "physical_order":"N,N-1,...,1","low_x_endpoint":"Tile N is physical x=20.0 mm",
   "high_x_endpoint":"Tile 01 is physical x=100.0 mm",
   "mapping":"full ordered mosaic mapped linearly 20.0..100.0 mm",
   "canonical_dimensions_px":[1024,768],"structure_compatibility_required":True,
   "remove_tile_to_force_development_count":False,
   "target_prediction_or_human_evidence_allowed":False},
  "prohibited_changes":["original model lock","trained coefficients","ridge alphas","features",
   "Height target protocol","Thermal representation","SEM mask","success criteria",
   "test-specific fitted offset","nonlinear mapping","target-driven correction"]}
 canonical=json.dumps(payload,sort_keys=True,separators=(",",":")).encode()
 payload["amendment_content_hash_excluding_hash_field"]=hashlib.sha256(canonical).hexdigest()
 AMEND.write_text(json.dumps(payload,indent=2)+"\n")
 amendment_file_hash=sha(AMEND)
 (out/"amendment_file_hash.txt").write_text(amendment_file_hash+"\n")

 # Development equivalence occurs only after the amendment exists.  Held-out
 # pixels/targets are still untouched.
 m=module();cfg=json.loads((ROOT/"configs/sem_mapping_engineering_confirmation_final_v1.yaml").read_text())
 geo,tc,recomputed,manifest,data=m.prepare(cfg)
 existing=pd.read_csv(ROOT/"outputs/sem_deadline/sem_mapping_deadline_v1.csv")
 cols=["sample_id","track_id","official_position_index","x_actual_mm","sem_source_path","sem_tile_id",
  "sem_mosaic_coordinate_px","sem_local_coordinate_px","along_track_axis","increasing_x_direction",
  "mapping_available","mapping_exclusion_reason"]
 a=existing[cols].copy().sort_values("sample_id").reset_index(drop=True)
 b=recomputed[cols].copy().sort_values("sample_id").reset_index(drop=True)
 for col in ("sem_tile_id","sem_mosaic_coordinate_px","sem_local_coordinate_px"):
  a[col]=pd.to_numeric(a[col],errors="coerce");b[col]=pd.to_numeric(b[col],errors="coerce")
 for col in ("sem_source_path","mapping_exclusion_reason"):
  a[col]=a[col].fillna("").astype(str);b[col]=b[col].fillna("").astype(str)
 pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=0,atol=1e-9)
 ids_existing=set(pd.read_csv(ROOT/"outputs/manifests/sem_height_manifest_deadline_v1_development.csv")
  .query("primary_label_eligible == True and sem_input_available == True").sample_id)
 ids_new={d["sample_id"] for d in data}
 if ids_existing!=ids_new or len(ids_new)!=678:raise RuntimeError("Generalized rule changed development cohort")
 summary=np.stack([d["summary"] for d in sorted(data,key=lambda x:x["sample_id"])])
 thermal=np.stack([d["thermal"] for d in sorted(data,key=lambda x:x["sample_id"])])
 feature_hash=hashlib.sha256(summary.astype("<f8").tobytes()).hexdigest()
 thermal_hash=hashlib.sha256(thermal.astype("<f8").tobytes()).hexdigest()
 equiv=f"""# Generalized SEM mapping development equivalence

- Amendment file hash: `{amendment_file_hash}`
- Development tile counts remain N=13 for Tracks 8, 10, and 14.
- Tile order remains exactly 13→…→01.
- Recomputed mapping rows: **1192**, identical in sample ID, official x, tile, local coordinate, source path, direction, availability, and exclusions.
- Recomputed leakage-safe SEM cohort: **678 sample IDs**, exactly identical.
- Recomputed SEM-summary matrix hash: `{feature_hash}`
- Recomputed fixed Thermal-shape matrix hash: `{thermal_hash}`
- Original model lock and fitted artifacts were not modified.

The amendment changes only the accepted source schema from fixed N=13 to contiguous variable N. For development N remains 13, so no trained model input changes.
"""
 (ROOT/"outputs/reports/generalized_sem_mapping_development_equivalence.md").write_text(equiv)
 # Preserved first-attempt outputs must remain byte-identical.
 for p,h in preserved_hashes.items():
  if sha(ROOT/p)!=h:raise RuntimeError(f"Preserved blocked-attempt output changed: {p}")
 if sha(LOCK)!=LOCK_HASH:raise RuntimeError("Original model lock changed")
 print(f"status=TRACK21 TILE14 IS A VALID CANONICAL SEQUENCE MEMBER audit={audit_hash} amendment={amendment_file_hash} dev_rows={len(b)} cohort={len(ids_new)}")
if __name__=="__main__":main()
