"""Read-only generalized SEM production implementation used by equivalence audit."""
from __future__ import annotations
import hashlib, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parents[1]
MASK=.4
CONTEXT=1.0

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _value(text,label):
 m=re.search(rf"{label}\s*=\s*([-+0-9.]+)",text,re.I)
 return float(m.group(1)) if m else np.nan
def discover_geometry(track):
 root=ROOT/"sem"/f"SEM_{track}"/"PlainImages"
 paths=sorted(root.glob("*.tif"),key=lambda p:int(re.search(r"_(\d+)\.tif$",p.name,re.I).group(1)))
 ids=[int(re.search(r"_(\d+)\.tif$",p.name,re.I).group(1)) for p in paths]
 if ids!=list(range(1,len(ids)+1)):raise RuntimeError(f"Noncontiguous canonical tiles: Track {track}: {ids}")
 items=[]
 for p,tile in zip(paths,ids):
  with Image.open(p) as im:
   if im.size!=(1024,768):raise RuntimeError(f"Noncanonical dimensions: {p}")
   text=str(im.tag_v2.get(34118,""))
   # Pixel arrays are loaded only after immutable hashes have been snapshotted.
   arr=np.asarray(ImageOps.grayscale(im))
  items.append({"tile":tile,"path":p.resolve(),"array":arr,
   "px_mm":_value(text,"Image Pixel Size")/1000,"stage_x":_value(text,"Stage at X"),
   "source_hash":sha(p)})
 # Production physical order is N..1; geometry itself uses stage extrema.
 items=list(reversed(items))
 stage=np.array([q["stage_x"] for q in items],float)
 tile_width=float(np.median([q["array"].shape[1]*q["px_mm"] for q in items]))
 slope=((100-tile_width/2)-(20+tile_width/2))/(stage.max()-stage.min())
 intercept=(20+tile_width/2)-slope*stage.min()
 for q in items:
  center=slope*q["stage_x"]+intercept
  q.update(center=center,xmin=center-tile_width/2,xmax=center+tile_width/2)
 return items,float(np.median([q["px_mm"] for q in items])),slope,intercept
def choose(items,x,lo,hi):
 candidates=[]
 for q in items:
  if q["xmin"]<=lo and q["xmax"]>=hi:
   u=(x-q["xmin"])/q["px_mm"]
   candidates.append((min(u,q["array"].shape[1]-1-u),q,u))
 return max(candidates,key=lambda z:z[0]) if candidates else None
def bounds(q,lo,hi,retained):
 a=(lo-q["xmin"])/q["px_mm"];b=(hi-q["xmin"])/q["px_mm"]
 x0=math.ceil(min(a,b)) if retained else math.floor(min(a,b))
 x1=math.floor(max(a,b)) if retained else math.ceil(max(a,b))
 return max(0,int(x0)),min(q["array"].shape[1],int(x1))
def reconstruct(track,positions,representations):
 """Return production mapping/manifest and in-memory retained-context features."""
 items,pixel_mm,slope,intercept=discover_geometry(track)
 mapping=[];manifest=[];summary={};upstream={}
 for r in positions.sort_values("sample_id").itertuples():
  x=float(r.x_actual_mm);mp=choose(items,x,x,x);ctx=choose(items,x,x-MASK-CONTEXT,x+MASK+CONTEXT)
  mosaic=(x-20)/pixel_mm
  if mp is None:
   source="";tile="";local=np.nan;dist=np.nan;map_available=False
   map_reason="official_position_outside_supported_full_sequence"
  else:
   dist,q,local=mp;source=str(q["path"]);tile=q["tile"];map_available=True;map_reason=""
  mapping.append({"sample_id":r.sample_id,"track_id":track,
   "official_position_index":int(r.official_position_index),"x_actual_mm":x,
   "sem_source_path":source,"sem_tile_id":tile,"sem_mosaic_coordinate_px":mosaic,
   "sem_local_coordinate_px":local,"sem_cross_track_reference_px":384 if mp else np.nan,
   "along_track_axis":"horizontal","increasing_x_direction":"left_to_right",
   "sem_mapping_id":"generalized_N_to_1_v1","mapping_source":"generalized production stage-X endpoint-linear",
   "mapping_confidence":"low","mapping_available":map_available,
   "mapping_exclusion_reason":map_reason,"distance_to_image_border_px":dist,
   "distance_to_tile_boundary_px":dist})
  available=mp is not None and ctx is not None
  row={"sample_id":r.sample_id,"track_id":track,"official_position_index":int(r.official_position_index),
   "x_actual_mm":x,"sem_source_path":source,"sem_tile_id":tile,
   "central_mask_native_bounds":"","upstream_native_bounds":"","downstream_native_bounds":"",
   "sem_input_available":available,"sem_exclusion_reason":"" if available else
    (map_reason or "full_2p8mm_context_not_supported_by_one_native_tile"),
   "border_flag":False,"tile_boundary_flag":not available,"context_clipped_flag":not available,
   "upstream_available":available,"downstream_available":available,"valid_pixel_fraction":1.0 if available else 0.0,
   "primary_label_eligible":bool(r.primary_label_eligible)}
  if available:
   cdist,cq,_=ctx
   mask=bounds(cq,x-MASK,x+MASK,False)
   up=bounds(cq,x-MASK-CONTEXT,x-MASK,True)
   down=bounds(cq,x+MASK,x+MASK+CONTEXT,True)
   if not (up[1]<=mask[0] and mask[1]<=down[0]):raise RuntimeError(f"Native overlap: {r.sample_id}")
   us=cq["array"][:,up[0]:up[1]];ds=cq["array"][:,down[0]:down[1]]
   feature,_=representations(us,ds,False,cdist<(MASK+CONTEXT)/cq["px_mm"])
   feature_up,_=representations(us,None,False,cdist<(MASK+CONTEXT)/cq["px_mm"])
   summary[r.sample_id]=np.asarray(feature,float);upstream[r.sample_id]=np.asarray(feature_up,float)
   row.update(central_mask_native_bounds=f"{mask[0]}:{mask[1]}",
    upstream_native_bounds=f"{up[0]}:{up[1]}",downstream_native_bounds=f"{down[0]}:{down[1]}",
    tile_boundary_flag=cdist<(MASK+CONTEXT)/cq["px_mm"])
  manifest.append(row)
 return pd.DataFrame(mapping),pd.DataFrame(manifest),summary,upstream,{"N":len(items),"slope":slope,"intercept":intercept,"pixel_mm":pixel_mm}
