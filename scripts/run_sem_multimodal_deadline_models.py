#!/usr/bin/env python3
"""Confirmed full-sequence SEM mapping and deadline LOTO linear models."""
from __future__ import annotations
import hashlib, importlib.util, json, math, os, re
from pathlib import Path

import cv2
os.environ.setdefault("MPLCONFIGDIR","/private/tmp/nsf_mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr

ROOT=Path(__file__).resolve().parents[1]
TRACKS=(8,10,14)
ALPHAS=(.1,1.,10.,100.,1000.)
PCS=(10,20,40,80)
CONF=ROOT/"configs/sem_mapping_engineering_confirmation_v1.yaml"
LABELS=ROOT/"outputs/height_labels/height_labels_v1_development.csv"
MASK_HALF=.40
CONTEXT=1.00

def fail_unconfirmed():
    if not CONF.exists(): raise RuntimeError(f"Missing engineering confirmation: {CONF}")
    try: cfg=json.loads(CONF.read_text())
    except Exception as e: raise RuntimeError(f"Confirmation file must remain JSON-compatible YAML: {e}")
    issues=[]
    for t in TRACKS:
        c=cfg.get("tracks",{}).get(str(t),{})
        if c.get("confirmed") is not True: issues.append(f"Track {t}: confirmed is not true")
        for key in ("along_track_axis","increasing_x_direction","tile_order","mapping_type","mapping_source"):
            v=c.get(key)
            if v in (None,"",[],"unresolved"): issues.append(f"Track {t}: {key} unresolved")
        if c.get("along_track_axis") not in ("horizontal","vertical"): issues.append(f"Track {t}: invalid along_track_axis")
        valid_dirs=("left_to_right","right_to_left") if c.get("along_track_axis")=="horizontal" else ("top_to_bottom","bottom_to_top")
        if c.get("increasing_x_direction") not in valid_dirs: issues.append(f"Track {t}: direction inconsistent with axis")
        if sorted(c.get("tile_order",[]))!=list(range(1,14)): issues.append(f"Track {t}: tile_order must contain 1..13 exactly")
    if issues: raise RuntimeError("SEM engineering confirmation incomplete; training blocked:\n- "+"\n- ".join(issues))
    return cfg

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def meta(p):
    with Image.open(p) as im:
        a=np.asarray(ImageOps.grayscale(im));txt=str(im.tag_v2.get(34118,""))
    def n(label):
        m=re.search(rf"{label}\s*=\s*([-+0-9.]+)",txt,re.I);return float(m.group(1)) if m else np.nan
    return a,n("Image Pixel Size")/1000,n("Stage at X"),n("Stage at Y")
def tile_path(t,tile):
    pp=list((ROOT/"sem"/f"SEM_{t}"/"PlainImages").glob(f"*_{tile:02d}.tif"))
    if len(pp)!=1:raise RuntimeError(f"Canonical tile not unique: Track {t} tile {tile}")
    return pp[0]

def build_geometry(cfg):
    geo={};track_cfg={}
    for t in TRACKS:
        c=cfg["tracks"][str(t)];items=[]
        for tile in c["tile_order"]:
            p=tile_path(t,tile);a,px,sx,sy=meta(p)
            items.append({"tile":tile,"path":p,"array":a,"px_mm":px,"stage_x":sx,"stage_y":sy})
        stage=np.array([q["stage_x"] for q in items]);fw=np.median([q["array"].shape[1]*q["px_mm"] for q in items])
        # Whole-sequence endpoint-linear fallback: extreme tile centres are
        # anchored once; no individual tile receives the full interval.
        slope=((100-fw/2)-(20+fw/2))/(stage.max()-stage.min());intercept=(20+fw/2)-slope*stage.min()
        for q in items:
            center=slope*q["stage_x"]+intercept;q["xmin"]=center-fw/2;q["xmax"]=center+fw/2
            q["source_hash"]=sha(q["path"]);geo[(t,q["tile"])]=q
        track_cfg[t]={"axis":c["along_track_axis"],"direction":c["increasing_x_direction"],
          "order":c["tile_order"],"mapping_source":c["mapping_source"],"confidence":c["confidence"],
          "slope":slope,"intercept":intercept,"field_mm":fw,
          "pixel_mm":float(np.median([q["px_mm"] for q in items]))}
    return geo,track_cfg

def local_for_x(q,axis,direction,x):
    size=q["array"].shape[1] if axis=="horizontal" else q["array"].shape[0]
    increasing=direction in ("left_to_right","top_to_bottom")
    u=(x-q["xmin"])/q["px_mm"] if increasing else (q["xmax"]-x)/q["px_mm"]
    return u,size,increasing
def interval_px(q,axis,direction,lo,hi,retained_context=False):
    a,_,_=local_for_x(q,axis,direction,lo);b,_,_=local_for_x(q,axis,direction,hi)
    lower=(math.ceil(min(a,b)) if retained_context else math.floor(min(a,b)))
    upper=(math.floor(max(a,b)) if retained_context else math.ceil(max(a,b)))
    return max(0,int(lower)),min(q["array"].shape[1] if axis=="horizontal" else q["array"].shape[0],int(upper))
def choose_tile(geo,tc,t,x,need_lo,need_hi):
    candidates=[]
    for tile in tc[t]["order"]:
        q=geo[(t,tile)]
        if q["xmin"]<=need_lo and q["xmax"]>=need_hi:
            u,size,_=local_for_x(q,tc[t]["axis"],tc[t]["direction"],x)
            candidates.append((min(u,size-1-u),q,u,size))
    return max(candidates,key=lambda z:z[0]) if candidates else None

def strips(q,axis,direction,x,protocol):
    mask=interval_px(q,axis,direction,x-MASK_HALF,x+MASK_HALF)
    up=interval_px(q,axis,direction,x-MASK_HALF-CONTEXT,x-MASK_HALF,retained_context=True)
    down=interval_px(q,axis,direction,x+MASK_HALF,x+MASK_HALF+CONTEXT,retained_context=True)
    def cut(bounds):
        a,b=bounds
        return q["array"][:,a:b] if axis=="horizontal" else q["array"][a:b,:]
    us=cut(up);ds=cut(down)
    return us,(ds if protocol=="symmetric" else None),mask,up,down
def stats(a):
    x=a.astype(float)/255;gx=np.diff(x,axis=1) if x.shape[1]>1 else np.diff(x,axis=0)
    lap=cv2.Laplacian(x.astype(np.float32),cv2.CV_32F)
    hist=np.histogram(x,bins=32,range=(0,1),density=True)[0];p=hist/max(hist.sum(),1);p=p[p>0]
    return [x.mean(),x.std(),np.median(x),*np.percentile(x,[5,25,75,95]),
      np.percentile(x,95)-np.percentile(x,5),np.mean(gx*gx),np.mean(np.abs(gx)>.08),
      np.var(lap),np.mean(np.abs(x-cv2.GaussianBlur(x.astype(np.float32),(0,0),3))),
      -np.sum(p*np.log2(p)),np.mean(x*x)]
def representations(us,ds,border,boundary):
    su=stats(us)
    if ds is None: sd=[0.]*len(su);diff=[0.,0.]
    else: sd=stats(ds);diff=[su[0]-sd[0],su[8]-sd[8]]
    summary=np.array(su+sd+diff+[1.,float(border),float(boundary)],float)
    def small(a):return cv2.resize(a,(16,16),interpolation=cv2.INTER_AREA).astype(float).ravel()/255
    raw=np.concatenate([small(us),small(ds) if ds is not None else np.zeros(256)])
    return summary,raw

def thermal_shape(labels):
    spec=importlib.util.spec_from_file_location("tra",ROOT/"scripts/run_thermal_representation_audit.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    out={}
    for t in TRACKS:
        frames=np.asarray(loadmat(ROOT/"thermal"/f"Thermal_{t}.mat")["temperature_data"])
        for r in labels[labels.track_id==t].itertuples():
            frame=np.asarray(frames[int(r.raw_frame_index)],float)
            success,cx,cy,mask=mod.hotspot(frame)
            out[r.sample_id]=mod.shape_features(frame,success,cx,cy,mask)
    return out

def prepare(cfg):
    geo,tc=build_geometry(cfg)
    labels=pd.read_csv(LABELS)
    labels=labels[labels.track_id.isin(TRACKS)].copy()
    if len(labels)!=1192 or labels.sample_id.duplicated().any():raise RuntimeError("Frozen development label identity failure")
    mapping=[];manifest=[];summaries={};raws={};up_summaries={};up_raws={}
    for r in labels.itertuples():
        need_lo=r.x_actual_mm-MASK_HALF-CONTEXT;need_hi=r.x_actual_mm+MASK_HALF+CONTEXT
        map_selected=choose_tile(geo,tc,r.track_id,r.x_actual_mm,r.x_actual_mm,r.x_actual_mm)
        selected=choose_tile(geo,tc,r.track_id,r.x_actual_mm,need_lo,need_hi)
        mosaic_px=(r.x_actual_mm-20)/tc[r.track_id]["pixel_mm"]
        if map_selected is None:
            mapping.append({"sample_id":r.sample_id,"track_id":r.track_id,"official_position_index":r.official_position_index,
              "x_actual_mm":r.x_actual_mm,"sem_source_path":"","sem_tile_id":"","sem_mosaic_coordinate_px":mosaic_px,
              "sem_local_coordinate_px":"","sem_cross_track_reference_px":"","along_track_axis":tc[r.track_id]["axis"],
              "increasing_x_direction":tc[r.track_id]["direction"],"sem_mapping_id":"sem_mapping_deadline_v1",
              "mapping_source":tc[r.track_id]["mapping_source"],"mapping_confidence":tc[r.track_id]["confidence"],
              "mapping_available":False,"mapping_exclusion_reason":"official_position_outside_supported_full_sequence",
              "distance_to_image_border_px":"","distance_to_tile_boundary_px":""})
            manifest.append({"sample_id":r.sample_id,"track_id":r.track_id,"segment_frame_index":r.segment_frame_index,
              "raw_frame_index":r.raw_frame_index,"official_position_index":r.official_position_index,
              "x_actual_mm":r.x_actual_mm,"target_width_mm":r.target_width_mm,
              "primary_label_eligible":r.primary_label_eligible,"sem_mapping_id":"sem_mapping_deadline_v1",
              "sem_protocol_id":"symmetric_two_strip_1mm_mask0p4mm_v1","sem_source_path":"",
              "sem_tile_id":"","central_mask_native_bounds":"","upstream_native_bounds":"",
              "downstream_native_bounds":"","sem_input_available":False,
              "sem_exclusion_reason":"official_position_outside_supported_full_sequence",
              "border_flag":False,"tile_boundary_flag":True,"context_clipped_flag":True,
              "upstream_available":False,"downstream_available":False,"valid_pixel_fraction":0.0})
            continue
        map_dist,map_q,map_u,map_size=map_selected;axis=tc[r.track_id]["axis"];direction=tc[r.track_id]["direction"]
        mapping.append({"sample_id":r.sample_id,"track_id":r.track_id,"official_position_index":r.official_position_index,
          "x_actual_mm":r.x_actual_mm,"sem_source_path":str(map_q["path"].resolve()),"sem_tile_id":map_q["tile"],
          "sem_mosaic_coordinate_px":mosaic_px,"sem_local_coordinate_px":map_u,
          "sem_cross_track_reference_px":map_q["array"].shape[0]/2 if axis=="horizontal" else map_q["array"].shape[1]/2,
          "along_track_axis":axis,"increasing_x_direction":direction,"sem_mapping_id":"sem_mapping_deadline_v1",
          "mapping_source":tc[r.track_id]["mapping_source"],"mapping_confidence":tc[r.track_id]["confidence"],
          "mapping_available":True,"mapping_exclusion_reason":"","distance_to_image_border_px":map_dist,
          "distance_to_tile_boundary_px":map_dist})
        if selected is None:
            manifest.append({"sample_id":r.sample_id,"track_id":r.track_id,"segment_frame_index":r.segment_frame_index,
              "raw_frame_index":r.raw_frame_index,"official_position_index":r.official_position_index,
              "x_actual_mm":r.x_actual_mm,"target_width_mm":r.target_width_mm,
              "primary_label_eligible":r.primary_label_eligible,"sem_mapping_id":"sem_mapping_deadline_v1",
              "sem_protocol_id":"symmetric_two_strip_1mm_mask0p4mm_v1","sem_source_path":str(map_q["path"].resolve()),
              "sem_tile_id":map_q["tile"],"central_mask_native_bounds":"","upstream_native_bounds":"",
              "downstream_native_bounds":"","sem_input_available":False,
              "sem_exclusion_reason":"full_2p8mm_context_not_supported_by_one_native_tile",
              "border_flag":False,"tile_boundary_flag":True,"context_clipped_flag":True,
              "upstream_available":False,"downstream_available":False,"valid_pixel_fraction":0.0})
            continue
        dist,q,u,size=selected
        us,ds,mask,up,down=strips(q,axis,direction,r.x_actual_mm,"symmetric")
        if min(us.shape)<=0 or min(ds.shape)<=0:raise RuntimeError(f"Empty context: {r.sample_id}")
        if not (max(up)<=min(mask) or max(mask)<=min(up)):raise RuntimeError("Upstream overlaps mask")
        if not (max(down)<=min(mask) or max(mask)<=min(down)):raise RuntimeError("Downstream overlaps mask")
        boundary=dist<(CONTEXT+MASK_HALF)/q["px_mm"]
        summaries[r.sample_id],raws[r.sample_id]=representations(us,ds,False,boundary)
        up_summaries[r.sample_id],up_raws[r.sample_id]=representations(us,None,False,boundary)
        manifest.append({"sample_id":r.sample_id,"track_id":r.track_id,"segment_frame_index":r.segment_frame_index,
          "raw_frame_index":r.raw_frame_index,"official_position_index":r.official_position_index,
          "x_actual_mm":r.x_actual_mm,"target_width_mm":r.target_width_mm,
          "primary_label_eligible":r.primary_label_eligible,"sem_mapping_id":"sem_mapping_deadline_v1",
          "sem_protocol_id":"symmetric_two_strip_1mm_mask0p4mm_v1","sem_source_path":str(q["path"].resolve()),
          "sem_tile_id":q["tile"],"central_mask_native_bounds":f"{mask[0]}:{mask[1]}",
          "upstream_native_bounds":f"{up[0]}:{up[1]}","downstream_native_bounds":f"{down[0]}:{down[1]}",
          "sem_input_available":True,"sem_exclusion_reason":"","border_flag":False,
          "tile_boundary_flag":boundary,"context_clipped_flag":False,
          "upstream_available":True,"downstream_available":True,"valid_pixel_fraction":1.0})
    mapdf=pd.DataFrame(mapping);mandf=pd.DataFrame(manifest)
    if mapdf[mapdf.mapping_available].groupby("track_id").x_actual_mm.apply(lambda x:x.is_monotonic_increasing).eq(False).any():
        raise RuntimeError("Official x projection is not monotonic")
    eligible=mandf[(mandf.primary_label_eligible==True)&(mandf.sem_input_available==True)&mandf.target_width_mm.notna()].copy()
    therm=thermal_shape(eligible)
    data=[]
    for r in eligible.itertuples():
        data.append({"sample_id":r.sample_id,"track_id":r.track_id,"x":r.x_actual_mm,"y":r.target_width_mm,
          "summary":summaries[r.sample_id],"raw":raws[r.sample_id],"up_summary":up_summaries[r.sample_id],
          "up_raw":up_raws[r.sample_id],"thermal":therm[r.sample_id]})
    return geo,tc,mapdf,mandf,data

def metric(y,p):
    e=p-y;ae=np.abs(e)
    def corr(fun):
        try:return float(fun(y,p).statistic)
        except Exception:return np.nan
    return {"sample_count":len(y),"mae":ae.mean(),"rmse":np.sqrt(np.mean(e*e)),
      "median_absolute_error":np.median(ae),"p90_absolute_error":np.percentile(ae,90),
      "maximum_absolute_error":ae.max(),"signed_bias":e.mean(),
      "r_squared":1-np.sum(e*e)/np.sum((y-y.mean())**2),
      "pearson_correlation":corr(pearsonr),"spearman_correlation":corr(spearmanr)}
def scale_fit(x):
    med=np.nanmedian(x,0);x=np.where(np.isfinite(x),x,med);mean=x.mean(0);std=x.std(0);std[std<1e-12]=1
    return med,mean,std
def scale(x,state):med,mean,std=state;return (np.where(np.isfinite(x),x,med)-mean)/std
def xmat(x):x=np.asarray(x);return np.column_stack([x,x*x])
def matrix(data,key,ids):return np.stack([data[i][key] for i in ids])
def pca_fit_transform(x,n):
    _,_,vt=np.linalg.svd(x,full_matrices=False);components=vt[:n]
    return x@components.T,components
def ridge_predict(x,y,xv,alpha):
    design=np.column_stack([np.ones(len(x)),x]);dv=np.column_stack([np.ones(len(xv)),xv])
    penalty=np.eye(design.shape[1]);penalty[0,0]=0
    coef=np.linalg.solve(design.T@design+alpha*penalty,design.T@y)
    return dv@coef

def select_alpha(data,key,train_tracks,pca_n=None,extra_key=None):
    scores={a:[] for a in ALPHAS}
    for va in train_tracks:
        tr=[i for i,d in enumerate(data) if d["track_id"] in train_tracks and d["track_id"]!=va]
        vv=[i for i,d in enumerate(data) if d["track_id"]==va]
        xtr=matrix(data,key,tr);xv=matrix(data,key,vv);y=np.array([data[i]["y"] for i in tr]);yv=np.array([data[i]["y"] for i in vv])
        st=scale_fit(xtr);xtr=scale(xtr,st);xv=scale(xv,st)
        if pca_n:
            n=min(pca_n,xtr.shape[0]-1,xtr.shape[1]);xtr,pc=pca_fit_transform(xtr,n);xv=xv@pc.T
        if extra_key is not None:
            etr=matrix(data,extra_key,tr);ev=matrix(data,extra_key,vv);est=scale_fit(etr)
            xtr=np.column_stack([xtr,scale(etr,est)]);xv=np.column_stack([xv,scale(ev,est)])
        for a in ALPHAS:scores[a].append(np.mean(np.abs(ridge_predict(xtr,y,xv,a)-yv)))
    return min(ALPHAS,key=lambda a:(np.mean(scores[a]),a)),scores
def fit_model(data,key,tr,va,alpha,pca_n=None,extra_key=None):
    xtr=matrix(data,key,tr);xv=matrix(data,key,va)
    st=scale_fit(xtr);xtr=scale(xtr,st);xv=scale(xv,st);actual_pc=None
    if pca_n:
        actual_pc=min(pca_n,xtr.shape[0]-1,xtr.shape[1]);xtr,pc=pca_fit_transform(xtr,actual_pc);xv=xv@pc.T
    if extra_key is not None:
        etr=matrix(data,extra_key,tr);ev=matrix(data,extra_key,va);est=scale_fit(etr)
        xtr=np.column_stack([xtr,scale(etr,est)]);xv=np.column_stack([xv,scale(ev,est)])
    y=np.array([data[i]["y"] for i in tr])
    return ridge_predict(xtr,y,xv,alpha),actual_pc

def models(data):
    grid=[];folds=[];preds=[]
    specs=[("fixed_thermal_shape_ridge","thermal",None,None),
      ("sem_summary_ridge","summary",None,None),("sem_pca_ridge","raw","pca",None),
      ("thermal_plus_sem_summary_ridge","summary",None,"thermal"),
      ("thermal_plus_sem_pca_ridge","raw","pca","thermal"),
      ("sem_summary_upstream_ridge","up_summary",None,None),
      ("sem_pca_upstream_ridge","up_raw","pca",None)]
    for held in TRACKS:
        train_tracks=[t for t in TRACKS if t!=held]
        tr=[i for i,d in enumerate(data) if d["track_id"] in train_tracks];va=[i for i,d in enumerate(data) if d["track_id"]==held]
        yv=np.array([data[i]["y"] for i in va]);xtr=np.array([data[i]["x"] for i in tr]);xv=np.array([data[i]["x"] for i in va])
        base={"training_mean":np.repeat(np.mean([data[i]["y"] for i in tr]),len(va))}
        # x-only alpha chosen by complete-track inner validation.
        temp=[dict(d,xfeat=xmat([d["x"]])[0]) for d in data]
        a,sc=select_alpha(temp,"xfeat",train_tracks);base["x_only"],_=fit_model(temp,"xfeat",tr,va,a)
        grid += [{"held_out_track":held,"model_id":"x_only","alpha":aa,"pca_components":"","inner_mean_mae":np.mean(v)} for aa,v in sc.items()]
        for name,key,ptype,extra in specs:
            choices=PCS if ptype else (None,);best=None
            for pc in choices:
                aa,scores=select_alpha(data,key,train_tracks,pc,extra)
                score=np.mean(scores[aa]);grid.append({"held_out_track":held,"model_id":name,"alpha":aa,"pca_components":pc or "","inner_mean_mae":score})
                if best is None or score<best[0]:best=(score,aa,pc)
            _,aa,pc=best;base[name],actual=fit_model(data,key,tr,va,aa,pc,extra)
        for name,p in base.items():
            m=metric(yv,p);folds.append({"held_out_track":held,"model_id":name,**m})
            for i,pr in zip(va,p):preds.append({"sample_id":data[i]["sample_id"],"track_id":held,"x_actual_mm":data[i]["x"],
              "target_width_mm":data[i]["y"],"model_id":name,"prediction_mm":pr,"absolute_error_mm":abs(pr-data[i]["y"])})
    f=pd.DataFrame(folds);x=f[f.model_id=="x_only"][["held_out_track","mae"]].rename(columns={"mae":"x_only_mae"})
    f=f.merge(x,on="held_out_track");f["absolute_mae_change_vs_x_only"]=f.mae-f.x_only_mae;f["relative_mae_change_vs_x_only"]=f.absolute_mae_change_vs_x_only/f.x_only_mae
    f["corresponding_unimodal_model"]=""
    f["absolute_mae_change_vs_corresponding_unimodal"]=np.nan
    f["relative_mae_change_vs_corresponding_unimodal"]=np.nan
    corresponding={"thermal_plus_sem_summary_ridge":"sem_summary_ridge",
                   "thermal_plus_sem_pca_ridge":"sem_pca_ridge"}
    for model,uni in corresponding.items():
        ref=f[f.model_id==uni][["held_out_track","mae"]].set_index("held_out_track").mae
        mask=f.model_id==model
        f.loc[mask,"corresponding_unimodal_model"]=uni
        vals=f.loc[mask,"held_out_track"].map(ref)
        f.loc[mask,"absolute_mae_change_vs_corresponding_unimodal"]=f.loc[mask,"mae"].to_numpy()-vals.to_numpy()
        f.loc[mask,"relative_mae_change_vs_corresponding_unimodal"]=f.loc[mask,"absolute_mae_change_vs_corresponding_unimodal"].to_numpy()/vals.to_numpy()
    agg=[]
    for name,g in pd.DataFrame(preds).groupby("model_id"):
        mm=metric(g.target_width_mm.to_numpy(),g.prediction_mm.to_numpy())
        fg=f[f.model_id==name];agg.append({"model_id":name,"aggregation":"pooled",**mm,
          "unweighted_track_mae":fg.mae.mean(),"improves_all_three_tracks":bool((fg.absolute_mae_change_vs_x_only<0).all()),
          "improves_corresponding_unimodal_all_three_tracks":bool(fg.absolute_mae_change_vs_corresponding_unimodal.notna().all() and (fg.absolute_mae_change_vs_corresponding_unimodal<0).all())})
    return pd.DataFrame(grid),f,pd.DataFrame(agg),pd.DataFrame(preds)

def save_qa(geo,tc,mandf):
    d=ROOT/"outputs/figures/sem_input_deadline_qa";d.mkdir(parents=True,exist_ok=True)
    for t in TRACKS:
        g=mandf[(mandf.track_id==t)&(mandf.sem_input_available==True)].sort_values("x_actual_mm")
        selections=[("beginning",g.iloc[0]),("middle",g.iloc[len(g)//2]),("end",g.iloc[-1])]
        boundary=g.sort_values("tile_boundary_flag",ascending=False).iloc[0];selections.append(("near_tile_boundary",boundary))
        for label,r in selections:
            q=geo[(t,int(r.sem_tile_id))];axis=tc[t]["axis"];direction=tc[t]["direction"]
            us,ds,mask,up,down=strips(q,axis,direction,r.x_actual_mm,"symmetric")
            fig,axs=plt.subplots(1,3,figsize=(11,3))
            axs[0].imshow(q["array"],cmap="gray");lo,hi=mask
            if axis=="horizontal":axs[0].axvspan(lo,hi,color="red",alpha=.45)
            else:axs[0].axhspan(lo,hi,color="red",alpha=.45)
            axs[0].set_title(f"native tile; target x={r.x_actual_mm:.1f}")
            axs[1].imshow(np.concatenate([cv2.resize(us,(64,64)),cv2.resize(ds,(64,64))],1),cmap="gray");axs[1].set_title("symmetric retained input")
            axs[2].imshow(cv2.resize(us,(64,64)),cmap="gray");axs[2].set_title("upstream-only sensitivity")
            for ax in axs:ax.axis("off")
            fig.suptitle(f"Track {t} {label}; mask {mask}; upstream {up}; downstream {down}; {direction}")
            fig.tight_layout();fig.savefig(d/f"track_{t}_{label}.png",dpi=160);plt.close(fig)

def write_outputs(cfg,geo,tc,mapdf,mandf,grid,folds,agg,preds):
    for p in ["outputs/sem_deadline","outputs/manifests","outputs/models","outputs/reports","outputs/figures/sem_multimodal_deadline_results"]:
        (ROOT/p).mkdir(parents=True,exist_ok=True)
    mapdf.to_csv(ROOT/"outputs/sem_deadline/sem_mapping_deadline_v1.csv",index=False)
    mandf.to_csv(ROOT/"outputs/manifests/sem_height_manifest_deadline_v1_development.csv",index=False)
    grid.to_csv(ROOT/"outputs/models/sem_multimodal_deadline_model_grid.csv",index=False)
    folds.to_csv(ROOT/"outputs/models/sem_multimodal_deadline_fold_metrics.csv",index=False)
    agg.to_csv(ROOT/"outputs/models/sem_multimodal_deadline_aggregate_metrics.csv",index=False)
    preds.to_csv(ROOT/"outputs/models/sem_multimodal_deadline_predictions.csv",index=False)
    sens=folds[folds.model_id.isin(["sem_summary_ridge","sem_pca_ridge","sem_summary_upstream_ridge","sem_pca_upstream_ridge"])].copy()
    sens["protocol_id"]=np.where(sens.model_id.str.contains("upstream"),"upstream_only_1mm_mask0p4mm_v1","symmetric_two_strip_1mm_mask0p4mm_v1")
    sens.to_csv(ROOT/"outputs/models/sem_multimodal_deadline_sensitivity_metrics.csv",index=False)
    mapping_cfg={"protocol_version":"sem_mapping_deadline_v1","confirmation_hash":sha(CONF),
      "mapping_domain":"full ordered tile sequence","tracks":cfg["tracks"],"silent_flip":False}
    (ROOT/"configs/sem_mapping_deadline_v1.yaml").write_text(json.dumps(mapping_cfg,indent=2)+"\n")
    mask={"protocol_version":"sem_mask_protocol_deadline_v1","primary_protocol_id":"symmetric_two_strip_1mm_mask0p4mm_v1",
      "sensitivity_protocol_id":"upstream_only_1mm_mask0p4mm_v1","units":"millimeters",
      "central_exclusion_half_width_mm":.4,"upstream_context_mm":1.0,"downstream_context_mm":1.0,
      "mask_in_native_coordinates_before_resize":True,"inpainting":False,"target_pixels_in_context":False}
    (ROOT/"configs/sem_mask_protocol_deadline_v1.yaml").write_text(json.dumps(mask,indent=2)+"\n")
    a=agg.set_index("model_id");f=folds
    sem=min(a.loc["sem_summary_ridge","unweighted_track_mae"],a.loc["sem_pca_ridge","unweighted_track_mae"])
    multi=min([a.loc[x,"unweighted_track_mae"] for x in ("thermal_plus_sem_summary_ridge","thermal_plus_sem_pca_ridge")])
    x=a.loc["x_only","unweighted_track_mae"]
    sem_cons=any(bool(a.loc[m,"improves_all_three_tracks"]) for m in ("sem_summary_ridge","sem_pca_ridge"))
    multi_cons=any(bool(a.loc[m,"improves_all_three_tracks"]) and bool(a.loc[m,"improves_corresponding_unimodal_all_three_tracks"]) for m in ("thermal_plus_sem_summary_ridge","thermal_plus_sem_pca_ridge"))
    if sem<x and sem_cons:status="SEM SIGNAL ESTABLISHED UNDER COMPLETE-TRACK HOLDOUT"
    elif multi<min(sem,a.loc["fixed_thermal_shape_ridge","unweighted_track_mae"]) and multi_cons:status="THERMAL PLUS SEM IMPROVES OVER UNIMODAL BASELINES"
    else:status="SEM AND MULTIMODAL SIGNAL NOT ESTABLISHED ACROSS DEVELOPMENT TRACKS"
    cols=["model_id","sample_count","mae","rmse","unweighted_track_mae","improves_all_three_tracks"]
    md=["| "+" | ".join(cols)+" |","|"+"|".join(["---"]*len(cols))+"|"]
    for _,r in agg.iterrows():
        md.append("| "+" | ".join(str(r[c]) for c in cols)+" |")
    review=["# SEM multimodal deadline review","",
      f"- Primary-label eligible rows: {len(mandf[mandf.primary_label_eligible==True])}.",
      f"- Leakage-safe SEM-input eligible rows used by SEM comparisons: {len(mandf[(mandf.primary_label_eligible==True)&(mandf.sem_input_available==True)])}.",
      "- Validation: three complete-Track outer folds; complete-Track inner selection only.",
      "- Thermal input: fixed `thermal_shape_features`; no Thermal reselection.",
      "- SEM center: masked ±0.40 mm in native coordinates before resizing/features.","",
      "## Aggregate metrics","",*md,"",status]
    (ROOT/"outputs/reports/sem_multimodal_deadline_review.md").write_text("\n".join(review)+"\n")
    leakage="""# SEM multimodal deadline leakage audit

- Only development Tracks 8, 10, and 14 were loaded.
- Mapping engineering confirmation was validated before labels or images entered training.
- Mapping used README/TIFF stage evidence and full ordered tile sequences, never Height labels.
- Central target pixels were excluded in native coordinates before resize or feature extraction.
- No Height-derived feature, human boundary, or Track ID entered model matrices.
- No random position split was used; all outer and inner splits held out complete Tracks.
- Imputation, scaling, PCA, and ridge fitting were training-fold-only.
- Thermal used only the previously fixed descriptive `thermal_shape_features`.
- No final-test predictions were created.
"""
    (ROOT/"outputs/reports/sem_multimodal_deadline_leakage_audit.md").write_text(leakage)
    build=f"""# SEM multimodal deadline build report

- Full-sequence mapping rows: {len(mapdf)}
- SEM manifest rows: {len(mandf)}
- Model-grid rows: {len(grid)}
- Fold-metric rows: {len(folds)}
- Prediction rows: {len(preds)}
- Required linear pipeline completed; optional CNN not run.
- Confirmation hash: `{sha(CONF)}`
"""
    (ROOT/"outputs/reports/sem_multimodal_deadline_build_report.md").write_text(build)
    fig,ax=plt.subplots(figsize=(9,4));z=agg.sort_values("unweighted_track_mae")
    ax.bar(z.model_id,z.unweighted_track_mae);ax.tick_params(axis="x",rotation=45);ax.set_ylabel("unweighted Track MAE (mm)");fig.tight_layout()
    fig.savefig(ROOT/"outputs/figures/sem_multimodal_deadline_results/model_mae.png",dpi=160);plt.close(fig)

def main():
    cfg=fail_unconfirmed()  # must be first: no labels/images/models before confirmation
    geo,tc,mapdf,mandf,data=prepare(cfg)
    grid,folds,agg,preds=models(data)
    save_qa(geo,tc,mandf);write_outputs(cfg,geo,tc,mapdf,mandf,grid,folds,agg,preds)
    print(f"mapped={mapdf.mapping_available.sum()} manifest={len(mandf)} eligible={len(data)} folds={len(folds)} predictions={len(preds)}")
if __name__=="__main__":main()
