"""Frozen Height boundary estimator v1 (Candidate A), plus objective diagnostics."""
from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import gaussian_filter1d

ESTIMATOR_ID="height_boundary_estimator_v1"
LEVELING_METHOD_ID="official_robust_plane_v1"
BOUNDARY_METHOD_ID="gradient_edges"
PARAMETER_SET_ID="threshold_5um_agg_0.1_v1"
AGGREGATION_HALF_WIDTH_MM=.10
CENTRAL_REGION=(.50,1.45)
SUBSTRATE_LEFT=(.05,.45)
SUBSTRATE_RIGHT=(1.50,1.85)
SMOOTHING_SIGMA_PIXELS=2.0
SMOOTHING_MIN_FINITE_COUNT=5
SMOOTHING_MIN_NORMALIZED_WEIGHT=.25
MIN_GRADIENT_FINITE_COUNT=20

def finite_smooth(values,sigma=SMOOTHING_SIGMA_PIXELS):
    # Preserve the stored float32 profile dtype: this is required for exact
    # compatibility with the original Candidate A generator at near-tied
    # gradient extrema.
    values=np.asarray(values);valid=np.isfinite(values)
    if valid.sum()<SMOOTHING_MIN_FINITE_COUNT:return np.full_like(values,np.nan)
    numerator=gaussian_filter1d(np.where(valid,values,0),sigma=sigma)
    denominator=gaussian_filter1d(valid.astype(float),sigma=sigma)
    out=np.full_like(values,np.nan);good=denominator>SMOOTHING_MIN_NORMALIZED_WEIGHT
    out[good]=numerator[good]/denominator[good]
    return out

def predict(profile,y):
    """Exact frozen gradient implementation. Returns (left,right,diagnostics)."""
    profile=np.asarray(profile);y=np.asarray(y,dtype=float)
    processed=finite_smooth(profile)
    result={"processed_profile":processed,"finite_profile_count":int(np.isfinite(profile).sum())}
    if np.isfinite(processed).sum()<MIN_GRADIENT_FINITE_COUNT:
        return np.nan,np.nan,{**result,"status":"insufficient_finite_profile"}
    valid=np.isfinite(processed)
    filled=np.interp(y,y[valid],processed[valid])
    gradient=np.gradient(filled,y)
    center=.5*sum(CENTRAL_REGION)
    left_indices=np.flatnonzero((y>=CENTRAL_REGION[0])&(y<=center))
    right_indices=np.flatnonzero((y>=center)&(y<=CENTRAL_REGION[1]))
    left=int(left_indices[np.argmax(gradient[left_indices])])
    right=int(right_indices[np.argmin(gradient[right_indices])])
    result.update(gradient=gradient,left_index=left,right_index=right,
                  left_gradient_magnitude=float(abs(gradient[left])),
                  right_gradient_magnitude=float(abs(gradient[right])))
    if right<=left:return np.nan,np.nan,{**result,"status":"boundaries_cross"}
    return float(y[left]),float(y[right]),{**result,"status":"ok"}

def width_center(left,right):
    if not (math.isfinite(float(left)) and math.isfinite(float(right))):return np.nan,np.nan
    return float(right-left),float((left+right)/2)
