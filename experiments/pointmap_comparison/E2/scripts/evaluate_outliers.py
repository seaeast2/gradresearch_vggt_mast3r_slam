#!/usr/bin/env python3
"""Measure same-pixel surface outliers, free-space violations, and retention curves."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

parser=argparse.ArgumentParser();parser.add_argument("--input",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
a=parser.parse_args();d=np.load(a.input,allow_pickle=False);gt=d["gt_points_camera"];gd=d["gt_depth"];gm=d["gt_valid_mask"].astype(bool)
models={"V-P":("vp_points_camera","vp_confidence","vp_valid_mask"),"V-D":("vd_points_camera","vd_confidence","vd_valid_mask"),"M-F":("mf_points_camera","mf_confidence","mf_valid_mask")}
result={"tau_m":.05,"free_space_gt_margin_m":.02,"frame_metrics":[],"retention_curves":[],"confidence_error_correlations":{},
 "fusion_count_relation":{"status":"not_estimable_with_one_exported_keyframe"}}
all_err={}
for name,(pk,ck,mk) in models.items():
  points=d[pk];conf=d[ck];mask=d[mk].astype(bool);errs=[];confs=[]
  for i in range(len(gt)):
    valid=gm[i]&mask[i]&np.isfinite(points[i]).all(axis=-1)&np.isfinite(conf[i])
    n=int(valid.sum());frame_id=int(d["dataset_frame_id"][i])
    if not n:
      result["frame_metrics"].append({"model":name,"dataset_frame_id":frame_id,"valid_count":0,"status":"no_model_output"});continue
    p=points[i][valid].astype(np.float64);g=gt[i][valid].astype(np.float64);c=conf[i][valid].astype(np.float64)
    err=np.linalg.norm(p-g,axis=1);free=p[:,2] < gd[i][valid]-.02;outlier=err>.05
    errs.append(err);confs.append(c);all_err.setdefault(name,[]).append(err)
    result["frame_metrics"].append({"model":name,"dataset_frame_id":frame_id,"valid_count":n,
      "surface_outlier_ratio_at_5cm":float(outlier.mean()),"free_space_violation_ratio":float(free.mean()),
      "gt_valid_count":int(gm[i].sum()),"coverage_valid_gt_fraction":float(n/gm[i].sum()) if gm[i].sum() else None})
    order=np.argsort(-c)
    for q in (1.0,.8,.6,.4):
      keep=order[:max(1,int(np.ceil(n*q)))]; e=err[keep]
      result["retention_curves"].append({"model":name,"dataset_frame_id":frame_id,"retained_fraction":q,
        "retained_count":len(keep),"surface_outlier_ratio_at_5cm":float(np.mean(e>.05)),
        "free_space_violation_ratio":float(np.mean(p[keep,2] < gd[i][valid][keep]-.02)),
        "coverage_of_gt_valid":float(len(keep)/gm[i].sum()) if gm[i].sum() else None})
  if errs:
    e=np.concatenate(errs);c=np.concatenate(confs)
    corr=spearmanr(c,e)
    result["confidence_error_correlations"][name]={"spearman_rho":float(corr.statistic) if np.isfinite(corr.statistic) else None,
        "p_value":float(corr.pvalue) if np.isfinite(corr.pvalue) else None,"point_count":len(e),
        "interpretation":"within-model descriptive association; not calibrated probability"}
vp=d["vp_points_camera"];vd=d["vd_points_camera"]
common=gm & d["vp_valid_mask"].astype(bool) & d["vd_valid_mask"].astype(bool)
common &= np.isfinite(vp).all(axis=-1)&np.isfinite(vd).all(axis=-1)&np.isfinite(gt).all(axis=-1)
if common.any():
    disagreement=np.linalg.norm(vp[common]-vd[common],axis=1)
    error_vp=np.linalg.norm(vp[common]-gt[common],axis=1)
    error_vd=np.linalg.norm(vd[common]-gt[common],axis=1)
    rho_vp=spearmanr(disagreement,error_vp).statistic;rho_vd=spearmanr(disagreement,error_vd).statistic
    result["vggt_vp_vd_disagreement"]={"point_count":int(common.sum()),"mean_m":float(disagreement.mean()),"median_m":float(np.median(disagreement)),
        "spearman_disagreement_vs_V-P_error":float(rho_vp) if np.isfinite(rho_vp) else None,
        "spearman_disagreement_vs_V-D_error":float(rho_vd) if np.isfinite(rho_vd) else None}
if np.any(d["mast3r_fusion_count"]>=0):
  unique=np.unique(d["mast3r_fusion_count"][d["mast3r_fusion_count"]>=0])
  result["fusion_count_relation"]={"status":"descriptive_frame_level_only" if len(unique)>1 else "not_estimable_with_one_fusion_count",
       "observed_fusion_counts":unique.astype(int).tolist()}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"status":"completed","frame_rows":len(result["frame_metrics"]),"retention_rows":len(result["retention_curves"])},indent=2))
