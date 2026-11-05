#!/usr/bin/env python3
"""Compare local camera-frame point clouds on the same observed GT surfaces."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def voxel_downsample(points,voxel):
    p=np.asarray(points,dtype=np.float32)
    if len(p)==0:return p
    keys=np.floor(p/voxel).astype(np.int64)
    _,inverse=np.unique(keys,axis=0,return_inverse=True)
    sums=np.zeros((inverse.max()+1,3),np.float64); counts=np.bincount(inverse)
    np.add.at(sums,inverse,p)
    return (sums/counts[:,None]).astype(np.float32)


def score(pred,gt,voxel=.02,thresholds=(.02,.05,.10)):
    p=voxel_downsample(pred,voxel);g=voxel_downsample(gt,voxel)
    if not len(p) or not len(g):return {"pred_voxels":len(p),"gt_voxels":len(g),"status":"empty_cloud"}
    tg=cKDTree(g);tp=cKDTree(p)
    dpg=tg.query(p,k=1,workers=-1)[0];dgp=tp.query(g,k=1,workers=-1)[0]
    out={"status":"passed","pred_voxels":len(p),"gt_voxels":len(g),
      "accuracy_mean_m":float(dpg.mean()),"accuracy_median_m":float(np.median(dpg)),"accuracy_p90_m":float(np.percentile(dpg,90)),
      "completeness_mean_m":float(dgp.mean()),"completeness_median_m":float(np.median(dgp)),"completeness_p90_m":float(np.percentile(dgp,90)),
      "voxel_size_m":voxel,"coordinate_frame":"per-frame OpenCV camera coordinates; local metric only"}
    for t in thresholds:
        precision=float(np.mean(dpg<=t));recall=float(np.mean(dgp<=t));f=2*precision*recall/(precision+recall) if precision+recall else 0
        out[f"precision_at_{int(t*100)}cm"]=precision;out[f"recall_at_{int(t*100)}cm"]=recall;out[f"fscore_at_{int(t*100)}cm"]=f
    return out


parser=argparse.ArgumentParser();parser.add_argument("--input",type=Path,required=True);parser.add_argument("--output-dir",type=Path,required=True)
a=parser.parse_args();d=np.load(a.input,allow_pickle=False);out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
gt=d["gt_points_camera"];gm=d["gt_valid_mask"].astype(bool)
models={"V-P":("vp_points_camera","vp_valid_mask"),"V-D":("vd_points_camera","vd_valid_mask"),"M-F":("mf_points_camera","mf_valid_mask")}
rows=[]
for name,(pk,mk) in models.items():
  pts=d[pk];mask=d[mk].astype(bool)
  for i in range(len(gt)):
    common=gm[i]&mask[i]&np.isfinite(gt[i]).all(axis=-1)&np.isfinite(pts[i]).all(axis=-1)
    metric=score(pts[i][common],gt[i][common])
    rows.append({"model":name,"dataset_frame_id":int(d["dataset_frame_id"][i]),"rgb_timestamp":str(d["rgb_timestamp"][i]),
                 "input_valid_points":int(mask[i].sum()),"common_observed_points":int(common.sum()),**metric})
with (out/"local_cloud_metrics.csv").open("w",newline="") as f:
  keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
result={"scope":"local camera-frame observed-surface metrics only","global_sim3_metrics":"not computed: 4-frame input is below the frozen 60-frame prefix minimum",
 "common_visibility":"per-frame GT-valid pixels shared with predicted valid mask","voxel_size_m":.02,"frame_metrics":rows}
(out/"local_cloud_metrics.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps({"status":"completed","rows":len(rows),"nonempty":sum(r.get("status")=="passed" for r in rows)},indent=2))
