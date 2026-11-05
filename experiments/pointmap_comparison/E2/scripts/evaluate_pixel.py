#!/usr/bin/env python3
"""Map exported predictions to original TUM RGB pixels and compute local GT metrics."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

K = np.array([[525.0,0,319.5],[0,525.0,239.5],[0,0,1]],dtype=np.float64)
DEPTH_SCALE=5000.0


def remap_field(field, source_hw, resized_hw, padding_tblr=None, is_mask=False):
    src_h,src_w=map(int,source_hw); out_h,out_w=map(int,resized_hw)
    yy,xx=np.mgrid[0:src_h,0:src_w].astype(np.float32)
    if padding_tblr is None:
        top,left=0,0
    else:
        top,_,left,_=map(int,padding_tblr)
    mx=((xx+0.5)*out_w/src_w-0.5+left).astype(np.float32)
    my=((yy+0.5)*out_h/src_h-0.5+top).astype(np.float32)
    interp=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.remap(field.astype(np.uint8 if is_mask else np.float32),mx,my,interp,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=0 if is_mask else float("nan"))


def camera_points(depth):
    h,w=depth.shape; v,u=np.mgrid[0:h,0:w]
    return np.stack(((u-K[0,2])*depth/K[0,0],(v-K[1,2])*depth/K[1,1],depth),axis=-1).astype(np.float32)


def metrics(points,conf,mask,gt_points,gt_depth,gt_valid):
    valid=mask & gt_valid & np.isfinite(points).all(axis=-1) & (points[...,2]>0)
    n_gt=int(gt_valid.sum()); n_pred=int((mask & np.isfinite(points).all(axis=-1)).sum()); n=int(valid.sum())
    if not n:
        return {"n_gt_valid":n_gt,"n_pred_valid":n_pred,"n_joint_valid":0,"valid_fraction":0.0}
    p=points[valid].astype(np.float64); g=gt_points[valid].astype(np.float64)
    z=p[:,2]; zg=gt_depth[valid].astype(np.float64)
    de=np.abs(z-zg); e3=np.linalg.norm(p-g,axis=1)
    out={"n_gt_valid":n_gt,"n_pred_valid":n_pred,"n_joint_valid":n,
         "valid_fraction":float(n/n_gt) if n_gt else 0,
         "depth_absrel_mean":float(np.mean(de/zg)),"depth_rmse_m":float(np.sqrt(np.mean((z-zg)**2))),
         "depth_medae_m":float(np.median(de)),"point3d_mean_m":float(np.mean(e3)),
         "point3d_median_m":float(np.median(e3)),"point3d_p90_m":float(np.percentile(e3,90)),
         "point3d_p95_m":float(np.percentile(e3,95))}
    out.update({f"point3d_within_{cm}cm_fraction":float(np.mean(e3<=cm/100)) for cm in (2,5,10)})
    if conf is not None:
        c=conf[valid].astype(np.float64)
        out["confidence_median"]=float(np.median(c))
        for q in (1.0,.8,.6,.4):
            threshold=np.quantile(c,1-q)
            retained=c>=threshold
            out[f"retained_{int(q*100)}pct_count"]=int(retained.sum())
            out[f"retained_{int(q*100)}pct_point3d_median_m"]=float(np.median(e3[retained]))
    return out


parser=argparse.ArgumentParser()
parser.add_argument("--vggt",type=Path,required=True); parser.add_argument("--mast3r",type=Path,required=True)
parser.add_argument("--frames",type=Path,required=True); parser.add_argument("--associations",type=Path,required=True)
parser.add_argument("--output-dir",type=Path,required=True)
a=parser.parse_args(); v=np.load(a.vggt,allow_pickle=False); m=np.load(a.mast3r,allow_pickle=False)
frames=list(csv.DictReader(a.frames.open(newline=""))); assoc=list(csv.DictReader(a.associations.open(newline="")))
by_ts={r["rgb_timestamp"]:r for r in assoc}
frame_count=len(frames); height,width=480,640
models={"V-P":[],"V-D":[],"M-F":[]}; confs={"V-P":[],"V-D":[],"M-F":[]}; masks={k:[] for k in models}
gt_depths=[]; gt_points=[]; gt_masks=[]; timestamps=[]; source_ids=[]; mf_fusion=[]; mf_updates=[]
m_slot={int(fid):i for i,fid in enumerate(m["dataset_frame_id"])}
for i,row in enumerate(frames):
    ts=row["rgb_timestamp"]; g=by_ts.get(ts)
    if g is None or g["within_tolerance"].lower()!="true": raise RuntimeError(f"no valid depth/pose association for E1 frame {ts}")
    raw=np.asarray(Image.open(g["depth_path"]),dtype=np.uint16)
    if raw.shape!=(height,width): raise RuntimeError(f"depth shape mismatch {raw.shape}")
    gd=raw.astype(np.float32)/DEPTH_SCALE; gv=np.isfinite(gd)&(gd>=.30)&(gd<=5.0)
    gp=camera_points(gd); gp[~gv]=np.nan
    gt_depths.append(gd);gt_points.append(gp);gt_masks.append(gv);timestamps.append(ts);source_ids.append(int(row["frame_index"]))
    src_hw=v["source_image_shape_hw"][i]; out_hw=v["image_shape_hw"][i]
    pad=v["padding_tblr"][i]; resize=v["resized_image_shape_hw"][i]
    pvp=remap_field(v["V_P"][i],src_hw,resize,pad); pvd=remap_field(v["V_D"][i],src_hw,resize,pad)
    ex=v["extrinsic"][i].astype(np.float64); R,t=ex[:,:3],ex[:,3]
    pvp_cam=pvp@R.T+t; pvd_cam=pvd@R.T+t
    vp_mask=remap_field(v["valid_mask"][i],src_hw,resize,pad,True).astype(bool)&np.isfinite(pvp_cam).all(axis=-1)
    vd_mask=remap_field(v["valid_mask"][i],src_hw,resize,pad,True).astype(bool)&np.isfinite(pvd_cam).all(axis=-1)
    vp_conf=remap_field(v["V_P_conf"][i],src_hw,resize,pad)
    vd_conf=remap_field(v["depth_conf"][i],src_hw,resize,pad)
    models["V-P"].append(pvp_cam);models["V-D"].append(pvd_cam);confs["V-P"].append(vp_conf);confs["V-D"].append(vd_conf)
    masks["V-P"].append(vp_mask);masks["V-D"].append(vd_mask)
    if i in m_slot:
        j=m_slot[i]; mhw=m["image_shape_hw"][j]; msrc=m["source_image_shape_hw"][j]
        mf=remap_field(m["X_canon_local"][j],msrc,mhw); mc=remap_field(m["confidence"][j],msrc,mhw)
        mm=remap_field(m["valid_mask"][j],msrc,mhw,None,True).astype(bool)&np.isfinite(mf).all(axis=-1)
        models["M-F"].append(mf);confs["M-F"].append(mc);masks["M-F"].append(mm)
        mf_fusion.append(int(m["fusion_count"][j]));mf_updates.append(int(m["update_count"][j]))
    else:
        models["M-F"].append(np.full((height,width,3),np.nan,np.float32));confs["M-F"].append(np.full((height,width),np.nan,np.float32));masks["M-F"].append(np.zeros((height,width),bool))
        mf_fusion.append(-1);mf_updates.append(-1)

gt_depths=np.stack(gt_depths);gt_points=np.stack(gt_points);gt_masks=np.stack(gt_masks);gt_frames=[]
summary={"schema":"pointmap-comparison.e2.v1","intrinsics":{"fx":525,"fy":525,"cx":319.5,"cy":239.5,"source":"TUM recommended ROS default calibration, no undistortion"},
 "depth_scale":DEPTH_SCALE,"valid_depth_m":[.3,5.0],"depth_tolerance_m":.02,"tau_m":.05,"voxel_size_m":.02,
 "input_frames":frame_count,"outputs":{},"global_sequence_alignment":"unavailable: only 4 input frames, below preregistered 60-frame minimum; local camera-frame metrics only"}
csv_rows=[]
for model in models:
    models[model]=np.stack(models[model]);confs[model]=np.stack(confs[model]);masks[model]=np.stack(masks[model])
    per=[]
    for i in range(frame_count):
        d=metrics(models[model][i],confs[model][i],masks[model][i],gt_points[i],gt_depths[i],gt_masks[i])
        rec={"model":model,"dataset_frame_id":int(source_ids[i]),"rgb_timestamp":timestamps[i],**d};csv_rows.append(rec);per.append(rec)
    good=[r for r in per if r.get("n_joint_valid",0)>0]
    summary["outputs"][model]={"exported_frame_count":int(sum(x.sum()>0 for x in masks[model])),"frame_metrics":per,
      "scene_macro":{"frames_with_metrics":len(good),"depth_absrel_mean":float(np.mean([r["depth_absrel_mean"] for r in good])) if good else None,
      "depth_rmse_m_macro":float(np.mean([r["depth_rmse_m"] for r in good])) if good else None,
      "point3d_median_m_macro":float(np.mean([r["point3d_median_m"] for r in good])) if good else None}}
out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
with (out/"pixel_metrics.csv").open("w",newline="") as f:
    keys=list(dict.fromkeys(k for r in csv_rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(csv_rows)
arrays={"dataset_frame_id":np.asarray(source_ids),"rgb_timestamp":np.asarray(timestamps),"gt_depth":gt_depths,"gt_points_camera":gt_points,"gt_valid_mask":gt_masks}
arrays["mast3r_fusion_count"]=np.asarray(mf_fusion,dtype=np.int32);arrays["mast3r_update_count"]=np.asarray(mf_updates,dtype=np.int32)
for model,key in [("V-P","vp"),("V-D","vd"),("M-F","mf")]:
    arrays[key+"_points_camera"]=models[model];arrays[key+"_confidence"]=confs[model];arrays[key+"_valid_mask"]=masks[model]
np.savez_compressed(out/"local_pointmaps_for_cloud_outlier.npz",**arrays)
(out/"pixel_metrics.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps({"status":"completed","frames":frame_count,"summary":summary["outputs"]},indent=2))
