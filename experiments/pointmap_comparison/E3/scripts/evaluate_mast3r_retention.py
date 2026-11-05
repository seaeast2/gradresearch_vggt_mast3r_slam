#!/usr/bin/env python3
"""Evaluate M-F confidence retention on its own exported keyframes in camera coordinates."""
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np
from PIL import Image

K=np.array([[525.,0,319.5],[0,525.,239.5],[0,0,1]],np.float64)
p=argparse.ArgumentParser();p.add_argument('--mast3r',type=Path,required=True);p.add_argument('--frames',type=Path,required=True);p.add_argument('--associations',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=np.load(a.mast3r,allow_pickle=False);fs=list(csv.DictReader(a.frames.open(newline='')));ass={r['rgb_timestamp']:r for r in csv.DictReader(a.associations.open(newline=''))};result={'scope':'per-keyframe local camera coordinates; no sequence/global Sim3','tau_m':.05,'free_space_gt_margin_m':.02,'frame_metrics':[],'retention_curves':[]};
for j,fid in enumerate(m['dataset_frame_id'].astype(int)):
 row=fs[fid];g=ass[row['rgb_timestamp']];raw=np.asarray(Image.open(g['depth_path']),np.uint16);gd=raw.astype(np.float32)/5000.;gv=np.isfinite(gd)&(gd>=.3)&(gd<=5.)
 h,w=gd.shape;yy,xx=np.mgrid[:h,:w];gt=np.stack(((xx-K[0,2])*gd/K[0,0],(yy-K[1,2])*gd/K[1,1],gd),-1)
 sh,sw=map(int,m['source_image_shape_hw'][j]);oh,ow=map(int,m['image_shape_hw'][j]);mx=((xx+.5)*ow/sw-.5).astype(np.float32);my=((yy+.5)*oh/sh-.5).astype(np.float32)
 points=cv2.remap(m['X_canon_local'][j].astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);conf=cv2.remap(m['confidence'][j].astype(np.float32),mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0);mask=cv2.remap(m['valid_mask'][j].astype(np.uint8),mx,my,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
 valid=gv&mask&np.isfinite(points).all(-1)&np.isfinite(conf);err=np.linalg.norm(points[valid]-gt[valid],axis=-1);c=conf[valid];z=points[valid,2];gtz=gd[valid];n=len(err)
 rec={'dataset_frame_id':int(row['frame_index']),'rgb_timestamp':row['rgb_timestamp'],'fusion_count':int(m['fusion_count'][j]),'valid_count':n,'coverage_of_gt_valid':float(n/gv.sum()) if gv.sum() else None,'point3d_median_m':float(np.median(err)) if n else None,'outlier_ratio_at_5cm':float(np.mean(err>.05)) if n else None,'free_space_violation_ratio':float(np.mean(z<gtz-.02)) if n else None};result['frame_metrics'].append(rec)
 if n:
  order=np.argsort(-c)
  for q in (1.,.8,.6,.4):
   keep=order[:max(1,int(np.ceil(n*q)))];result['retention_curves'].append({'dataset_frame_id':int(row['frame_index']),'retained_fraction':q,'retained_count':int(len(keep)),'coverage_of_gt_valid':float(len(keep)/gv.sum()),'outlier_ratio_at_5cm':float(np.mean(err[keep]>.05)),'free_space_violation_ratio':float(np.mean(points[valid][keep,2]<gd[valid][keep]-.02)),'point3d_median_m':float(np.median(err[keep]))})
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'frames':len(result['frame_metrics']),'curves':len(result['retention_curves'])},indent=2))
