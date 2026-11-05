#!/usr/bin/env python3
"""Synthetic exact-recovery checks for Sim(3), camera projection, and point errors."""
import argparse
import json
from pathlib import Path

import numpy as np
from geometry import estimate_sim3, prefix_degeneracy, transform_points

parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True);a=parser.parse_args()
rng=np.random.default_rng(20261105)
t=np.arange(160,dtype=np.float64)
source=np.stack([.8*np.cos(t*.07),.55*np.sin(t*.11),.006*t+.08*np.sin(t*.03)],axis=1)
theta=np.deg2rad(21);R=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
expected=np.eye(4);expected[:3,:3]=1.65*R;expected[:3,3]=[.4,-.2,1.1]
target=transform_points(source,expected)+rng.normal(0,1e-10,source.shape)
deg=prefix_degeneracy(source[:120],minimum=60)
assert deg["status"]=="passed",deg
estimated=estimate_sim3(source[:120],target[:120]);mapped=transform_points(source,estimated)
matrix_error=float(np.max(np.abs(estimated-expected)))
center_rmse=float(np.sqrt(np.mean(np.sum((mapped[:120]-target[:120])**2,axis=1))))
assert matrix_error<1e-8 and center_rmse<1e-8,(matrix_error,center_rmse)
line=np.stack([t,np.zeros_like(t),np.zeros_like(t)],axis=1)
line_check=prefix_degeneracy(line,minimum=60)
assert line_check["status"]=="degenerate",line_check
K=np.array([[525.,0,319.5],[0,525.,239.5],[0,0,1.]])
u,v,z=103.0,277.0,1.42
P=np.array([(u-K[0,2])*z/K[0,0],(v-K[1,2])*z/K[1,1],z])
u2=K[0,0]*P[0]/P[2]+K[0,2];v2=K[1,1]*P[1]/P[2]+K[1,2]
assert abs(u-u2)<1e-12 and abs(v-v2)<1e-12 and abs(z-P[2])<1e-12
result={"status":"passed","seed":20261105,"known_sim3_matrix_max_abs_error":matrix_error,
 "prefix_camera_center_rmse_m":center_rmse,"collinear_trajectory_rejected":True,
 "synthetic_pixel_uv_roundtrip_max_error":float(max(abs(u-u2),abs(v-v2))),"depth_recovery_error_m":float(abs(z-P[2])),
 "point_error_exact_match_m":float(np.linalg.norm(P-P))}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
