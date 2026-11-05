#!/usr/bin/env python3
"""Associate TUM RGB, registered depth, and mocap pose timestamps."""
import argparse
import bisect
import csv
import json
from pathlib import Path


def read_pairs(path):
    out = []
    for line in path.read_text().splitlines():
        line=line.strip()
        if line and not line.startswith("#"):
            ts, name, *_ = line.split()
            out.append((float(ts), name))
    return out


def nearest(rows, stamps, target):
    i=bisect.bisect_left(stamps,target)
    candidates=rows[max(0,i-1):min(len(rows),i+1)]
    return min(candidates,key=lambda r:abs(r[0]-target))


parser=argparse.ArgumentParser()
parser.add_argument("--dataset-dir",type=Path,required=True)
parser.add_argument("--output-dir",type=Path,required=True)
parser.add_argument("--depth-tolerance-ms",type=float,default=20)
parser.add_argument("--pose-tolerance-ms",type=float,default=20)
a=parser.parse_args(); root=a.dataset_dir.resolve(); out=a.output_dir.resolve()
rgb=read_pairs(root/"rgb.txt"); depth=read_pairs(root/"depth.txt")
pose=[]
for line in (root/"groundtruth.txt").read_text().splitlines():
    line=line.strip()
    if line and not line.startswith("#"):
        f=line.split(); pose.append((float(f[0])," ".join(f[1:])))
if not rgb or not depth or not pose: raise RuntimeError("TUM sequence must contain rgb.txt, depth.txt, and groundtruth.txt")
ds=[x[0] for x in depth]; ps=[x[0] for x in pose]
rows=[]
for i,(rt,rf) in enumerate(rgb):
    dt,df=nearest(depth,ds,rt); pt,pf=nearest(pose,ps,rt)
    dd, pd=abs(dt-rt),abs(pt-rt)
    rows.append({"dataset_frame_id":i,"rgb_timestamp":f"{rt:.6f}","rgb_file":rf,
      "rgb_path":str(root/rf),"depth_timestamp":f"{dt:.6f}","depth_file":df,
      "depth_path":str(root/df),"depth_abs_delta_s":f"{dd:.9f}",
      "groundtruth_timestamp":f"{pt:.6f}","groundtruth_pose_xyzw":pf,
      "groundtruth_abs_delta_s":f"{pd:.9f}",
      "depth_within_tolerance":dd<=a.depth_tolerance_ms/1000,
      "pose_within_tolerance":pd<=a.pose_tolerance_ms/1000,
      "within_tolerance":dd<=a.depth_tolerance_ms/1000 and pd<=a.pose_tolerance_ms/1000})
out.mkdir(parents=True,exist_ok=True)
with (out/"frames_associated.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summary={"dataset_dir":str(root),"rgb_count":len(rgb),"depth_count":len(depth),"pose_count":len(pose),
 "valid_rgb_depth_pose_count":sum(r["within_tolerance"] for r in rows),
 "depth_tolerance_ms":a.depth_tolerance_ms,"pose_tolerance_ms":a.pose_tolerance_ms,
 "association":"independent nearest timestamp to RGB; inclusive tolerance",
 "depth_scale":5000,"depth_registered_to_rgb":True}
(out/"association_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
print(json.dumps(summary,indent=2))
