#!/usr/bin/env python3
"""Estimate a fixed prefix Sim(3), or record why it cannot be estimated."""
import argparse
import json
from pathlib import Path

import numpy as np
from geometry import estimate_prefix_sim3

parser=argparse.ArgumentParser();parser.add_argument("--input",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--minimum",type=int,default=60);parser.add_argument("--prefix-count",type=int,default=None)
a=parser.parse_args();data=json.loads(a.input.read_text());src=np.asarray(data["source_centers"],float);dst=np.asarray(data["target_centers"],float)
n=a.prefix_count or int(np.ceil(.2*len(src)));n=max(n,a.minimum)
if len(src)!=len(dst): raise ValueError("source and target centers have different lengths")
if len(src)<n:
 result={"status":"insufficient_prefix","available_count":len(src),"required_prefix_count":n,
         "frozen_minimum":a.minimum,"prefix_policy":"first 20 percent, at least the frozen minimum"}
else:
 fit=estimate_prefix_sim3(src[:n],dst[:n],minimum=a.minimum)
 result={**fit,"available_count":len(src),"alignment_prefix_count":n,
         "evaluation_frame_count":len(src)-n,"prefix_policy":"first 20 percent, at least the frozen minimum"}
 if fit.get("status")=="passed":
  S=np.asarray(fit["matrix_source_to_target"]);aligned=src[n:]@S[:3,:3].T+S[:3,3]
  result["heldout_camera_center_rmse_m"]=float(np.sqrt(np.mean(np.sum((aligned-dst[n:])**2,axis=1)))) if n<len(src) else None
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+"\n");print(json.dumps(result,indent=2))
