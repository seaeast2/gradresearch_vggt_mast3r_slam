#!/usr/bin/env python3
"""Hash E2 run inputs/outputs and write its terminal manifest."""
import argparse,hashlib,json,subprocess
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--stage',type=Path,required=True);parser.add_argument('--run-id',required=True)
parser.add_argument('--e0-stage',type=Path,required=True);parser.add_argument('--e1-stage',type=Path,required=True)
parser.add_argument('--status',choices=['running','completed','failed'],required=True);parser.add_argument('--exit-code',type=int,default=0)
a=parser.parse_args();stage=a.stage.resolve()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
outputs=[]
for p in sorted(stage.rglob('*')):
 if p.is_file() and p.name!='manifest.json':outputs.append({'path':str(p.relative_to(stage)),'size_bytes':p.stat().st_size,'sha256':sha(p)})
inputs=[]
for label,root,files in [('E0 data',a.e0_stage,[a.e0_stage/'artifacts/data/frames_associated.csv',a.e0_stage/'artifacts/data/rgbd_dataset_freiburg1_desk.tgz']),
                         ('E1 exports',a.e1_stage,[a.e1_stage/'artifacts/exports/vggt/vggt_pointmaps.npz',a.e1_stage/'artifacts/exports/mast3r/mast3r_mf_pointmaps.npz'])]:
 inputs.append({'name':label,'stage_path':str(root),'files':[{'path':str(p),'sha256':sha(p)} for p in files]})
manifest={'run_id':a.run_id,'stage_code':'E2','stage_name':'GT-based quantitative evaluation pipeline','status':a.status,
 'execution_context':'host','source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
 'inputs':inputs,'settings':{'timestamp_tolerance_ms':20,'depth_scale':5000,'valid_depth_m':[.3,5.0],
 'gt_intrinsics_ros_default':[525,525,319.5,239.5],'alignment_prefix_minimum':60,'tau_m':.05,'voxel_size_m':.02},'outputs':outputs}
if a.status=='failed':manifest['exit_code']=a.exit_code
(stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
