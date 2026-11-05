#!/usr/bin/env python3
"""Write E3 stage provenance from dataset/model inputs and condition manifests."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--run-id',required=True);p.add_argument('--status',choices=['running','completed','completed_with_failures'],required=True);a=p.parse_args();stage=a.stage.resolve()
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
inputs=[]
for scene,path in [('desk',stage/'artifacts/data/desk/association_summary.json'),('room archive',stage/'artifacts/data/tum_room/rgbd_dataset_freiburg1_room.tgz'),('room associations',stage/'artifacts/data/room/association_summary.json')]:
 if path.exists():inputs.append({'name':scene,'path':str(path.relative_to(stage)),'size_bytes':path.stat().st_size,'sha256':digest(path)})
e0=stage.parents[2]/'tum-freiburg1_desk-20261105-001/stages/E0'
for label,path in [('VGGT checkpoint',e0/'artifacts/checkpoints/vggt/model.pt'),('MASt3R model checkpoint',e0/'artifacts/checkpoints/mast3r_official_hf/model.safetensors'),('MASt3R retrieval checkpoint',e0/'artifacts/checkpoints/mast3r_retrieval_mirror/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth')]:
 if path.exists():inputs.append({'name':label,'path':str(path),'size_bytes':path.stat().st_size,'sha256':digest(path)})
conditions=[]
for mp in sorted((stage/'artifacts/conditions').rglob('manifest.json')):
 try:m=json.load(mp.open());conditions.append({'path':str(mp.relative_to(stage)),'status':m.get('status'),'sha256':digest(mp)})
 except Exception as e:conditions.append({'path':str(mp.relative_to(stage)),'error':str(e)})
summary=stage/'artifacts/summary/condition_matrix.json';rep={'condition_count':len(conditions),'condition_status_counts':{s:sum(c.get('status')==s for c in conditions) for s in sorted({c.get('status') for c in conditions if c.get('status')})},'summary_path':str(summary.relative_to(stage)) if summary.exists() else None,'summary_sha256':digest(summary) if summary.exists() else None}
code=[]
for path in sorted((stage/'config').rglob('*')):
 if path.is_file():code.append({'path':str(path.relative_to(stage)),'size_bytes':path.stat().st_size,'sha256':digest(path)})
doc={'run_id':a.run_id,'stage_code':'E3','stage_name':'P1 pilot matrix, sensitivity, repeatability, and P2','status':a.status,'execution_context':'host','source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'inputs':inputs,'code_snapshot':code,'settings':{'p1_budgets':[4,8,16,32],'rgb_depth_pose_tolerance_ms':20,'vggt_safe_input_limit':4,'confidence_retention':[1.0,.8,.6,.4],'p2_dataset':'freiburg1_room','p2_vggt_uniform_frames':4},'condition_manifests':conditions,'summary':rep}
(stage/'manifest.json').write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps({'status':a.status,**rep},indent=2))
