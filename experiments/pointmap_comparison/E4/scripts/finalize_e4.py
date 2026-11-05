#!/usr/bin/env python3
"""Hash E4 inputs, code snapshot, figures, and tables into one run manifest."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--e3-stage',type=Path,required=True);p.add_argument('--run-id',required=True);a=p.parse_args();stage=a.stage.resolve();e3=a.e3_stage.resolve()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
inputs=[]
for name,path in [('E3 manifest',e3/'manifest.json'),('desk P1 paired pointmaps',e3/'artifacts/conditions/desk/P1/C0/N04/metrics/vggt-r01_mast3r-r03/pixel/local_pointmaps_for_cloud_outlier.npz'),('room P1 paired pointmaps',e3/'artifacts/conditions/room/P1/C0/N04/metrics/vggt-r01_mast3r-r01/pixel/local_pointmaps_for_cloud_outlier.npz'),('room P2 VGGT export',e3/'artifacts/conditions/room/P2/VGGT/fixed04/repeat01/artifacts/exports/vggt_pointmaps.npz')]:
 if path.exists():inputs.append({'name':name,'path':str(path),'size_bytes':path.stat().st_size,'sha256':sha(path)})
outputs=[]
for path in sorted(stage.rglob('*')):
 if path.is_file() and path.name!='manifest.json':outputs.append({'path':str(path.relative_to(stage)),'size_bytes':path.stat().st_size,'sha256':sha(path)})
doc={'run_id':a.run_id,'stage_code':'E4','stage_name':'GT-labeled outlier cases, uncertainty indicators, and same-view clouds','status':'completed','execution_context':'host','analysis_device':'CPU','source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'inputs':inputs,'outlier_threshold_m':.05,'gt_depth_valid_m':[.3,5.0],'case_count':5,'semantic_labels':'manual exploratory tags; GT labels are geometric only','outputs':outputs}
(stage/'manifest.json').write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps({'status':doc['status'],'inputs':len(inputs),'outputs':len(outputs),'manifest':str(stage/'manifest.json')},indent=2))
