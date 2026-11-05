#!/usr/bin/env python3
"""Create stage-level CSV/JSON tables from immutable E3 condition manifests."""
import argparse,csv,json,statistics,hashlib,subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();stage=a.stage.resolve();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);rows=[]
for mp in sorted((stage/'artifacts/conditions').rglob('manifest.json')):
 m=json.load(mp.open());parts=mp.relative_to(stage/'artifacts/conditions').parts
 dataset=parts[0];protocol=parts[1];model='VGGT' if 'VGGT' in parts else 'MASt3R';cal='C1' if 'C1' in parts else ('C0' if 'C0' in parts else 'n/a')
 budget=None
 for part in parts:
  if part.startswith('N') and part[1:].isdigit():budget=int(part[1:])
  if part.startswith('fixed') and part[5:].isdigit():budget=int(part[5:])
 if protocol=='P2' and model=='MASt3R':budget=1362
 repeat=next((int(x[6:]) for x in parts if x.startswith('repeat')),1)
 meta_path=mp.parent/'artifacts/exports'/'metadata.json';meta=json.load(meta_path.open()) if meta_path.exists() else {}
 outputs=meta.get('exported_keyframe_count',meta.get('count',0));files=mp.parent/'artifacts/exports'
 retention_path=mp.parent/'artifacts/metrics/mf_retention.json';ret=json.load(retention_path.open()) if retention_path.exists() else {}
 fs=ret.get('frame_metrics',[]);curves=ret.get('retention_curves',[]);top40=[x for x in curves if x['retained_fraction']==.4]
 wall=m.get('wall_time_s');mem=meta.get('peak_allocated_bytes');reserved=meta.get('peak_reserved_bytes');gpu_mib=m.get('peak_gpu_memory_mib')
 paired_path=stage/'artifacts/conditions'/dataset/'P1'/'C0'/f'N{budget:02d}'/'metrics'/(('vggt-r01_mast3r-r03' if dataset=='desk' else 'vggt-r01_mast3r-r01') if budget==4 else 'missing')/'pixel'/'pixel_metrics.json' if protocol=='P1' and cal=='C0' and budget==4 else Path('/missing')
 paired=json.load(paired_path.open()) if paired_path.exists() else {}
 rows.append({'dataset':dataset,'protocol':protocol,'N':budget,'calibration':cal,'model':model,'repeat':repeat,'status':m.get('status'),'exit_code':m.get('exit_code'),'wall_time_s':wall,'exported_frames':outputs,'coverage_fraction':outputs/budget if budget else None,'peak_allocated_bytes':mem,'peak_reserved_bytes':reserved,'peak_gpu_memory_mib':gpu_mib,'m_fusion_count':','.join(map(str,meta.get('fusion_count',[]))),'mf_camera_local_median_error_macro_m':statistics.mean([x['point3d_median_m'] for x in fs if x.get('point3d_median_m') is not None]) if fs else None,'mf_top40_outlier_ratio_macro':statistics.mean([x['outlier_ratio_at_5cm'] for x in top40]) if top40 else None,'mf_top40_coverage_macro':statistics.mean([x['coverage_of_gt_valid'] for x in top40]) if top40 else None,'paired_VP_AbsRel_macro':paired.get('outputs',{}).get('V-P',{}).get('scene_macro',{}).get('depth_absrel_mean'),'paired_VD_AbsRel_macro':paired.get('outputs',{}).get('V-D',{}).get('scene_macro',{}).get('depth_absrel_mean'),'paired_MF_AbsRel_macro':paired.get('outputs',{}).get('M-F',{}).get('scene_macro',{}).get('depth_absrel_mean'),'manifest':str(mp.relative_to(stage)),'vram_note':'VGGT PyTorch peak allocated/reserved bytes' if mem else ('nvidia-smi 100 ms sampled peak' if gpu_mib else ('not sampled by MASt3R launcher' if model=='MASt3R' else 'unavailable on failed run'))})
keys=list(rows[0]) if rows else []
with (out/'condition_matrix.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
summary={'schema':'pointmap-comparison.e3.v1','run_stage':str(stage),'condition_rows':rows,'condition_count':len(rows),'status_counts':{s:sum(x['status']==s for x in rows) for s in sorted({x['status'] for x in rows})},'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
repeatability=[]
for scene,model,n in [('desk','VGGT',4),('room','VGGT',4),('desk','MASt3R',32),('room','MASt3R',32)]:
 pth=stage/'artifacts/conditions'/scene/'P1'/'C0'/f'N{n:02d}'/model
 rel='artifacts/exports/vggt_pointmaps.npz' if model=='VGGT' else 'artifacts/exports/mast3r_mf_pointmaps.npz';items=[]
 for mp in sorted(pth.glob('repeat*/manifest.json')):
  m=json.load(mp.open());artifact=mp.parent/rel;items.append({'repeat':mp.parent.name,'status':m.get('status'),'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest() if artifact.exists() else None})
 successes=[x['sha256'] for x in items if x['status']=='completed' and x['sha256']]
 repeatability.append({'dataset':scene,'model':model,'protocol':'P1-C0','N':n,'successful_repeats':len(successes),'bit_identical_exports':bool(successes) and len(set(successes))==1,'runs':items})
summary['repeatability']=repeatability
(out/'condition_matrix.json').write_text(json.dumps(summary,indent=2)+'\n');return_value={'csv':str(out/'condition_matrix.csv'),'json':str(out/'condition_matrix.json'),'conditions':len(rows)};print(json.dumps(return_value,indent=2))
