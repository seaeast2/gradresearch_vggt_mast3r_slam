#!/usr/bin/env python3
"""Build one exact, timestamp-ordered shared P1 frame list from TUM associations."""
import argparse,csv,json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--dataset-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--budgets',default='4,8,16,32')
a=p.parse_args();root=a.dataset_dir.resolve();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
def pairs(path):
 values=[]
 for line in path.open():
  line=line.strip()
  if line and not line.startswith('#'):
   stamp,name=line.split()[:2];values.append((float(stamp),name))
 return values
rgb=pairs(root/'rgb.txt');depth=pairs(root/'depth.txt')
poses=[]
with (root/'groundtruth.txt').open() as f:
 for line in f:
  line=line.strip()
  if line and not line.startswith('#'):
   fields=line.split();poses.append((float(fields[0]),fields[1:]))
def nearest(stamps,t):return min(stamps,key=lambda x:abs(x-t))
dt=[x[0] for x in depth];pt=[x[0] for x in poses]
associated=[]
for i,(ts,rgb_file) in enumerate(rgb):
 d=nearest(dt,ts);q=nearest(pt,ts)
 dp=next(x for x in depth if x[0]==d);gp=next(x for x in poses if x[0]==q)
 dd,gd=abs(d-ts),abs(q-ts)
 associated.append({'frame_index':i,'rgb_timestamp':f'{ts:.6f}','rgb_path':str((root/rgb_file).resolve()),'depth_timestamp':f'{d:.6f}','depth_path':str((root/dp[1]).resolve()),'depth_abs_delta_s':f'{dd:.6f}','groundtruth_timestamp':f'{q:.6f}','groundtruth_pose_xyzw':' '.join(gp[1]),'groundtruth_abs_delta_s':f'{gd:.6f}','within_tolerance':str(dd<=.020 and gd<=.020).lower()})
valid=[r for r in associated if r['within_tolerance']=='true']
with (out/'frames_associated.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=associated[0].keys());w.writeheader();w.writerows(associated)
with (out/'frames_all.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=valid[0].keys());w.writeheader();w.writerows(valid)
for n in map(int,a.budgets.split(',')):
 if len(valid)<n:raise RuntimeError(f'{a.dataset}: budget {n} requested, only {len(valid)} RGB-depth-pose matches')
 subset=valid[:n];path=out/f'frames_{n:02d}.csv'
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=subset[0].keys());w.writeheader();w.writerows(subset)
summary={'dataset':a.dataset,'rgb_count':len(rgb),'depth_count':len(depth),'pose_count':len(poses),'valid_rgb_depth_pose_count':len(valid),'tolerance_ms':20,'selection':'first N within-tolerance rows in original RGB order','budgets':{str(n):{'frame_indices':[int(x['frame_index']) for x in valid[:n]],'timestamps':[x['rgb_timestamp'] for x in valid[:n]],'frame_list':str(out/f'frames_{n:02d}.csv')} for n in map(int,a.budgets.split(','))}}
idx=[round(i*(len(valid)-1)/3) for i in range(4)];p2=[valid[i] for i in idx]
with (out/'frames_p2_vggt.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=p2[0].keys());w.writeheader();w.writerows(p2)
summary['P2']={'mast3r_full_frame_list':str(out/'frames_all.csv'),'vggt_uniform_four_frame_indices':[int(r['frame_index']) for r in p2],'vggt_frame_list':str(out/'frames_p2_vggt.csv'),'protocol_note':'not an equal-input P1 accuracy comparison'}
(out/'association_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
