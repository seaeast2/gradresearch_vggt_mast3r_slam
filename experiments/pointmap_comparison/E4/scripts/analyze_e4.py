#!/usr/bin/env python3
"""Make preregistered E4 case figures and point-level uncertainty summaries."""
import argparse,csv,json,math
from pathlib import Path
import cv2,numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

K=np.array([[525.,0,319.5],[0,525.,239.5],[0,0,1]],np.float64);TAU=.05;DEPTH_SCALE=5000.
p=argparse.ArgumentParser();p.add_argument('--e3-stage',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();e3=a.e3_stage.resolve();out=a.output_dir.resolve();figdir=out/'figures';cloud_dir=out/'same_view_clouds';figdir.mkdir(parents=True,exist_ok=True);cloud_dir.mkdir(parents=True,exist_ok=True)
cases=[
 {'id':'C01','category':'desk edge','scene':'desk','protocol':'P1','source_frame_index':17,'roi_xyxy':[235,175,590,430],'note':'intersecting desk planes and exposed edge'},
 {'id':'C02','category':'occlusion boundary','scene':'room','protocol':'P1','source_frame_index':0,'roi_xyxy':[0,75,280,470],'note':'foreground teddy bear and monitor occlude the desks'},
 {'id':'C03','category':'texture-poor wall','scene':'room','protocol':'P2','source_frame_index':454,'roi_xyxy':[300,45,525,390],'note':'large low-texture wall/divider area on the long route'},
 {'id':'C04','category':'dark display surface','scene':'desk','protocol':'P1','source_frame_index':18,'roi_xyxy':[0,15,250,235],'note':'dark monitor glass; reflective-surface hypothesis is exploratory'},
 {'id':'C05','category':'long motion endpoint','scene':'room','protocol':'P2','source_frame_index':1361,'roi_xyxy':[55,35,600,445],'note':'last P2 view after traversing the full room sequence'},
]
(out/'cases.json').write_text(json.dumps(cases,indent=2)+'\n')
def read_csv(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))
def img_points(depth):
 h,w=depth.shape;y,x=np.mgrid[:h,:w];return np.stack(((x-K[0,2])*depth/K[0,0],(y-K[1,2])*depth/K[1,1],depth),-1).astype(np.float32)
def remap_field(field,source_hw,net_hw,pad=None,nearest=False):
 h,w=map(int,source_hw);nh,nw=map(int,net_hw);yy,xx=np.mgrid[:h,:w].astype(np.float32)
 if pad is None:top,left=0,0
 else:top,_,left,_=map(int,pad)
 mx=((xx+.5)*nw/w-.5+left).astype(np.float32);my=((yy+.5)*nh/h-.5+top).astype(np.float32)
 return cv2.remap(np.asarray(field,dtype=np.uint8 if nearest else np.float32),mx,my,cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
def load_p1(scene):
 cond=e3/'artifacts/conditions'/scene/'P1'/'C0'/'N04'/'metrics'/('vggt-r01_mast3r-r03' if scene=='desk' else 'vggt-r01_mast3r-r01')/'pixel'
 d=np.load(cond/'local_pointmaps_for_cloud_outlier.npz',allow_pickle=False);rows=read_csv(e3/'artifacts/data'/scene/'frames_04.csv');images={int(r['frame_index']):np.asarray(Image.open(r['rgb_path']).convert('RGB'))/255. for r in rows};data={}
 for i,fid in enumerate(d['dataset_frame_id'].astype(int)):
  data[fid]={'rgb':images[fid],'gt':d['gt_points_camera'][i],'depth':d['gt_depth'][i],'gtmask':d['gt_valid_mask'][i].astype(bool),
   'models':{'V-P':d['vp_points_camera'][i],'V-D':d['vd_points_camera'][i],'M-F':d['mf_points_camera'][i]},
   'masks':{'V-P':d['vp_valid_mask'][i].astype(bool),'V-D':d['vd_valid_mask'][i].astype(bool),'M-F':d['mf_valid_mask'][i].astype(bool)},
   'confs':{'V-P':d['vp_confidence'][i],'V-D':d['vd_confidence'][i],'M-F':d['mf_confidence'][i]},'timestamps':str(d['rgb_timestamp'][i])}
 return data
def load_p2_room():
 path=e3/'artifacts/conditions/room/P2/VGGT/fixed04/repeat01/artifacts/exports/vggt_pointmaps.npz';frames=read_csv(e3/'artifacts/data/room/frames_p2_vggt.csv');ass={r['rgb_timestamp']:r for r in read_csv(e3/'artifacts/data/room/frames_associated.csv')};v=np.load(path,allow_pickle=False);data={}
 for i,row in enumerate(frames):
  fid=int(row['frame_index']);g=ass[row['rgb_timestamp']];raw=np.asarray(Image.open(g['depth_path']),np.uint16);gd=raw.astype(np.float32)/DEPTH_SCALE;gm=np.isfinite(gd)&(gd>=.3)&(gd<=5.);gt=img_points(gd);gt[~gm]=np.nan
  src=v['source_image_shape_hw'][i];resize=v['resized_image_shape_hw'][i];pad=v['padding_tblr'][i]
  vp=remap_field(v['V_P'][i],src,resize,pad);vd=remap_field(v['V_D'][i],src,resize,pad);ex=v['extrinsic'][i].astype(np.float64);R,t=ex[:,:3],ex[:,3];vp=vp@R.T+t;vd=vd@R.T+t
  vm=remap_field(v['valid_mask'][i],src,resize,pad,True).astype(bool);confp=remap_field(v['V_P_conf'][i],src,resize,pad);confd=remap_field(v['depth_conf'][i],src,resize,pad)
  data[fid]={'rgb':np.asarray(Image.open(row['rgb_path']).convert('RGB'))/255.,'gt':gt,'depth':gd,'gtmask':gm,'models':{'V-P':vp,'V-D':vd},'masks':{'V-P':vm&np.isfinite(vp).all(-1),'V-D':vm&np.isfinite(vd).all(-1)},'confs':{'V-P':confp,'V-D':confd},'timestamps':row['rgb_timestamp']}
 return data

all_data={'desk':load_p1('desk'),'room':load_p1('room')};all_data['room_p2']=load_p2_room()
def summarize_case(case,d):
 x0,y0,x1,y1=case['roi_xyxy'];sl=np.s_[y0:y1,x0:x1];gm=d['gtmask'][sl];g=d['gt'][sl];valid_gt=gm&np.isfinite(g).all(-1);entry={**case,'rgb_timestamp':d['timestamps'],'gt_valid_roi_pixels':int(valid_gt.sum()),'models':{},'vggt_disagreement':None}
 for name,pts in d['models'].items():
  conf=d['confs'][name];mask=d['masks'][name];joint=valid_gt&mask[sl]&np.isfinite(pts[sl]).all(-1)&np.isfinite(conf[sl]);err=np.linalg.norm(pts[sl][joint]-g[joint],axis=-1);c=conf[sl][joint];predz=pts[sl][...,2][joint];gtz=d['depth'][sl][joint];n=len(err);base=float(np.mean(err>TAU)) if n else None
  m={'joint_pixels':n,'coverage_of_gt_valid_roi':n/valid_gt.sum() if valid_gt.sum() else None,'error_median_m':float(np.median(err)) if n else None,'error_p90_m':float(np.percentile(err,90)) if n else None,'outlier_ratio_at_5cm':base,'free_space_violation_ratio':float(np.mean(predz<gtz-.02)) if n else None,'confidence_median':float(np.median(c)) if n else None,'retention':[]}
  for q in (1.,.8,.6,.4):
   keep=np.argsort(-c)[:max(1,math.ceil(n*q))] if n else np.array([],int);e=err[keep];m['retention'].append({'fraction':q,'retained_count':len(keep),'coverage_of_gt_valid_roi':len(keep)/valid_gt.sum() if valid_gt.sum() else None,'outlier_ratio_at_5cm':float(np.mean(e>TAU)) if len(e) else None,'outlier_rate_reduction':(base-float(np.mean(e>TAU))) if len(e) and base is not None else None})
  entry['models'][name]=m
 if 'V-P' in d['models'] and 'V-D' in d['models']:
  common=valid_gt&d['masks']['V-P'][sl]&d['masks']['V-D'][sl]&np.isfinite(d['models']['V-P'][sl]).all(-1)&np.isfinite(d['models']['V-D'][sl]).all(-1)
  dis=np.linalg.norm(d['models']['V-P'][sl][common]-d['models']['V-D'][sl][common],axis=-1);evp=np.linalg.norm(d['models']['V-P'][sl][common]-g[common],axis=-1);evd=np.linalg.norm(d['models']['V-D'][sl][common]-g[common],axis=-1)
  entry['vggt_disagreement']={'pixels':len(dis),'mean_m':float(dis.mean()) if len(dis) else None,'median_m':float(np.median(dis)) if len(dis) else None,'spearman_vs_VP_error':float(spearmanr(dis,evp).statistic) if len(dis)>2 else None,'spearman_vs_VD_error':float(spearmanr(dis,evd).statistic) if len(dis)>2 else None}
 return entry
def draw_case(case,d):
 x0,y0,x1,y1=case['roi_xyxy'];rgb=d['rgb'];gtz=d['depth'];vp=d['models']['V-P'];vd=d['models']['V-D'];gm=d['gtmask'];vm=d['masks']['V-P']&d['masks']['V-D']
 evp=np.linalg.norm(vp-d['gt'],axis=-1);evd=np.linalg.norm(vd-d['gt'],axis=-1);dis=np.linalg.norm(vp-vd,axis=-1);confp=d['confs']['V-P'];confd=d['confs']['V-D']
 fig,axs=plt.subplots(2,4,figsize=(16,8),constrained_layout=True);axs=axs.ravel();axs[0].imshow(rgb);axs[0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='yellow',lw=2));axs[0].set_title('RGB + analyzed ROI')
 im=axs[1].imshow(np.where(gm,gtz,np.nan),cmap='turbo',vmin=.3,vmax=5);axs[1].set_title('GT depth (m)');fig.colorbar(im,ax=axs[1],shrink=.7)
 panels=[(axs[2],np.where(gm&d['masks']['V-P'],evp,np.nan),'V-P 3D error (m)'),(axs[3],np.where(gm&d['masks']['V-D'],evd,np.nan),'V-D 3D error (m)'),(axs[4],np.where(gm&vm,dis,np.nan),'||V-P - V-D|| (m)'),(axs[5],np.where(gm&d['masks']['V-P'],confp,np.nan),'V-P confidence'),(axs[6],np.where(gm&d['masks']['V-D'],confd,np.nan),'V-D confidence')]
 for ax,z,title in panels:
  vmax=max(.1,float(np.nanpercentile(z,95)) if np.isfinite(z).any() else 1);im=ax.imshow(z,cmap='magma',vmin=0,vmax=vmax);ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7)
 for ax in axs:
  if ax is not axs[0]:ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='cyan',lw=1))
  ax.set_xticks([]);ax.set_yticks([])
 if 'M-F' in d['models'] and d['masks']['M-F'].any():
  mf=d['models']['M-F'];em=np.linalg.norm(mf-d['gt'],axis=-1);z=np.where(gm&d['masks']['M-F'],em,np.nan);im=axs[7].imshow(z,cmap='magma',vmin=0,vmax=max(.1,float(np.nanpercentile(z,95))));axs[7].set_title('M-F error (m); fused keyframe');fig.colorbar(im,ax=axs[7],shrink=.7)
 else:axs[7].axis('off');axs[7].text(.05,.5,'M-F output absent\nfor this source frame',transform=axs[7].transAxes)
 fig.suptitle(f"{case['id']} — {case['category']} — frame {case['source_frame_index']} — geometric outlier: 3D error > 5 cm",fontsize=14);fig.savefig(figdir/f"{case['id']}_diagnostics.png",dpi=160);plt.close(fig)
def draw_cloud(case,d):
 x0,y0,x1,y1=case['roi_xyxy'];sl=np.s_[y0:y1,x0:x1];names=['GT','V-P','V-D','M-F'];allpts=[]
 for name in names:
  pts=d['gt'] if name=='GT' else d['models'].get(name)
  if pts is None:continue
  pts=pts[sl];mask=d['gtmask'][sl] if name=='GT' else d['gtmask'][sl]&d['masks'][name][sl];q=pts[mask&np.isfinite(pts).all(-1)]
  if len(q):allpts.append(q)
 if not allpts:return
 allp=np.concatenate(allpts);lo=np.nanpercentile(allp,1,axis=0);hi=np.nanpercentile(allp,99,axis=0);pad=np.maximum((hi-lo)*.05,.03);lo-=pad;hi+=pad
 fig=plt.figure(figsize=(15,4))
 for j,name in enumerate(names,1):
  ax=fig.add_subplot(1,4,j,projection='3d');ax.set_title(name+' (camera frame)');pts=d['gt'] if name=='GT' else d['models'].get(name)
  if pts is None:ax.text2D(.1,.5,'not exported',transform=ax.transAxes);continue
  pts=pts[sl];mask=d['gtmask'][sl] if name=='GT' else d['gtmask'][sl]&d['masks'][name][sl];mask &= np.isfinite(pts).all(-1);idx=np.flatnonzero(mask.ravel());idx=idx[::max(1,len(idx)//5000)];q=pts.reshape(-1,3)[idx];color=d['rgb'][sl].reshape(-1,3)[idx]
  ax.scatter(q[:,0],q[:,1],q[:,2],c=color,s=1,depthshade=False);ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2]);ax.set_xlabel('X');ax.set_ylabel('Y');ax.set_zlabel('Z');ax.view_init(elev=22,azim=-62)
 fig.suptitle(f"{case['id']} same camera-coordinate view (GT, V-P, V-D, M-F)");fig.tight_layout();fig.savefig(cloud_dir/f"{case['id']}_same_view_cloud.png",dpi=150);plt.close(fig)

case_results=[]
for case in cases:
 key='room_p2' if case['protocol']=='P2' else case['scene'];d=all_data[key][case['source_frame_index']];case_results.append(summarize_case(case,d));draw_case(case,d);draw_cloud(case,d)
# Pixel-level indicator correlations; these pixels are not independent scene replicates.
def rho(x,y):
 if not x or not y:return {'rho':None,'p':None,'n':0}
 x=np.concatenate(x);y=np.concatenate(y);r=spearmanr(x,y);return {'rho':float(r.statistic) if np.isfinite(r.statistic) else None,'p':float(r.pvalue) if np.isfinite(r.pvalue) else None,'n':len(x)}
correlations={};pooled=[]
for scene in ('desk','room'):
 ds=all_data[scene];dlist={k:[] for k in ('disVP','errVP','errVD','confVP','confVD')};clist={k:[] for k in ('V-P','V-D','M-F')}
 for d in ds.values():
  gm=d['gtmask'];ppts=d['models']['V-P'];dpts=d['models']['V-D'];common=gm&d['masks']['V-P']&d['masks']['V-D']&np.isfinite(ppts).all(-1)&np.isfinite(dpts).all(-1)
  if common.any():
   g=d['gt'][common];dis=np.linalg.norm(ppts[common]-dpts[common],axis=-1);ep=np.linalg.norm(ppts[common]-g,axis=-1);ed=np.linalg.norm(dpts[common]-g,axis=-1);dlist['disVP'].append(dis);dlist['errVP'].append(ep);dlist['errVD'].append(ed);dlist['confVP'].append(d['confs']['V-P'][common]);dlist['confVD'].append(d['confs']['V-D'][common]);pooled.append((dis,ep,ed))
  for name in clist:
   if name not in d['models']:continue
   mask=gm&d['masks'][name]&np.isfinite(d['models'][name]).all(-1);clist[name].append((d['confs'][name][mask],np.linalg.norm(d['models'][name][mask]-d['gt'][mask],axis=-1)))
 correlations[scene]={'VP_VD_disagreement_vs_VP_error':rho(dlist['disVP'],dlist['errVP']),'VP_VD_disagreement_vs_VD_error':rho(dlist['disVP'],dlist['errVD']),'confidence_vs_3D_error':{k:rho([x[0] for x in v],[x[1] for x in v]) for k,v in clist.items()}}
if pooled:correlations['pooled_desk_room']={'note':'pixel-level descriptive association; pixel samples are not independent scene replicates','disagreement_vs_VP_error':rho([x[0] for x in pooled],[x[1] for x in pooled]),'disagreement_vs_VD_error':rho([x[0] for x in pooled],[x[2] for x in pooled])}
# Fusion counts are measured per keyframe; current N=4 and N=32 samples are too small for inference.
fusion={}
for scene in ('desk','room'):
 for n,rep in ((4,3 if scene=='desk' else 1),(32,1)):
  cond=e3/'artifacts/conditions'/scene/'P1'/'C0'/f'N{n:02d}'/'MASt3R'/f'repeat{rep:02d}';npz=cond/'artifacts/exports/mast3r_mf_pointmaps.npz';rp=cond/'artifacts/metrics/mf_retention.json'
  if not npz.exists():continue
  m=np.load(npz,allow_pickle=False);ret=json.load(rp.open()) if rp.exists() else {'frame_metrics':[]};fusion[f'{scene}_N{n}']={'keyframe_count':len(m['dataset_frame_id']),'fusion_counts':m['fusion_count'].astype(int).tolist(),'per_keyframe_gt_metrics':ret['frame_metrics'],'status':'estimable' if len(ret['frame_metrics'])>=3 and len(set(m['fusion_count'].tolist()))>1 else 'not_estimable_too_few_or_constant_fusion_counts'}
# Long route: endpoint displacement and accumulated GT camera-center travel.
assoc=read_csv(e3/'artifacts/data/room/frames_associated.csv');poses={int(r['frame_index']):np.array(list(map(float,r['groundtruth_pose_xyzw'].split()))[:3]) for r in assoc if r['within_tolerance'].lower()=='true'};ordered=sorted(poses);path_m=sum(np.linalg.norm(poses[b]-poses[a]) for a,b in zip(ordered,ordered[1:]));endpoint=float(np.linalg.norm(poses[ordered[-1]]-poses[ordered[0]]));trajectory={'valid_pose_count':len(ordered),'first_frame_index':ordered[0],'last_frame_index':ordered[-1],'endpoint_displacement_m':endpoint,'cumulative_camera_center_path_m':float(path_m)}
# Tabular case metrics and retention tradeoff.
with (out/'case_metrics.csv').open('w',newline='') as f:
 cols=['id','category','scene','protocol','source_frame_index','rgb_timestamp','roi_xyxy','note','gt_valid_roi_pixels','model','joint_pixels','coverage_of_gt_valid_roi','error_median_m','error_p90_m','outlier_ratio_at_5cm','free_space_violation_ratio','confidence_median'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
 for c in case_results:
  for model,m in c['models'].items():w.writerow({**{k:c.get(k) for k in cols[:9]},'model':model,**{k:v for k,v in m.items() if k in cols[10:]}})
with (out/'retention_tradeoff.csv').open('w',newline='') as f:
 cols=['id','category','model','fraction','retained_count','coverage_of_gt_valid_roi','outlier_ratio_at_5cm','outlier_rate_reduction'];w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
 for c in case_results:
  for model,m in c['models'].items():
   for q in m['retention']:w.writerow({'id':c['id'],'category':c['category'],'model':model,**q})
result={'schema':'pointmap-comparison.e4.v1','run_stage':str(out.parent),'outlier_definition':{'camera_local_3d_error_m_gt':TAU,'depth_scale':DEPTH_SCALE,'gt_valid_depth_m':[.3,5.],'free_space_margin_m':.02},'case_selection':'manual scene tags; geometric outlier label is independent 5 cm GT point-error rule','cases':case_results,'uncertainty_correlations':correlations,'mast3r_fusion_count_relation':fusion,'room_p2_trajectory':trajectory,'figures':[str(x.relative_to(out)) for x in sorted(out.rglob('*.png'))]}
(out/'e4_summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'cases':len(case_results),'figures':len(result['figures']),'summary':str(out/'e4_summary.json'),'trajectory':trajectory},indent=2))
