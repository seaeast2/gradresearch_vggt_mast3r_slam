#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
e0_run="${E0_RUN_ID:-tum-freiburg1_desk-20261105-001}"
e1_run="${E1_RUN_ID:-tum-freiburg1_desk-20261105-014}"
e0_stage="$repo_root/results/pointmap_comparison/$e0_run/stages/E0"
e1_stage="$repo_root/results/pointmap_comparison/$e1_run/stages/E1"
run_id="${RUN_ID:-tum-freiburg1_desk-E2-$(date +%Y%m%d-%H%M%S)}"
stage="$repo_root/results/pointmap_comparison/$run_id/stages/E2"
[[ -d "$e0_stage/artifacts/data/rgbd_dataset_freiburg1_desk" ]] || { echo "Missing extracted E0 TUM dataset" >&2; exit 1; }
[[ -f "$e1_stage/artifacts/exports/vggt/vggt_pointmaps.npz" ]] || { echo "Missing completed E1 VGGT export" >&2; exit 1; }
[[ ! -e "$stage" ]] || { echo "Refusing to overwrite E2 stage: $stage" >&2; exit 1; }
mkdir -p "$stage/config" "$stage/logs" "$stage/artifacts/association" "$stage/artifacts/validation" "$stage/artifacts/metrics" "$stage/artifacts/alignment"
py="$e0_stage/environment/venvs/mast3r_slam/bin/python"
script="$repo_root/experiments/pointmap_comparison/E2/scripts"
python3 "$script/finalize_e2.py" --stage "$stage" --run-id "$run_id" --e0-stage "$e0_stage" --e1-stage "$e1_stage" --status running
finish(){ local rc=$?; if [[ $rc -ne 0 ]]; then "$py" "$script/finalize_e2.py" --stage "$stage" --run-id "$run_id" --e0-stage "$e0_stage" --e1-stage "$e1_stage" --status failed --exit-code "$rc" || true; fi; }
trap finish EXIT
cp "$e1_stage/config/frames_e1.csv" "$stage/config/frames_e1.csv"
cp "$e1_stage/artifacts/exports/vggt/vggt_pointmaps.npz" "$stage/config/"
cp "$e1_stage/artifacts/exports/mast3r/mast3r_mf_pointmaps.npz" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E2/scripts/"*.py "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E2/scripts/run_e2.sh" "$stage/config/"
"$py" "$script/associate_rgb_depth_pose.py" --dataset-dir "$e0_stage/artifacts/data/rgbd_dataset_freiburg1_desk" \
 --output-dir "$stage/artifacts/association" > "$stage/logs/associate.log" 2>&1
"$py" "$script/validate_synthetic.py" --output "$stage/artifacts/validation/synthetic_geometry_check.json" \
 > "$stage/logs/synthetic_geometry_check.log" 2>&1
"$py" - "$e1_stage" "$e0_stage/artifacts/data/frames_associated.csv" "$stage/config/frames_e1.csv" "$stage/artifacts/alignment" <<'PY'
import csv,json,math,pathlib,sys,numpy as np
e1=pathlib.Path(sys.argv[1]);assoc=list(csv.DictReader(open(sys.argv[2],newline='')))
frames=list(csv.DictReader(open(sys.argv[3],newline='')));out=pathlib.Path(sys.argv[4])
byts={r['rgb_timestamp']:r for r in assoc}
def quatrot(qx,qy,qz,qw):
 return np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
 [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
 [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
def gtcenter(r):
 f=list(map(float,r['groundtruth_pose_xyzw'].split()));return np.array(f[:3])
gt=[]
for row in frames:gt.append(gtcenter(byts[row['rgb_timestamp']]))
v=np.load(e1/'artifacts/exports/vggt/vggt_pointmaps.npz');ex=v['extrinsic'];vg=[]
for e in ex:vg.append(-e[:,:3].T@e[:,3])
m=np.load(e1/'artifacts/exports/mast3r/mast3r_mf_pointmaps.npz');mt=[];mg=[]
for fid,mat in zip(m['dataset_frame_id'],m['T_WC_matrix']):mt.append(mat[:3,3]);mg.append(gt[int(fid)])
for name,s,t in [('vggt',vg,gt),('mast3r_mf',mt,mg)]:
 (out/f'{name}_centers.json').write_text(json.dumps({'source_centers':np.asarray(s).tolist(),'target_centers':np.asarray(t).tolist()},indent=2)+'\n')
PY
for model in vggt mast3r_mf; do
 "$py" "$script/align_sim3.py" --input "$stage/artifacts/alignment/${model}_centers.json" \
  --output "$stage/artifacts/alignment/${model}_sim3.json" > "$stage/logs/align_${model}.log" 2>&1
done
"$py" "$script/evaluate_pixel.py" --vggt "$stage/config/vggt_pointmaps.npz" --mast3r "$stage/config/mast3r_mf_pointmaps.npz" \
 --frames "$stage/config/frames_e1.csv" --associations "$stage/artifacts/association/frames_associated.csv" \
 --output-dir "$stage/artifacts/metrics/pixel" > "$stage/logs/evaluate_pixel.log" 2>&1
"$py" "$script/evaluate_cloud.py" --input "$stage/artifacts/metrics/pixel/local_pointmaps_for_cloud_outlier.npz" \
 --output-dir "$stage/artifacts/metrics/cloud" > "$stage/logs/evaluate_cloud.log" 2>&1
"$py" "$script/evaluate_outliers.py" --input "$stage/artifacts/metrics/pixel/local_pointmaps_for_cloud_outlier.npz" \
 --output "$stage/artifacts/metrics/outliers.json" > "$stage/logs/evaluate_outliers.log" 2>&1
"$py" "$script/finalize_e2.py" --stage "$stage" --run-id "$run_id" --e0-stage "$e0_stage" --e1-stage "$e1_stage" --status completed
trap - EXIT
echo "E2 completed: $stage"
