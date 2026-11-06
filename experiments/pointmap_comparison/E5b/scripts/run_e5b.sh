#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
run_id="${1:?usage: run_e5b.sh <run_id>}"
[[ "$run_id" == hkairport01-E5b-* ]] || { echo "run_id must start with hkairport01-E5b-" >&2; exit 2; }
stage="$repo_root/results/pointmap_comparison/$run_id/stages/E5b"
[[ ! -e "$stage" ]] || { echo "Refusing to overwrite $stage" >&2; exit 2; }
e0_stage="$repo_root/results/pointmap_comparison/tum-freiburg1_desk-20261105-001/stages/E0"
e5_stage="$repo_root/results/pointmap_comparison/hkairport01-E5-20261105-002/stages/E5"
vggt_python="$e0_stage/environment/venvs/vggt/bin/python"
mast3r_python="$e0_stage/environment/venvs/mast3r_slam/bin/python"
mast3r_source="$e0_stage/artifacts/sources/mast3r_slam"
[[ -x "$vggt_python" && -x "$mast3r_python" ]] || { echo "Pinned model environments are missing" >&2; exit 2; }
mkdir -p "$stage"/{config,logs,artifacts/data,artifacts/raw/mast3r,artifacts/exports/mast3r,artifacts/exports/vggt,artifacts/exports/aligned,artifacts/work/mast3r}
cp "$repo_root/experiments/pointmap_comparison/E5b/scripts/"*.py "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E5b/scripts/run_e5b.sh" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E0a/scripts/run_mast3r_e0a.py" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E1/scripts/export_mast3r_slam.py" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E3/scripts/export_vggt_e3.py" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E5/scripts/finalize_e5.py" "$stage/config/"
cp "$e5_stage/artifacts/data/HKairport01/mapping_frames.csv" "$stage/config/e5_mapping_frames.csv"
cp "$e5_stage/artifacts/outputs/hkairport01-E5-20261105-002/rgbd_dataset_freiburg1_HKairport01_E5.txt" "$stage/config/e5_trajectory.txt"
cp "$e5_stage/config/mast3r_e5.yaml" "$stage/config/"
ln -s "$mast3r_source/config" "$stage/artifacts/work/mast3r/config"
ln -s "$e0_stage/artifacts/checkpoints/mast3r_retrieval_mirror" "$stage/artifacts/work/mast3r/checkpoints"

"$vggt_python" - <<'PY' > "$stage/logs/host_environment.txt" 2>&1
import json,subprocess,torch
print(subprocess.run(['nvidia-smi'],check=False,text=True,capture_output=True).stdout)
print(json.dumps({'torch':torch.__version__,'torch_cuda':torch.version.cuda,
 'cuda_available':torch.cuda.is_available(),
 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
 'compute_capability':list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None},indent=2))
if not torch.cuda.is_available(): raise SystemExit('E5b must execute in the host CUDA environment')
PY

trap 'rc=$?; if (( rc != 0 )); then "$vggt_python" "$stage/config/finalize_e5b.py" --repo-root "$repo_root" --stage "$stage" --run-id "$run_id" --status failed --exit-code "$rc" || true; fi' EXIT

"$vggt_python" "$stage/config/prepare_e5b.py" \
  --mapping-csv "$stage/config/e5_mapping_frames.csv" \
  --trajectory "$stage/config/e5_trajectory.txt" \
  --dataset "$stage/artifacts/data/tum/rgbd_dataset_freiburg1_HKairport01_E5b" \
  --all-frames-csv "$stage/artifacts/data/mapping_frames.csv" \
  --baseline-keys-csv "$stage/artifacts/data/e5_baseline_keyframes.csv" \
  --summary "$stage/artifacts/data/input_summary.json" > "$stage/logs/prepare.log" 2>&1
cat "$stage/logs/prepare.log"

export MAST3R_MODEL_PATH="$e0_stage/artifacts/checkpoints/mast3r_official_hf"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PYTHONPATH="$mast3r_source${PYTHONPATH:+:$PYTHONPATH}"
export E0A_EXPORT_DIR="$stage/artifacts/raw/mast3r"
export E0A_FRAME_MAP_CSV="$stage/artifacts/data/mapping_frames.csv"
base_main="$stage/artifacts/work/mast3r/main_e5_base.py"
patched_main="$stage/artifacts/work/mast3r/main_e5b.py"
"$mast3r_python" - "$mast3r_source/main.py" "$base_main" <<'PY'
import pathlib,sys
source=pathlib.Path(sys.argv[1]).read_text()
old='    keyframes = SharedKeyframes(manager, h, w, buffer=len(dataset))\n'
new='    # E5b: preserve E5 buffer policy on the 8 GiB host GPU.\n    keyframes = SharedKeyframes(manager, h, w, buffer=min(len(dataset), 32))\n'
if source.count(old)!=1: raise SystemExit('Pinned MASt3R source no longer matches the E5 keyframe allocation point')
pathlib.Path(sys.argv[2]).write_text(source.replace(old,new,1))
PY
"$mast3r_python" "$stage/config/run_mast3r_e0a.py" "$base_main" "$patched_main" > "$stage/logs/build_mast3r_overlay.log" 2>&1

work="$stage/artifacts/work/mast3r"
monitor_pid=""
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 500 \
  > "$stage/logs/mast3r_gpu_memory.csv" 2> "$stage/logs/nvidia_smi.log" & monitor_pid=$!
started="$(date +%s.%N)"
cd "$work"
set +e
/usr/bin/time -v "$mast3r_python" "$patched_main" \
  --dataset "$stage/artifacts/data/tum/rgbd_dataset_freiburg1_HKairport01_E5b" \
  --no-viz --save-as "$run_id" --config "$stage/config/mast3r_e5.yaml" \
  > "$stage/logs/mast3r_slam.log" 2>&1
mast3r_rc=$?
set -e
ended="$(date +%s.%N)"
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
if (( mast3r_rc != 0 )); then cat "$stage/logs/mast3r_slam.log"; exit "$mast3r_rc"; fi
run_output="$work/logs/$run_id"
[[ -d "$run_output" ]] || { echo "MASt3R-SLAM output directory missing: $run_output" >&2; exit 3; }
mkdir -p "$stage/artifacts/outputs/$run_id"
mv "$run_output"/* "$stage/artifacts/outputs/$run_id/"

"$mast3r_python" "$stage/config/export_mast3r_slam.py" \
  --hook-export "$stage/artifacts/raw/mast3r/mast3r_e0a_fused_keyframes.npz" \
  --frame-list "$stage/artifacts/data/mapping_frames.csv" \
  --output-dir "$stage/artifacts/exports/mast3r" \
  > "$stage/logs/normalize_mast3r.log" 2>&1
cat "$stage/logs/normalize_mast3r.log"
"$vggt_python" "$stage/config/prepare_vggt_keyframes_e5b.py" \
  --mast3r-npz "$stage/artifacts/exports/mast3r/mast3r_mf_pointmaps.npz" \
  --baseline-csv "$stage/artifacts/data/e5_baseline_keyframes.csv" \
  --all-frames-csv "$stage/artifacts/data/mapping_frames.csv" \
  --output-csv "$stage/artifacts/data/matched_keyframes.csv" \
  --summary "$stage/artifacts/data/keyframe_match_summary.json" \
  > "$stage/logs/match_keyframes.log" 2>&1
cat "$stage/logs/match_keyframes.log"

"$vggt_python" - "$stage/artifacts/data/matched_keyframes.csv" "$stage/artifacts/data/vggt_batches" <<'PY'
import csv,pathlib,sys
rows=list(csv.DictReader(open(sys.argv[1],newline=''))); root=pathlib.Path(sys.argv[2]);root.mkdir(parents=True,exist_ok=True)
for start in range(0,len(rows),4):
    group=rows[start:start+4]; path=root/f'batch_{start//4+1:02d}.csv'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(group[0]));writer.writeheader();writer.writerows(group)
print(f'Prepared {(len(rows)+3)//4} batches for {len(rows)} matched keyframes (max 4 views/batch)')
PY

batch_dir="$stage/artifacts/data/vggt_batches"
for batch_csv in "$batch_dir"/batch_*.csv; do
  batch_name="$(basename "$batch_csv" .csv)"
  out="$stage/artifacts/exports/vggt/$batch_name"
  mkdir -p "$out"
  "$vggt_python" "$stage/config/export_vggt_e3.py" \
    --frame-list "$batch_csv" \
    --checkpoint-dir "$e0_stage/artifacts/checkpoints/vggt" \
    --output-dir "$out" > "$stage/logs/vggt_${batch_name}.log" 2>&1
done

"$vggt_python" "$stage/config/align_e5b_clouds.py" \
  --mast3r-npz "$stage/artifacts/exports/mast3r/mast3r_mf_pointmaps.npz" \
  --vggt-batch-root "$stage/artifacts/exports/vggt" \
  --output-dir "$stage/artifacts/exports/aligned" \
  > "$stage/logs/align_clouds.log" 2>&1
cat "$stage/logs/align_clouds.log"
"$vggt_python" - "$stage" "$run_id" "$started" "$ended" <<'PY'
import json,pathlib,sys
stage=pathlib.Path(sys.argv[1]);run_id=sys.argv[2]
inp=json.loads((stage/'artifacts/data/input_summary.json').read_text())
match=json.loads((stage/'artifacts/data/keyframe_match_summary.json').read_text())
cloud=json.loads((stage/'artifacts/exports/aligned/cloud_summary.json').read_text())
result={'schema':'pointmap-comparison.e5b-summary.v1','run_id':run_id,'status':'completed',
 'execution_context':'host','mast3r_wall_time_seconds':float(sys.argv[4])-float(sys.argv[3]),
 'mapping_frame_count':inp['mapping_frame_count'],'matched_keyframe_count':match['replayed_mast3r_keyframe_count'],
 'baseline_keyframes_exact_match':match['exact_keyframe_id_match'],
 'frame_indices':match['frame_indices'],'timestamps':match['timestamps'],
 'mast3r_mf_vertices':cloud['mast3r_mf_vertices'],'vggt_vp_aligned_vertices':cloud['vggt_vp_aligned_vertices'],
 'shared_pose_frame':cloud['shared_pose_frame'],'accuracy_claim':'qualitative; no HKairport01 geometry GT',
 'outputs':cloud['outputs']}
(stage/'artifacts/e5b_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY
"$vggt_python" "$stage/config/finalize_e5b.py" --repo-root "$repo_root" --stage "$stage" \
  --run-id "$run_id" --status completed --exit-code 0 > "$stage/logs/finalize.log" 2>&1
# Re-hash after finalize.log has received the finalizer's stdout.
"$vggt_python" "$stage/config/finalize_e5b.py" --repo-root "$repo_root" --stage "$stage" \
  --run-id "$run_id" --status completed --exit-code 0 > /dev/null
trap - EXIT
cat "$stage/logs/finalize.log"
echo "E5b completed: $stage"
