#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
run_id="${1:?usage: run_e5d.sh <run_id>}"
[[ "$run_id" == hkairport01-E5d-* ]] || { echo "run_id must start with hkairport01-E5d-" >&2; exit 2; }
stage="$repo_root/results/pointmap_comparison/$run_id/stages/E5d"
[[ ! -e "$stage" ]] || { echo "Refusing to overwrite $stage" >&2; exit 2; }
e0_stage="$repo_root/results/pointmap_comparison/tum-freiburg1_desk-20261105-001/stages/E0"
e5_stage="$repo_root/results/pointmap_comparison/hkairport01-E5-20261105-002/stages/E5"
vggt_python="$e0_stage/environment/venvs/vggt/bin/python"
mast3r_python="$e0_stage/environment/venvs/mast3r_slam/bin/python"
mast3r_source="$e0_stage/artifacts/sources/mast3r_slam"
[[ -x "$vggt_python" && -x "$mast3r_python" ]] || { echo "Pinned model environments are missing" >&2; exit 2; }
mkdir -p "$stage"/{config,logs,artifacts/data,artifacts/raw/mast3r,artifacts/exports/mast3r,artifacts/exports/vggt,artifacts/outputs,artifacts/work/mast3r}
cp "$repo_root/experiments/pointmap_comparison/E5d/scripts/"*.py "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E5d/scripts/run_e5d.sh" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E0a/scripts/run_mast3r_e0a.py" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E1/scripts/export_mast3r_slam.py" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E3/scripts/export_vggt_e3.py" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E5a/scripts/merge_direct_ply.py" "$stage/config/"
cp "$e5_stage/artifacts/data/HKairport01/mapping_frames.csv" "$stage/config/e5_mapping_frames.csv"
cp "$e5_stage/config/mast3r_e5.yaml" "$stage/config/mast3r_e5d.yaml"
ln -s "$mast3r_source/config" "$stage/artifacts/work/mast3r/config"
ln -s "$e0_stage/artifacts/checkpoints/mast3r_retrieval_mirror" "$stage/artifacts/work/mast3r/checkpoints"
mast3r_rc=99
vggt_rc=99
finish_on_error() {
  local rc=$?
  if (( rc != 0 )); then
    "$vggt_python" "$stage/config/finalize_e5d.py" --repo-root "$repo_root" --stage "$stage" \
      --run-id "$run_id" --status failed --mast3r-exit-code "$mast3r_rc" --vggt-exit-code "$vggt_rc" || true
  fi
}
trap finish_on_error EXIT

"$vggt_python" - <<'PY' > "$stage/logs/host_environment.txt" 2>&1
import json,subprocess,torch
print(subprocess.run(['nvidia-smi'],check=False,text=True,capture_output=True).stdout)
print(json.dumps({'torch':torch.__version__,'torch_cuda':torch.version.cuda,
 'cuda_available':torch.cuda.is_available(),
 'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
 'compute_capability':list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None},indent=2))
if not torch.cuda.is_available(): raise SystemExit('E5d must execute in the host CUDA environment')
PY

"$vggt_python" "$stage/config/prepare_e5d.py" \
  --mapping-csv "$stage/config/e5_mapping_frames.csv" \
  --selected-csv "$stage/artifacts/data/frames_e5d.csv" \
  --dataset "$stage/artifacts/data/tum/rgbd_dataset_freiburg1_HKairport01_E5d" \
  --summary "$stage/artifacts/data/input_summary.json" > "$stage/logs/prepare.log" 2>&1
cat "$stage/logs/prepare.log"

export MAST3R_MODEL_PATH="$e0_stage/artifacts/checkpoints/mast3r_official_hf"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PYTHONPATH="$mast3r_source${PYTHONPATH:+:$PYTHONPATH}"
export E0A_EXPORT_DIR="$stage/artifacts/raw/mast3r"
export E0A_FRAME_MAP_CSV="$stage/artifacts/data/frames_e5d.csv"
work="$stage/artifacts/work/mast3r"
base_main="$work/main_e5d_base.py"
patched_main="$work/main_e5d.py"
"$mast3r_python" - "$mast3r_source/main.py" "$base_main" <<'PY'
import pathlib,sys
source=pathlib.Path(sys.argv[1]).read_text()
old='    keyframes = SharedKeyframes(manager, h, w, buffer=len(dataset))\n'
new='    # E5d: only four input frames are resident in this matched-frame run.\n    keyframes = SharedKeyframes(manager, h, w, buffer=min(len(dataset), 4))\n'
if source.count(old)!=1: raise SystemExit('Pinned MASt3R source no longer matches the expected keyframe allocation point')
pathlib.Path(sys.argv[2]).write_text(source.replace(old,new,1))
PY
"$mast3r_python" "$stage/config/run_mast3r_e0a.py" "$base_main" "$patched_main" \
  > "$stage/logs/build_mast3r_overlay.log" 2>&1

started="$(date +%s.%N)"
cd "$work"
set +e
/usr/bin/time -v "$mast3r_python" "$patched_main" \
  --dataset "$stage/artifacts/data/tum/rgbd_dataset_freiburg1_HKairport01_E5d" \
  --no-viz --save-as "$run_id" --config "$stage/config/mast3r_e5d.yaml" \
  > "$stage/logs/mast3r_slam.log" 2>&1
mast3r_rc=$?
set -e
if [[ -d "$work/logs/$run_id" ]]; then
  mkdir -p "$stage/artifacts/outputs/$run_id"
  mv "$work/logs/$run_id"/* "$stage/artifacts/outputs/$run_id/"
fi
if [[ -f "$stage/artifacts/raw/mast3r/mast3r_e0a_fused_keyframes.npz" ]]; then
  set +e
  "$mast3r_python" "$stage/config/export_mast3r_slam.py" \
    --hook-export "$stage/artifacts/raw/mast3r/mast3r_e0a_fused_keyframes.npz" \
    --frame-list "$stage/artifacts/data/frames_e5d.csv" \
    --output-dir "$stage/artifacts/exports/mast3r" > "$stage/logs/export_mast3r.log" 2>&1
  mast3r_export_rc=$?
  set -e
  if (( mast3r_export_rc == 0 )); then
    shopt -s nullglob
    mast3r_plys=("$stage/artifacts/exports/mast3r"/*_M-F.ply)
    if ((${#mast3r_plys[@]})); then
      "$vggt_python" "$stage/config/merge_direct_ply.py" --inputs "${mast3r_plys[@]}" \
        --output "$stage/artifacts/exports/mast3r/HKairport01_E5d_M-F_N04.ply" \
        > "$stage/logs/merge_mast3r.log" 2>&1 || mast3r_export_rc=$?
    fi
  fi
  if (( mast3r_export_rc != 0 )); then mast3r_rc="$mast3r_export_rc"; fi
else
  [[ $mast3r_rc -ne 0 ]] || mast3r_rc=4
fi
set -e

set +e
"$vggt_python" "$stage/config/export_vggt_e3.py" \
  --frame-list "$stage/artifacts/data/frames_e5d.csv" \
  --checkpoint-dir "$e0_stage/artifacts/checkpoints/vggt" \
  --output-dir "$stage/artifacts/exports/vggt" > "$stage/logs/vggt.log" 2>&1
vggt_rc=$?
set -e
if (( vggt_rc == 0 )); then
  shopt -s nullglob
  vggt_plys=("$stage/artifacts/exports/vggt"/*_V-P.ply)
  "$vggt_python" "$stage/config/merge_direct_ply.py" --inputs "${vggt_plys[@]}" \
    --output "$stage/artifacts/exports/vggt/HKairport01_E5d_V-P_N04.ply" \
    > "$stage/logs/merge_vggt.log" 2>&1 || vggt_rc=$?
fi
ended="$(date +%s.%N)"

status="completed"
if (( mast3r_rc != 0 || vggt_rc != 0 )); then status="partial"; fi
if [[ ! -f "$stage/artifacts/exports/mast3r/HKairport01_E5d_M-F_N04.ply" && \
      ! -f "$stage/artifacts/exports/vggt/HKairport01_E5d_V-P_N04.ply" ]]; then status="failed"; fi
"$vggt_python" - "$stage" "$run_id" "$status" "$mast3r_rc" "$vggt_rc" "$started" "$ended" <<'PY'
import json,pathlib,sys
stage=pathlib.Path(sys.argv[1]);run_id,status=sys.argv[2:4];mast_rc,vggt_rc=map(int,sys.argv[4:6]);start,end=map(float,sys.argv[6:8])
inp=json.loads((stage/'artifacts/data/input_summary.json').read_text())
result={'schema':'pointmap-comparison.e5d-summary.v1','run_id':run_id,'status':status,
 'execution_context':'host','input_frame_count':4,'source_frame_indices':inp['source_frame_indices'],
 'candidate_grid_indices':inp['selected_candidate_grid_indices'],
 'timestamps':inp['timestamps'],'interval_seconds':inp['interval_seconds'],
 'mast3r_exit_code':mast_rc,'vggt_exit_code':vggt_rc,
 'total_wall_time_seconds':end-start,'coordinate_note':'Separate model-native PLYs; no cross-model alignment applied.',
 'accuracy_claim':'qualitative; HKairport01 has no geometry ground truth'}
meta=stage/'artifacts/exports/mast3r/metadata.json'
if meta.is_file():
 m=json.loads(meta.read_text());result['mast3r_exported_keyframes']=m['exported_keyframe_count'];result['mast3r_dataset_frame_ids']=m['dataset_frame_ids']
 import numpy as np
 with np.load(stage/'artifacts/exports/mast3r/mast3r_mf_pointmaps.npz',allow_pickle=False) as data:
  result['mast3r_source_frame_indices']=data['source_frame_index'].astype(int).tolist()
  result['mast3r_fusion_count']=data['fusion_count'].astype(int).tolist()
  result['mast3r_update_count']=data['update_count'].astype(int).tolist()
else:result['mast3r_exported_keyframes']=0;result['mast3r_dataset_frame_ids']=[]
vmeta=stage/'artifacts/exports/vggt/metadata.json'
if vmeta.is_file():result['vggt_exported_frames']=json.loads(vmeta.read_text()).get('count',0)
else:result['vggt_exported_frames']=0
(stage/'artifacts/e5d_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
PY

if (( mast3r_rc == 0 && vggt_rc == 0 )) && \
   [[ -f "$stage/artifacts/exports/mast3r/HKairport01_E5d_M-F_N04.ply" && -f "$stage/artifacts/exports/vggt/HKairport01_E5d_V-P_N04.ply" ]]; then status="completed"; fi
"$vggt_python" "$stage/config/validate_e5d.py" --stage "$stage" \
  > "$stage/logs/validate.log" 2>&1 || { status="partial"; cat "$stage/logs/validate.log"; }
"$vggt_python" "$stage/config/finalize_e5d.py" --repo-root "$repo_root" --stage "$stage" \
  --run-id "$run_id" --status "$status" --mast3r-exit-code "$mast3r_rc" --vggt-exit-code "$vggt_rc" \
  > "$stage/logs/finalize.log" 2>&1
"$vggt_python" "$stage/config/finalize_e5d.py" --repo-root "$repo_root" --stage "$stage" \
  --run-id "$run_id" --status "$status" --mast3r-exit-code "$mast3r_rc" --vggt-exit-code "$vggt_rc" \
  > /dev/null
cat "$stage/logs/finalize.log"
if [[ "$status" == "failed" ]]; then exit 1; fi
echo "E5d $status: $stage"
