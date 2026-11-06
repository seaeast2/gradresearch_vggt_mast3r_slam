#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
run_id="${1:?usage: run_e5e_force_keyframes.sh <run_id> [vggt-source-run-id]}"
vggt_source_run="${2:-hkairport01-E5e-20261106-001}"
[[ "$run_id" == hkairport01-E5e-* ]] || { echo "run_id must start with hkairport01-E5e-" >&2; exit 2; }
stage="$repo_root/results/pointmap_comparison/$run_id/stages/E5e"
[[ ! -e "$stage" ]] || { echo "Refusing to overwrite $stage" >&2; exit 2; }
e0_stage="$repo_root/results/pointmap_comparison/tum-freiburg1_desk-20261105-001/stages/E0"
e5_stage="$repo_root/results/pointmap_comparison/hkairport01-E5-20261105-002/stages/E5"
source_stage="$repo_root/results/pointmap_comparison/$vggt_source_run/stages/E5e"
vggt_python="$e0_stage/environment/venvs/vggt/bin/python"
mast3r_python="$e0_stage/environment/venvs/mast3r_slam/bin/python"
mast3r_source="$e0_stage/artifacts/sources/mast3r_slam"
[[ -x "$vggt_python" && -x "$mast3r_python" ]] || { echo "Pinned model environments are missing" >&2; exit 2; }
[[ -f "$source_stage/artifacts/exports/vggt/HKairport01_E5e_V-P_N04.ply" ]] || { echo "VGGT source output is missing: $source_stage" >&2; exit 2; }
mkdir -p "$stage"/{config,logs,artifacts/data,artifacts/raw/mast3r,artifacts/exports/mast3r,artifacts/exports/vggt,artifacts/outputs,artifacts/work/mast3r}
cp "$repo_root/experiments/pointmap_comparison/E5e/scripts/"*.py "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E0a/scripts/run_mast3r_e0a.py" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E1/scripts/export_mast3r_slam.py" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E5a/scripts/merge_direct_ply.py" "$stage/config/"
cp "$e5_stage/artifacts/data/HKairport01/mapping_frames.csv" "$stage/config/e5_mapping_frames.csv"
cp "$e5_stage/config/mast3r_e5.yaml" "$stage/config/mast3r_e5e.yaml"
cp -a "$source_stage/artifacts/exports/vggt/." "$stage/artifacts/exports/vggt/"
ln -s "$mast3r_source/config" "$stage/artifacts/work/mast3r/config"
ln -s "$e0_stage/artifacts/checkpoints/mast3r_retrieval_mirror" "$stage/artifacts/work/mast3r/checkpoints"
cat > "$stage/artifacts/data/mast3r_keyframe_policy.json" <<EOF
{
  "policy": "force_each_successful_tracking_frame",
  "input_filenames": ["000050.jpg", "000070.jpg", "000090.jpg", "000110.jpg"],
  "behavior": "The first input initializes the map. Each later input becomes a keyframe only after tracker pose estimation succeeds. Frames that request relocalization are not force-added by this override.",
  "vggt_reused_from_run_id": "$vggt_source_run"
}
EOF
mast3r_rc=99
finish_on_error() {
  local rc=$?
  if (( rc != 0 )); then
    "$vggt_python" "$stage/config/finalize_e5e.py" --repo-root "$repo_root" --stage "$stage" \
      --run-id "$run_id" --status failed --mast3r-exit-code "$mast3r_rc" --vggt-exit-code 0 \
      --keyframe-policy force_each_successful_tracking_frame --vggt-source-run-id "$vggt_source_run" || true
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
if not torch.cuda.is_available(): raise SystemExit('E5e must execute in the host CUDA environment')
PY

"$vggt_python" "$stage/config/prepare_e5e.py" \
  --mapping-csv "$stage/config/e5_mapping_frames.csv" \
  --selected-csv "$stage/artifacts/data/frames_e5e.csv" \
  --dataset "$stage/artifacts/data/tum/rgbd_dataset_freiburg1_HKairport01_E5e" \
  --summary "$stage/artifacts/data/input_summary.json" > "$stage/logs/prepare.log" 2>&1
cat "$stage/logs/prepare.log"

export MAST3R_MODEL_PATH="$e0_stage/artifacts/checkpoints/mast3r_official_hf"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PYTHONPATH="$mast3r_source${PYTHONPATH:+:$PYTHONPATH}"
export E0A_EXPORT_DIR="$stage/artifacts/raw/mast3r"
export E0A_FRAME_MAP_CSV="$stage/artifacts/data/frames_e5e.csv"
work="$stage/artifacts/work/mast3r"
base_main="$work/main_e5e_base.py"
patched_main="$work/main_e5e.py"
"$mast3r_python" - "$mast3r_source/main.py" "$base_main" <<'PY'
import pathlib,sys
source=pathlib.Path(sys.argv[1]).read_text()
old='    keyframes = SharedKeyframes(manager, h, w, buffer=len(dataset))\n'
new='    # E5e forced-keyframe run: keep one slot for each selected input frame.\n    keyframes = SharedKeyframes(manager, h, w, buffer=min(len(dataset), 4))\n'
if source.count(old)!=1: raise SystemExit('Pinned MASt3R source no longer matches expected keyframe allocation point')
pathlib.Path(sys.argv[2]).write_text(source.replace(old,new,1))
PY
"$mast3r_python" "$stage/config/run_mast3r_e0a.py" "$base_main" "$patched_main" \
  > "$stage/logs/build_mast3r_overlay.log" 2>&1
"$mast3r_python" - "$patched_main" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]); source=p.read_text()
old='''            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
'''
new='''            add_new_kf, match_info, try_reloc = tracker.track(frame)
            # E5e: after successful pose tracking, retain this selected input as a keyframe.
            if not try_reloc:
                add_new_kf = True
            if try_reloc:
'''
if source.count(old)!=1: raise SystemExit('Pinned MASt3R tracking block did not match expected patch point')
p.write_text(source.replace(old,new,1))
PY

cd "$work"
set +e
/usr/bin/time -v "$mast3r_python" "$patched_main" \
  --dataset "$stage/artifacts/data/tum/rgbd_dataset_freiburg1_HKairport01_E5e" \
  --no-viz --save-as "$run_id" --config "$stage/config/mast3r_e5e.yaml" \
  > "$stage/logs/mast3r_slam.log" 2>&1
mast3r_rc=$?
set -e
if [[ -d "$work/logs/$run_id" ]]; then
  mkdir -p "$stage/artifacts/outputs/$run_id"
  mv "$work/logs/$run_id"/* "$stage/artifacts/outputs/$run_id/"
fi
export_rc=0
if [[ -f "$stage/artifacts/raw/mast3r/mast3r_e0a_fused_keyframes.npz" ]]; then
  "$mast3r_python" "$stage/config/export_mast3r_slam.py" \
    --hook-export "$stage/artifacts/raw/mast3r/mast3r_e0a_fused_keyframes.npz" \
    --frame-list "$stage/artifacts/data/frames_e5e.csv" \
    --output-dir "$stage/artifacts/exports/mast3r" > "$stage/logs/export_mast3r.log" 2>&1 || export_rc=$?
  if (( export_rc == 0 )); then
    shopt -s nullglob
    mast3r_plys=("$stage/artifacts/exports/mast3r"/*_M-F.ply)
    if ((${#mast3r_plys[@]})); then
      "$vggt_python" "$stage/config/merge_direct_ply.py" --inputs "${mast3r_plys[@]}" \
        --output "$stage/artifacts/exports/mast3r/HKairport01_E5e_M-F_N04.ply" \
        > "$stage/logs/merge_mast3r.log" 2>&1 || export_rc=$?
    else
      export_rc=5
    fi
  fi
else
  export_rc=4
fi
if (( mast3r_rc == 0 && export_rc != 0 )); then mast3r_rc=$export_rc; fi

status=completed
if (( mast3r_rc != 0 )) || [[ ! -f "$stage/artifacts/exports/mast3r/HKairport01_E5e_M-F_N04.ply" ]]; then status=partial; fi
if [[ ! -f "$stage/artifacts/exports/mast3r/HKairport01_E5e_M-F_N04.ply" ]]; then status=failed; fi
if (( mast3r_rc == 0 )); then mast3r_rc=0; fi
"$vggt_python" "$stage/config/validate_e5e.py" --stage "$stage" --require-mast3r-all-inputs \
  > "$stage/logs/validate.log" 2>&1 || { status=partial; cat "$stage/logs/validate.log"; }
"$vggt_python" "$stage/config/finalize_e5e.py" --repo-root "$repo_root" --stage "$stage" \
  --run-id "$run_id" --status "$status" --mast3r-exit-code "$mast3r_rc" --vggt-exit-code 0 \
  --keyframe-policy force_each_successful_tracking_frame --vggt-source-run-id "$vggt_source_run" \
  > "$stage/logs/finalize.log" 2>&1
cat "$stage/logs/finalize.log"
if [[ "$status" != completed ]]; then exit 1; fi
echo "E5e forced-keyframe $status: $stage"
