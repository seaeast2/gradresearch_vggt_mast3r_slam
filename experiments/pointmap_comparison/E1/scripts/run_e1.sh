#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
e0_run="${E0_RUN_ID:-tum-freiburg1_desk-20261105-001}"
e0a_run="${E0A_RUN_ID:-tum-freiburg1_desk-20261105-005}"
e0_stage="$repo_root/results/pointmap_comparison/$e0_run/stages/E0"
e0a_stage="$repo_root/results/pointmap_comparison/$e0a_run/stages/E0a"
run_id="${RUN_ID:-tum-freiburg1_desk-E1-$(date +%Y%m%d-%H%M%S)}"
stage="$repo_root/results/pointmap_comparison/$run_id/stages/E1"
[[ -d "$e0_stage" ]] || { echo "Missing E0 stage: $e0_stage" >&2; exit 1; }
[[ -d "$e0a_stage" ]] || { echo "Missing E0a stage: $e0a_stage" >&2; exit 1; }
[[ ! -e "$stage" ]] || { echo "Refusing to overwrite existing E1 stage: $stage" >&2; exit 1; }
mkdir -p "$stage/config" "$stage/logs" "$stage/artifacts/data" "$stage/artifacts/exports/vggt" "$stage/artifacts/exports/mast3r" "$stage/artifacts/raw/mast3r_hook" "$stage/artifacts/work/mast3r"
finalizer="$repo_root/experiments/pointmap_comparison/E1/scripts/finalize_e1.py"
python3 "$finalizer" --stage "$stage" --run-id "$run_id" --e0-stage "$e0_stage" --e0a-stage "$e0a_stage" --status running
finish() {
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    python3 "$finalizer" --stage "$stage" --run-id "$run_id" --e0-stage "$e0_stage" --e0a-stage "$e0a_stage" --status failed --exit-code "$rc" || true
  fi
}
trap finish EXIT

cp "$e0_stage/artifacts/data/frames_e0.csv" "$stage/config/frames_e1.csv"
cp "$repo_root/experiments/pointmap_comparison/E0a/config/mast3r_e0a.yaml" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E1/scripts/"*.py "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E1/scripts/run_e1.sh" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E0a/scripts/run_mast3r_e0a.py" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E1/README.md" "$stage/config/README.md"
cp "$e0_stage/artifacts/sources/vggt/vggt/utils/load_fn.py" "$stage/config/vggt_load_fn.py"
cp "$e0_stage/artifacts/sources/mast3r_slam/mast3r_slam/mast3r_utils.py" "$stage/config/mast3r_utils.py"
python3 - "$stage/config/frames_e1.csv" "$stage/artifacts/data/tum/rgbd_dataset_freiburg1_desk_E1" <<'PY'
import csv, pathlib, sys
csv_path, root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
root.mkdir(parents=True, exist_ok=True)
with csv_path.open(newline='') as stream: rows=list(csv.DictReader(stream))
lines=['# E1 shared input frames','# timestamp filename']
for row in rows:
    src=pathlib.Path(row['rgb_path']).resolve(); target=root/'rgb'/src.name
    target.parent.mkdir(parents=True,exist_ok=True); target.symlink_to(src)
    lines.append(f"{row['rgb_timestamp']} rgb/{src.name}")
(root/'rgb.txt').write_text('\n'.join(lines)+'\n')
PY
vggt_python="$e0_stage/environment/venvs/vggt/bin/python"
"$vggt_python" - <<'PY' > "$stage/logs/host_environment.txt" 2>&1
import json, subprocess, torch
print(subprocess.run(['nvidia-smi'],check=False,text=True,capture_output=True).stdout)
print(json.dumps({'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),
                  'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},indent=2))
PY

"$vggt_python" "$repo_root/experiments/pointmap_comparison/E1/scripts/export_vggt.py" \
  --frame-list "$stage/config/frames_e1.csv" --checkpoint-dir "$e0_stage/artifacts/checkpoints/vggt" \
  --output-dir "$stage/artifacts/exports/vggt" 2>&1 | tee "$stage/logs/export_vggt.log"

mast3r_source="$e0_stage/artifacts/sources/mast3r_slam"
mast3r_python="$e0_stage/environment/venvs/mast3r_slam/bin/python"
work="$stage/artifacts/work/mast3r"
ln -s "$mast3r_source/config" "$work/config"
ln -s "$e0_stage/artifacts/checkpoints/mast3r_retrieval_mirror" "$work/checkpoints"
export E0A_EXPORT_DIR="$stage/artifacts/raw/mast3r_hook"
export E0A_FRAME_MAP_CSV="$stage/config/frames_e1.csv"
export MAST3R_MODEL_PATH="$e0_stage/artifacts/checkpoints/mast3r_official_hf"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PYTHONPATH="$mast3r_source${PYTHONPATH:+:$PYTHONPATH}"
overlay="$stage/artifacts/work/main_e1.py"
"$mast3r_python" "$repo_root/experiments/pointmap_comparison/E0a/scripts/run_mast3r_e0a.py" \
  "$mast3r_source/main.py" "$overlay" 2>&1 | tee "$stage/logs/build_mast3r_overlay.log"
cd "$work"
"$mast3r_python" "$overlay" --dataset "$stage/artifacts/data/tum/rgbd_dataset_freiburg1_desk_E1" \
  --no-viz --save-as tum_e1 --config "$stage/config/mast3r_e0a.yaml" \
  2>&1 | tee "$stage/logs/export_mast3r_slam.log"
"$mast3r_python" "$repo_root/experiments/pointmap_comparison/E1/scripts/export_mast3r_slam.py" \
  --hook-export "$E0A_EXPORT_DIR/mast3r_e0a_fused_keyframes.npz" \
  --frame-list "$stage/config/frames_e1.csv" --output-dir "$stage/artifacts/exports/mast3r" \
  2>&1 | tee "$stage/logs/normalize_mast3r_export.log"
"$vggt_python" "$repo_root/experiments/pointmap_comparison/E1/scripts/validate_e1_exports.py" \
  --vggt "$stage/artifacts/exports/vggt/vggt_pointmaps.npz" \
  --mast3r "$stage/artifacts/exports/mast3r/mast3r_mf_pointmaps.npz" \
  --frame-list "$stage/config/frames_e1.csv" --output "$stage/artifacts/validation/e1_export_validation.json" \
  2>&1 | tee "$stage/logs/validate_e1_exports.log"
python3 "$finalizer" --stage "$stage" --run-id "$run_id" --e0-stage "$e0_stage" --e0a-stage "$e0a_stage" --status completed
trap - EXIT
echo "E1 completed: $stage"
