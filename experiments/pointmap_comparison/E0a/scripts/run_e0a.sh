#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
default_e0_stage="$repo_root/results/pointmap_comparison/tum-freiburg1_desk-20261105-001/stages/E0"
e0_stage="${E0_STAGE_DIR:-$default_e0_stage}"
run_id="${RUN_ID:-tum-freiburg1_desk-E0a-$(date +%Y%m%d-%H%M%S)}"
stage_dir="$repo_root/results/pointmap_comparison/$run_id/stages/E0a"
input_csv="$stage_dir/artifacts/data/frames_e0.csv"
preview_dir="$stage_dir/artifacts/data/tum/rgbd_dataset_freiburg1_desk_E0a"
work_dir="$stage_dir/artifacts/demo_outputs/mast3r_work"
overlay_main="$stage_dir/artifacts/source_overlay/main_e0a.py"
if [[ -f "$e0_stage/artifacts/data/frames_e0.csv" ]]; then
  input_source="$e0_stage/artifacts/data/frames_e0.csv"
  default_source="$e0_stage/artifacts/sources/mast3r_slam"
  checkpoint_dir="$e0_stage/artifacts/checkpoints"
  mast3r_model_dir="$checkpoint_dir/mast3r_official_hf"
  retrieval_dir="$checkpoint_dir/mast3r_retrieval_mirror"
  default_python="$e0_stage/environment/venvs/mast3r_slam/bin/python"
else
  dataset_dir="${TUM_DATASET_DIR:-$repo_root/data/pointmap_comparison/tum/rgbd_dataset_freiburg1_desk}"
  [[ -f "$dataset_dir/rgb.txt" ]] || { echo "TUM sequence missing; run scripts/pointmap_comparison/download_e0_resources.sh and extract it, or set E0_STAGE_DIR." >&2; exit 1; }
  input_source="$stage_dir/artifacts/data/source_sync/frames_e0.csv"
  python3 "$repo_root/experiments/pointmap_comparison/E0/scripts/prepare_e0_data.py" \
    --dataset-dir "$dataset_dir" --output-dir "$stage_dir/artifacts/data/source_sync"
  default_source="$repo_root/thirdparty/pointmap_comparison/mast3r_slam"
  checkpoint_dir="$repo_root/checkpoints/pointmap_comparison"
  mast3r_model_dir="$checkpoint_dir/mast3r"
  retrieval_dir="$checkpoint_dir/mast3r_retrieval"
  default_python="python"
fi
source_dir="${MAST3R_SOURCE_DIR:-$default_source}"
python_bin="${MAST3R_PYTHON:-$default_python}"

[[ -f "$source_dir/main.py" ]] || { echo "Pinned MASt3R-SLAM source checkout not found: $source_dir" >&2; exit 1; }
if [[ "$python_bin" == */* ]]; then
  [[ -x "$python_bin" ]] || { echo "MASt3R Python interpreter is not executable: $python_bin" >&2; exit 1; }
else
  command -v "$python_bin" >/dev/null || { echo "MASt3R Python interpreter is not on PATH: $python_bin" >&2; exit 1; }
fi
mkdir -p "$stage_dir/config" "$stage_dir/logs" "$stage_dir/artifacts/data" "$preview_dir/rgb" "$work_dir"
cp "$input_source" "$input_csv"

python3 - "$input_csv" "$preview_dir" <<'PY'
import csv, pathlib, sys
csv_path, preview_dir = map(pathlib.Path, sys.argv[1:])
preview_dir = preview_dir.resolve()
with csv_path.open(newline='') as f:
    rows = list(csv.DictReader(f))
lines = ['# E0a shared RGB preview subset', '# timestamp filename']
for row in rows:
    source = pathlib.Path(row['rgb_path']).resolve()
    target = preview_dir / 'rgb' / source.name
    target.symlink_to(source)
    lines.append(f"{row['rgb_timestamp']} rgb/{source.name}")
(preview_dir / 'rgb.txt').write_text('\n'.join(lines) + '\n')
PY

cp "$repo_root/experiments/pointmap_comparison/E0a/config/mast3r_e0a.yaml" "$stage_dir/config/"
ln -sfn "$source_dir/config" "$work_dir/config"
ln -sfn "$retrieval_dir" "$work_dir/checkpoints"

export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
export E0A_EXPORT_DIR="$stage_dir/artifacts/demo_outputs/mast3r_e0a"
export E0A_FRAME_MAP_CSV="$input_csv"
export MAST3R_MODEL_PATH="$mast3r_model_dir"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
"$python_bin" "$repo_root/experiments/pointmap_comparison/E0a/scripts/run_mast3r_e0a.py" \
  "$source_dir/main.py" "$overlay_main"
cd "$work_dir"
"$python_bin" "$overlay_main" --dataset "$preview_dir" --no-viz --save-as tum_e0a \
  --config "$repo_root/experiments/pointmap_comparison/E0a/config/mast3r_e0a.yaml" \
  2>&1 | tee "$stage_dir/logs/mast3r_e0a_inference.log"

"$python_bin" "$repo_root/experiments/pointmap_comparison/E0a/scripts/validate_e0a_export.py" \
  --input-frames "$input_csv" \
  --export "$E0A_EXPORT_DIR/mast3r_e0a_fused_keyframes.npz" \
  --output "$stage_dir/artifacts/validation/e0a_export_validation.json" \
  2>&1 | tee "$stage_dir/logs/e0a_validation.log"
printf 'E0a artifacts: %s\n' "$stage_dir/artifacts"
