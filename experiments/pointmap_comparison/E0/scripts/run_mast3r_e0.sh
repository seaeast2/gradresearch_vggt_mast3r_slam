#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
run_id="${RUN_ID:-tum-freiburg1_desk-$(date +%Y%m%d-%H%M%S)}"
python_bin="${MAST3R_PYTHON:-python}"
source_dir="$repo_root/thirdparty/pointmap_comparison/mast3r_slam"
checkpoint_dir="$repo_root/checkpoints/pointmap_comparison"
stage_dir="$repo_root/results/pointmap_comparison/$run_id/stages/E0"
preview_dir="$stage_dir/artifacts/data/tum/rgbd_dataset_freiburg1_desk_E0"
work_dir="$stage_dir/artifacts/demo_outputs/mast3r_work"

[[ -d "$source_dir" ]] || { echo "Run scripts/pointmap_comparison/checkout_e0_sources.sh first." >&2; exit 1; }
[[ -f "$preview_dir/rgb.txt" ]] || { echo "Prepare data with prepare_e0_data.py first." >&2; exit 1; }
mkdir -p "$stage_dir/logs" "$work_dir"
ln -sfn "$source_dir/config" "$work_dir/config"
ln -sfn "$checkpoint_dir/mast3r_retrieval" "$work_dir/checkpoints"

cd "$work_dir"
MAST3R_MODEL_PATH="$checkpoint_dir/mast3r" \
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" \
"$python_bin" "$source_dir/main.py" --dataset "$preview_dir" --no-viz --save-as tum_e0 \
  --config "$repo_root/experiments/pointmap_comparison/E0/config/mast3r_e0.yaml" \
  2>&1 | tee "$stage_dir/logs/mast3r_e0_inference.log"
printf 'MASt3R-SLAM output directory: %s\n' "$work_dir"
