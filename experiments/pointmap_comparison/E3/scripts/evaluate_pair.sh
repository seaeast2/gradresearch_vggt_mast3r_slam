#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)";stage="$(realpath "$1")";scene="$2";n="$3";repeat="$4";mast_repeat="${5:-$repeat}";calib="C0"
cond="$stage/artifacts/conditions/$scene/P1/$calib/N$(printf '%02d' "$n")";v="$cond/VGGT/repeat$(printf '%02d' "$repeat")/artifacts/exports/vggt_pointmaps.npz";m="$cond/MASt3R/repeat$(printf '%02d' "$mast_repeat")/artifacts/exports/mast3r_mf_pointmaps.npz"
[[ -f "$v" && -f "$m" ]] || { echo "Missing paired predictions: $v or $m" >&2; exit 2; }
data="$stage/artifacts/data/$scene";py="${EVAL_PYTHON:-$(dirname "$(dirname "$repo_root/results/pointmap_comparison/tum-freiburg1_desk-20261105-001/stages/E0/environment/venvs/mast3r_slam/bin/python")")}/bin/python"
[[ -x "$py" ]] || py="$repo_root/results/pointmap_comparison/tum-freiburg1_desk-20261105-001/stages/E0/environment/venvs/mast3r_slam/bin/python"
out="$cond/metrics/vggt-r$(printf '%02d' "$repeat")_mast3r-r$(printf '%02d' "$mast_repeat")";mkdir -p "$out"
"$py" "$repo_root/experiments/pointmap_comparison/E2/scripts/evaluate_pixel.py" --vggt "$v" --mast3r "$m" --frames "$data/frames_$(printf '%02d' "$n").csv" --associations "$data/frames_associated.csv" --output-dir "$out/pixel" > "$cond/evaluate_pixel.log" 2>&1
"$py" "$repo_root/experiments/pointmap_comparison/E2/scripts/evaluate_cloud.py" --input "$out/pixel/local_pointmaps_for_cloud_outlier.npz" --output-dir "$out/cloud" > "$cond/evaluate_cloud.log" 2>&1
"$py" "$repo_root/experiments/pointmap_comparison/E2/scripts/evaluate_outliers.py" --input "$out/pixel/local_pointmaps_for_cloud_outlier.npz" --output "$out/outliers.json" > "$cond/evaluate_outliers.log" 2>&1
