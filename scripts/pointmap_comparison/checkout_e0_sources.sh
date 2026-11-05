#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source_dir="$repo_root/thirdparty/pointmap_comparison"
mkdir -p "$source_dir"

checkout_pinned() {
  local name="$1" url="$2" commit="$3" path="$4"
  if [[ ! -d "$path/.git" ]]; then
    git clone --recursive "$url" "$path"
  fi
  git -C "$path" fetch --all --tags
  git -C "$path" checkout --detach "$commit"
  git -C "$path" submodule update --init --recursive
  if [[ "$(git -C "$path" rev-parse HEAD)" != "$commit" ]]; then
    printf 'Unexpected %s source revision at %s\n' "$name" "$path" >&2
    return 1
  fi
  printf '%s source pinned at %s\n' "$name" "$commit"
}

checkout_pinned "VGGT" "https://github.com/facebookresearch/vggt.git" \
  "a288dd0f14786c93483e45524328726ab7b1b4ce" "$source_dir/vggt"
checkout_pinned "MASt3R-SLAM" "https://github.com/rmurai0610/MASt3R-SLAM.git" \
  "e6f4e3d474fad0e11f561482012be864ba8c3f17" "$source_dir/mast3r_slam"

patch_file="$repo_root/experiments/pointmap_comparison/E0/patches/mast3r_slam_e0.patch"
if git -C "$source_dir/mast3r_slam" apply --check "$patch_file"; then
  git -C "$source_dir/mast3r_slam" apply "$patch_file"
elif ! git -C "$source_dir/mast3r_slam" apply --reverse --check "$patch_file"; then
  printf 'E0 patch does not apply cleanly to the pinned MASt3R-SLAM checkout.\n' >&2
  exit 1
else
  printf 'E0 patch is already applied.\n'
fi
