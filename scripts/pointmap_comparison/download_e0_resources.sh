#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_dir="$repo_root/data/pointmap_comparison"
checkpoint_dir="$repo_root/checkpoints/pointmap_comparison"
mkdir -p "$data_dir" "$checkpoint_dir/vggt" "$checkpoint_dir/mast3r" "$checkpoint_dir/mast3r_retrieval"

fetch_verified() {
  local name="$1" expected_sha="$2" url="$3" destination="$4"
  mkdir -p "$(dirname "$destination")"
  if [[ -f "$destination" ]] && [[ "$(sha256sum "$destination" | awk '{print $1}')" == "$expected_sha" ]]; then
    printf 'Already verified: %s\n' "$destination"
    return
  fi
  local partial="${destination}.partial"
  printf 'Downloading %s\n' "$name"
  if [[ -f "$partial" ]]; then
    curl --fail --location --retry 5 --retry-all-errors --continue-at - "$url" --output "$partial"
  else
    curl --fail --location --retry 5 --retry-all-errors "$url" --output "$partial"
  fi
  local actual_sha
  actual_sha="$(sha256sum "$partial" | awk '{print $1}')"
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    rm -f "$partial"
    printf 'SHA-256 mismatch for %s: expected %s, got %s\n' "$name" "$expected_sha" "$actual_sha" >&2
    return 1
  fi
  mv "$partial" "$destination"
}

# TUM RGB-D benchmark sequence, archive SHA-256 recorded by E0.
fetch_verified "TUM freiburg1_desk" \
  "e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c" \
  "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz" \
  "$data_dir/rgbd_dataset_freiburg1_desk.tgz"

# VGGT-1B and MASt3R model files. Downloaded files remain outside Git.
fetch_verified "VGGT-1B model.pt" \
  "d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0" \
  "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt" \
  "$checkpoint_dir/vggt/model.pt"
fetch_verified "MASt3R model.safetensors" \
  "0a615eb05fa9db654050aa655945ee5696e7c6c1b7f93f1ee8c37249010f6feb" \
  "https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/resolve/8991601/model.safetensors" \
  "$checkpoint_dir/mast3r/model.safetensors"
fetch_verified "MASt3R config.json" \
  "718eb93dc4f9e4332b60cc0041af962d712cbd346d7770ce35c5b22cff68eae4" \
  "https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric/resolve/8991601/config.json" \
  "$checkpoint_dir/mast3r/config.json"

# Retrieval assets are mirrored by vslamlab; preserve their provenance and verify content.
fetch_verified "MASt3R training-free retrieval checkpoint (vslamlab mirror)" \
  "000a324f06f9369ab3e32d03dd4b8a838d48e98e3f3bf9d7dd034dce1bfd4c50" \
  "https://huggingface.co/vslamlab/mast3rslam_weights/resolve/main/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth" \
  "$checkpoint_dir/mast3r_retrieval/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
fetch_verified "MASt3R retrieval codebook (vslamlab mirror)" \
  "700af9e31ad0ca6159955dc80b4fd6e14234780b52f5c91884f6553991933c1a" \
  "https://huggingface.co/vslamlab/mast3rslam_weights/resolve/main/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl" \
  "$checkpoint_dir/mast3r_retrieval/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"

printf 'E0 resources downloaded and SHA-256 verified.\n'
