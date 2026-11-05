#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
run_id="${1:?usage: prepare_e5_data.sh <run_id>}"
stage="$repo_root/results/pointmap_comparison/$run_id/stages/E5"
run_dir="$repo_root/results/pointmap_comparison/$run_id"
run_manifest="$run_dir/run_manifest.json"
mcap="$stage/artifacts/input/HKairport01_0.mcap"
mkdir -p "$(dirname "$mcap")"
[[ ! -e "$stage/manifest.json" ]] || { echo "Refusing to overwrite $stage/manifest.json" >&2; exit 2; }
if [[ ! -e "$run_manifest" ]]; then
  python3 - "$run_manifest" "$run_id" "$repo_root" <<'PY'
import datetime,json,pathlib,subprocess,sys
path=pathlib.Path(sys.argv[1]);path.parent.mkdir(parents=True,exist_ok=True)
obj={'run_id':sys.argv[2],'experiment':'E5 HKairport01 continuous MASt3R-SLAM and PLY','status':'running','started_at':datetime.datetime.now().astimezone().isoformat(timespec='seconds'),'source_git_commit':subprocess.check_output(['git','-C',sys.argv[3],'rev-parse','HEAD'],text=True).strip(),'stages':['E5'],'stage_directories':{'E5':'stages/E5'},'execution_context':'host'}
path.write_text(json.dumps(obj,indent=2)+'\n')
PY
fi
curl --fail --location --retry 4 --retry-delay 2 --continue-at - \
  --output "$mcap" \
  'https://huggingface.co/datasets/DapengFeng/MCAP/resolve/4dff3556f5283b586e955e0a06b01c2f8165a40d/mars_lvig/HKairport01/HKairport01_0.mcap'
expected='cfcdcedcb3a06efe5a3d1041cd7e9c2bf88cffa7376ec3a878aff754fa34a4cc'
actual="$(sha256sum "$mcap" | cut -d' ' -f1)"
[[ "$actual" == "$expected" ]] || { echo "MCAP hash mismatch: $actual" >&2; exit 1; }
python_bin="$repo_root/results/pointmap_comparison/tum-freiburg1_desk-20261105-001/stages/E0/environment/venvs/mast3r_slam/bin/python"
"$python_bin" "$repo_root/experiments/pointmap_comparison/E5/scripts/extract_hkairport01.py" \
  --mcap "$mcap" --output "$stage/artifacts/data/HKairport01"
