#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
stage="$(realpath "$1")"; dataset="$2"; n="$3"; calibration="$4"; repeat="$5"; protocol="${6:-P1}"
e0_run="${E0_RUN_ID:-tum-freiburg1_desk-20261105-001}"
e0_stage="$repo_root/results/pointmap_comparison/$e0_run/stages/E0"
data="$stage/artifacts/data/$dataset"
if [[ "$protocol" == P2 ]]; then input="$data/frames_all.csv";cond="$stage/artifacts/conditions/$dataset/P2/MASt3R/full/repeat$(printf '%02d' "$repeat")";else input="$data/frames_$(printf '%02d' "$n").csv";cond="$stage/artifacts/conditions/$dataset/P1/$calibration/N$(printf '%02d' "$n")/MASt3R/repeat$(printf '%02d' "$repeat")";fi
[[ -f "$input" ]] || { echo "Missing frame list $input" >&2; exit 2; }
[[ ! -e "$cond" ]] || { echo "Refusing to overwrite $cond" >&2; exit 2; }
mkdir -p "$cond/config" "$cond/logs" "$cond/artifacts/raw" "$cond/artifacts/exports" "$cond/artifacts/work"
cp "$input" "$cond/config/frames.csv"
cp "$repo_root/experiments/pointmap_comparison/E0a/config/mast3r_e0a.yaml" "$cond/config/"
cp "$repo_root/experiments/pointmap_comparison/E0a/scripts/run_mast3r_e0a.py" "$cond/config/"
cp "$repo_root/experiments/pointmap_comparison/E1/scripts/export_mast3r_slam.py" "$cond/config/"
cp "$repo_root/experiments/pointmap_comparison/E3/scripts/run_mast3r_attempt.sh" "$cond/config/"
cp "$repo_root/experiments/pointmap_comparison/E3/scripts/calibration_freiburg1.yaml" "$cond/config/" 2>/dev/null || true
python="$e0_stage/environment/venvs/mast3r_slam/bin/python"
src="$e0_stage/artifacts/sources/mast3r_slam"; work="$cond/artifacts/work"
ln -s "$src/config" "$work/config"; ln -s "$e0_stage/artifacts/checkpoints/mast3r_retrieval_mirror" "$work/checkpoints"
"$python" - "$input" "$data/tum/rgbd_dataset_freiburg1_${dataset}_N${n}" <<'PY'
import csv,pathlib,sys
inp,out=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2])
with inp.open(newline='') as f:rows=list(csv.DictReader(f))
lines=['# E3 shared P1 inputs','# timestamp filename']
for r in rows:
 p=pathlib.Path(r['rgb_path']);lines.append(f"{r['rgb_timestamp']} rgb/{p.name}")
if out.exists():
 if (out/'rgb.txt').read_text().splitlines()!=lines:raise RuntimeError(f'Existing shared input list differs: {out}')
 raise SystemExit(0)
out.mkdir(parents=True)
for r in rows:
 p=pathlib.Path(r['rgb_path']);dest=out/'rgb'/p.name;dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(p.resolve())
(out/'rgb.txt').write_text('\n'.join(lines)+'\n')
PY
export E0A_EXPORT_DIR="$cond/artifacts/raw" E0A_FRAME_MAP_CSV="$input"
export MAST3R_MODEL_PATH="$e0_stage/artifacts/checkpoints/mast3r_official_hf" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PYTHONPATH="$src${PYTHONPATH:+:$PYTHONPATH}"
"$python" "$repo_root/experiments/pointmap_comparison/E0a/scripts/run_mast3r_e0a.py" "$src/main.py" "$work/main_e3.py" > "$cond/logs/build_overlay.log" 2>&1
cd "$work"
calib_args=(); [[ "$calibration" == C1 ]] && calib_args=(--calib "$cond/config/calibration_freiburg1.yaml")
start=$(date +%s.%N)
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 100 > "$cond/logs/gpu_memory_samples.csv" 2> "$cond/logs/gpu_monitor.log" & monitor_pid=$!
set +e
"$python" "$work/main_e3.py" --dataset "$data/tum/rgbd_dataset_freiburg1_${dataset}_N${n}" --no-viz --save-as "e3_${dataset}_${calibration}_n${n}_r${repeat}" --config "$cond/config/mast3r_e0a.yaml" "${calib_args[@]}" > "$cond/logs/mast3r.log" 2>&1
rc=$?
set -e
kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true
end=$(date +%s.%N)
"$python" - "$cond" "$rc" "$start" "$end" <<'PY'
import hashlib,json,pathlib,sys,subprocess
root=pathlib.Path(sys.argv[1]);rc=int(sys.argv[2]);start=float(sys.argv[3]);end=float(sys.argv[4]);outputs=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and p.name!='manifest.json':outputs.append({'path':str(p.relative_to(root)),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
samples=[]
try:
 for line in (root/'logs/gpu_memory_samples.csv').read_text().splitlines():samples.append(int(line.split(',')[-1].strip()))
except Exception:pass
m={'stage_code':'E3','model':'MASt3R-SLAM','condition':str(root),'status':'completed' if rc==0 else 'failed','exit_code':rc,'wall_time_s':end-start,'peak_gpu_memory_mib':max(samples) if samples else None,'gpu_memory_samples':len(samples),'execution_context':'host','source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'outputs':outputs}
(root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps(m,indent=2))
PY
[[ $rc -eq 0 ]] || exit $rc
"$python" "$repo_root/experiments/pointmap_comparison/E1/scripts/export_mast3r_slam.py" --hook-export "$cond/artifacts/raw/mast3r_e0a_fused_keyframes.npz" --frame-list "$input" --output-dir "$cond/artifacts/exports" > "$cond/logs/normalize.log" 2>&1
"$python" - "$cond" "$rc" "$start" "$end" <<'PY'
import hashlib,json,pathlib,sys,subprocess
root=pathlib.Path(sys.argv[1]);rc=int(sys.argv[2]);start=float(sys.argv[3]);end=float(sys.argv[4]);outputs=[]
for p in sorted(root.rglob('*')):
 if p.is_file() and p.name!='manifest.json':outputs.append({'path':str(p.relative_to(root)),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
samples=[]
try:
 for line in (root/'logs/gpu_memory_samples.csv').read_text().splitlines():samples.append(int(line.split(',')[-1].strip()))
except Exception:pass
m={'stage_code':'E3','model':'MASt3R-SLAM','status':'completed' if rc==0 else 'failed','exit_code':rc,'wall_time_s':end-start,'peak_gpu_memory_mib':max(samples) if samples else None,'gpu_memory_samples':len(samples),'execution_context':'host','source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'outputs':outputs}
(root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
PY
