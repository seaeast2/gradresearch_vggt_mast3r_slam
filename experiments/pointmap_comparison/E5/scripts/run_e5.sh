#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
stage="$(realpath "$1")"
run_id="$(basename "$(dirname "$(dirname "$stage")")")"
[[ "$stage" == "$repo_root/results/pointmap_comparison/"* ]] || { echo "Stage must be inside results/pointmap_comparison." >&2; exit 2; }
[[ -f "$stage/artifacts/data/HKairport01/mapping/rgb.txt" ]] || { echo "Run extract_hkairport01.py first." >&2; exit 2; }
[[ ! -e "$stage/manifest.json" ]] || { echo "Refusing to overwrite stage manifest." >&2; exit 2; }

e0_run="${E0_RUN_ID:-tum-freiburg1_desk-20261105-001}"
e0_stage="$repo_root/results/pointmap_comparison/$e0_run/stages/E0"
python_bin="$e0_stage/environment/venvs/mast3r_slam/bin/python"
source_dir="$e0_stage/artifacts/sources/mast3r_slam"
model_dir="$e0_stage/artifacts/checkpoints/mast3r_official_hf"
retrieval_dir="$e0_stage/artifacts/checkpoints/mast3r_retrieval_mirror"
data="$stage/artifacts/data/HKairport01"
dataset="$stage/artifacts/data/tum/rgbd_dataset_freiburg1_HKairport01_E5"
config="$stage/config/mast3r_e5.yaml"
work="$stage/artifacts/work"
outputs="$stage/artifacts/outputs"
mkdir -p "$stage/config" "$stage/logs" "$work" "$outputs"
cp "$repo_root/experiments/pointmap_comparison/E0/config/mast3r_e0.yaml" "$config"
cp "$repo_root/experiments/pointmap_comparison/E5/scripts/extract_hkairport01.py" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E5/scripts/finalize_e5.py" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E5/scripts/run_e5.sh" "$stage/config/"
ln -s "$source_dir/config" "$work/config"
ln -s "$retrieval_dir" "$work/checkpoints"
"$python_bin" - "$data" "$dataset" <<'PY'
import csv,pathlib,sys
source=pathlib.Path(sys.argv[1]);dataset=pathlib.Path(sys.argv[2]);rgb=dataset/'rgb'
if dataset.exists() and any(dataset.iterdir()):raise SystemExit(f'Refusing to overwrite {dataset}')
rgb.mkdir(parents=True,exist_ok=True)
with (source/'mapping_frames.csv').open(newline='') as f:rows=list(csv.DictReader(f))
for row in rows:
    target=rgb/row['filename']
    if not target.exists():target.symlink_to((source/'mapping'/'rgb'/row['filename']).resolve())
lines=['# timestamp filename']+[f"{r['rgb_timestamp']} rgb/{r['filename']}" for r in rows]
(dataset/'rgb.txt').write_text(chr(10).join(lines)+chr(10))
PY

export MAST3R_MODEL_PATH="$model_dir"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${MAST3R_PYTHON:-$python_bin}"
overlay="$work/main_e5.py"
"$python_bin" - "$source_dir/main.py" "$overlay" <<'PY'
import pathlib,sys
source=pathlib.Path(sys.argv[1]).read_text()
old='    keyframes = SharedKeyframes(manager, h, w, buffer=len(dataset))\n'
new='    # E5: cap GPU-resident keyframe tensors for the 8 GiB host GPU.\n    keyframes = SharedKeyframes(manager, h, w, buffer=min(len(dataset), 32))\n'
if source.count(old)!=1:
    raise SystemExit('Pinned source changed: expected exactly one keyframe allocation line')
pathlib.Path(sys.argv[2]).write_text(source.replace(old,new,1))
PY

started="$(date --iso-8601=seconds)"
start_epoch="$(date +%s.%N)"
"$python_bin" - "$stage/manifest.json" "$run_id" "$started" "$repo_root" <<'PY'
import json,pathlib,subprocess,sys
path=pathlib.Path(sys.argv[1])
path.write_text(json.dumps({'run_id':sys.argv[2],'stage_code':'E5','stage_name':'HKairport01 continuous MASt3R-SLAM and fused PLY export','status':'running','started_at':sys.argv[3],'execution_context':'host','source_git_commit':subprocess.check_output(['git','-C',sys.argv[4],'rev-parse','HEAD'],text=True).strip()},indent=2)+'\n')
PY

cd "$work"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader,nounits -lms 200 > "$stage/logs/gpu_memory_samples.csv" 2> "$stage/logs/gpu_monitor.log" &
monitor_pid=$!
set +e
/usr/bin/time -v "$python_bin" "$overlay" --dataset "$dataset" --no-viz --save-as "$run_id" --config "$config" > "$stage/logs/mast3r.log" 2>&1
exit_code=$?
set -e
kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
end_epoch="$(date +%s.%N)"
ended="$(date --iso-8601=seconds)"
if [[ -d "$work/logs/$run_id" ]]; then mv "$work/logs/$run_id" "$outputs/$run_id"; fi
if [[ $exit_code -eq 0 ]]; then
  ply="$outputs/$run_id/mapping.ply"
  [[ -f "$ply" ]] || ply="$(find "$outputs/$run_id" -maxdepth 2 -type f -name '*.ply' -print -quit)"
  [[ -n "$ply" && -f "$ply" ]] || { echo "MASt3R-SLAM exited successfully without a PLY." >&2; exit_code=3; }
fi

"$python_bin" - "$stage" "$run_id" "$exit_code" "$start_epoch" "$end_epoch" "$ended" "$repo_root" <<'PY'
import hashlib,json,pathlib,re,subprocess,sys
root=pathlib.Path(sys.argv[1]);run_id=sys.argv[2];rc=int(sys.argv[3]);start=float(sys.argv[4]);end=float(sys.argv[5]);outputs=[]
for path in sorted(root.rglob('*')):
    if not path.is_file() or path.name=='manifest.json':continue
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    item={'path':str(path.relative_to(root)),'size_bytes':path.stat().st_size,'sha256':digest.hexdigest()}
    if path.suffix=='.ply':
        header=path.open('rb').read(1024*1024).split(b'end_header',1)[0].decode('ascii','replace')
        m=re.search(r'^element vertex (\d+)$',header,re.M)
        item['vertex_count']=int(m.group(1)) if m else None
    outputs.append(item)
samples=[]
for line in (root/'logs/gpu_memory_samples.csv').read_text(errors='replace').splitlines():
    try:samples.append(int(line.rsplit(',',1)[1].strip()))
    except Exception:pass
manifest={'run_id':run_id,'stage_code':'E5','stage_name':'HKairport01 continuous MASt3R-SLAM and fused PLY export','status':'completed' if rc==0 else 'failed','exit_code':rc,'finished_at':sys.argv[6],'wall_time_seconds':end-start,'execution_context':'host','gpu':'NVIDIA GeForce RTX 5060','gpu_memory_sample_period_ms':200,'peak_gpu_memory_mib_sampled':max(samples) if samples else None,'source_git_commit':subprocess.check_output(['git','-C',sys.argv[7],'rev-parse','HEAD'],text=True).strip(),'outputs':outputs}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
if rc==0 and not any(x.get('vertex_count',0)>0 for x in outputs if x['path'].endswith('.ply')):raise SystemExit('PLY validation failed: no non-empty vertex element')
PY
if [[ $exit_code -ne 0 ]]; then
  "$python_bin" - "$stage/manifest.json" "$repo_root" <<'PY'
import datetime,json,pathlib,subprocess,sys
stage=pathlib.Path(sys.argv[1]).parent
run_manifest=stage.parent.parent/'run_manifest.json'
if run_manifest.exists():
    data=json.loads(run_manifest.read_text());data['status']='failed';data['finished_at']=datetime.datetime.now().astimezone().isoformat(timespec='seconds');data['failure_summary']=json.loads(pathlib.Path(sys.argv[1]).read_text()).get('status')
    run_manifest.write_text(json.dumps(data,indent=2)+'\n')
PY
  exit "$exit_code"
fi
"$python_bin" "$repo_root/experiments/pointmap_comparison/E5/scripts/finalize_e5.py" --stage "$stage" > "$stage/logs/finalize.log" 2>&1
