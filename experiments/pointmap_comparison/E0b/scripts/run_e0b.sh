#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
e0_run="${E0_RUN_ID:-tum-freiburg1_desk-20261105-001}"
e0_stage="$repo_root/results/pointmap_comparison/$e0_run/stages/E0"
run_id="${RUN_ID:-tum-freiburg1_desk-E0b-$(date +%Y%m%d-%H%M%S)}"
stage="$repo_root/results/pointmap_comparison/$run_id/stages/E0b"
[[ -d "$e0_stage" ]] || { echo "Missing E0 stage: $e0_stage" >&2; exit 1; }
[[ ! -e "$stage" ]] || { echo "Refusing to overwrite existing stage: $stage" >&2; exit 1; }
mkdir -p "$stage/config" "$stage/logs" "$stage/artifacts/measurements"
printf '{"run_id":"%s","stage_code":"E0b","status":"running","execution_context":"host","e0_source_run":"%s"}\n' "$run_id" "$e0_run" > "$stage/manifest.json"
cp "$e0_stage/artifacts/data/frames_associated.csv" "$stage/config/"
cp "$e0_stage/artifacts/data/frames_e0.csv" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E0b/scripts/"*.py "$stage/config/"
python_bin="$e0_stage/environment/venvs/vggt/bin/python"
checkpoint="$e0_stage/artifacts/checkpoints/vggt"
script="$repo_root/experiments/pointmap_comparison/E0b/scripts"
"$python_bin" "$script/analyze_e0b.py" --e0-stage "$e0_stage" --output "$stage/artifacts/measurements/preflight.json" > "$stage/logs/analysis.log" 2>&1
for n in 1 2 4 5 6; do
  set +e
  "$python_bin" "$script/probe_vram_e0b.py" --frame-list "$e0_stage/artifacts/data/frames_e0.csv" --checkpoint-dir "$checkpoint" --n "$n" --output "$stage/artifacts/measurements/vram_N${n}.json" > "$stage/logs/vram_N${n}.log" 2>&1
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    printf '{"status":"failed","n_frames":%s,"exit_code":%s,"failure":"see logs/vram_N%s.log"}\n' "$n" "$status" "$n" > "$stage/artifacts/measurements/vram_N${n}.json"
  fi
done
python3 - "$stage" "$run_id" "$e0_run" <<'PY'
import hashlib, json, pathlib, sys
stage=pathlib.Path(sys.argv[1]); run_id,e0_run=sys.argv[2:]
def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
outputs=[]
for p in sorted(stage.rglob('*')):
 if p.is_file() and p.name!='manifest.json': outputs.append({'path':str(p.relative_to(stage)),'size_bytes':p.stat().st_size,'sha256':sha(p)})
man={'run_id':run_id,'stage_code':'E0b','stage_name':'E1 implementation preflight: licenses, source transforms, GT prefix, thresholds and GPU capacity','status':'completed','execution_context':'host','source_run_id':e0_run,'outputs':outputs,'gpu_probe_sizes':[1,2,4,5,6]}
(stage/'manifest.json').write_text(json.dumps(man,indent=2)+'\n')
PY
echo "E0b stage completed: $stage"
