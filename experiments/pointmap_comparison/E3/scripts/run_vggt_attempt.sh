#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)";stage="$(realpath "$1")";dataset="$2";n="$3";repeat="$4";protocol="${5:-P1}"
e0_run="${E0_RUN_ID:-tum-freiburg1_desk-20261105-001}";e0_stage="$repo_root/results/pointmap_comparison/$e0_run/stages/E0"
if [[ "$protocol" == P2 ]]; then input="$stage/artifacts/data/$dataset/frames_p2_vggt.csv";cond="$stage/artifacts/conditions/$dataset/P2/VGGT/fixed04/repeat$(printf '%02d' "$repeat")";else input="$stage/artifacts/data/$dataset/frames_$(printf '%02d' "$n").csv";cond="$stage/artifacts/conditions/$dataset/P1/C0/N$(printf '%02d' "$n")/VGGT/repeat$(printf '%02d' "$repeat")";fi
[[ -f "$input" ]] || { echo "Missing frame list $input" >&2;exit 2; };[[ ! -e "$cond" ]] || { echo "Refusing to overwrite $cond" >&2;exit 2; }
mkdir -p "$cond/config" "$cond/logs" "$cond/artifacts/exports"
cp "$input" "$cond/config/frames.csv";cp "$repo_root/experiments/pointmap_comparison/E3/scripts/export_vggt_e3.py" "$cond/config/";cp "$repo_root/experiments/pointmap_comparison/E3/scripts/run_vggt_attempt.sh" "$cond/config/"
python="$e0_stage/environment/venvs/vggt/bin/python";start=$(date +%s.%N);set +e
args=();((n>4)) && args=(--allow-unsafe-n)
"$python" "$repo_root/experiments/pointmap_comparison/E3/scripts/export_vggt_e3.py" --frame-list "$input" --checkpoint-dir "$e0_stage/artifacts/checkpoints/vggt" --output-dir "$cond/artifacts/exports" "${args[@]}" > "$cond/logs/vggt.log" 2>&1
rc=$?;end=$(date +%s.%N);set -e
"$python" - "$cond" "$dataset" "$n" "$repeat" "$rc" "$start" "$end" "$protocol" <<'PY'
import hashlib,json,pathlib,sys,subprocess
r=pathlib.Path(sys.argv[1]);data,n,rep,rc=sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5]);start,end=map(float,sys.argv[6:8]);files=[]
for p in sorted(r.rglob('*')):
 if p.is_file() and p.name!='manifest.json':files.append({'path':str(p.relative_to(r)),'size_bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
m={'stage_code':'E3','model':'VGGT-1B','dataset':data,'protocol':sys.argv[8],'input_budget':n,'repeat':rep,'status':'completed' if rc==0 else 'failed_or_oom','exit_code':rc,'wall_time_s':end-start,'execution_context':'host','source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'outputs':files}
(r/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m,indent=2))
PY
exit 0
