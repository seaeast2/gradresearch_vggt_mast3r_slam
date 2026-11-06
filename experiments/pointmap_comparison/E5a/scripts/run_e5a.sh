#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
run_id="${1:?usage: run_e5a.sh <run_id>}"
run_root="$repo_root/results/pointmap_comparison/$run_id"
stage="$run_root/stages/E5a"
[[ "$run_id" == hkairport01-E5a-* ]] || { echo "run_id must start with hkairport01-E5a-" >&2; exit 2; }
[[ ! -e "$stage" ]] || { echo "Refusing to overwrite $stage" >&2; exit 2; }
e0_stage="$repo_root/results/pointmap_comparison/tum-freiburg1_desk-20261105-001/stages/E0"
e5_stage="$repo_root/results/pointmap_comparison/hkairport01-E5-20261105-002/stages/E5"
python="$e0_stage/environment/venvs/vggt/bin/python"
checkpoint="$e0_stage/artifacts/checkpoints/vggt"
[[ -x "$python" && -f "$checkpoint/model.pt" ]] || { echo "Pinned VGGT environment/checkpoint missing" >&2; exit 2; }

mkdir -p "$stage"/{config,logs,artifacts/data,artifacts/exports}
cp "$repo_root/experiments/pointmap_comparison/E3/scripts/export_vggt_e3.py" "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E5a/scripts/"*.py "$stage/config/"
cp "$repo_root/experiments/pointmap_comparison/E5a/scripts/run_e5a.sh" "$stage/config/"
cp "$e5_stage/artifacts/data/HKairport01/mapping_frames.csv" "$stage/config/e5_mapping_frames.csv"

set +e
"$python" "$stage/config/prepare_e5a.py" \
  --mapping-csv "$stage/config/e5_mapping_frames.csv" \
  --output-csv "$stage/artifacts/data/vggt_frames.csv" \
  --summary "$stage/artifacts/data/input_summary.json" > "$stage/logs/prepare.log" 2>&1
rc=$?
set -e
if (( rc != 0 )); then cat "$stage/logs/prepare.log"; exit "$rc"; fi

set +e
"$python" "$stage/config/export_vggt_e3.py" \
  --frame-list "$stage/artifacts/data/vggt_frames.csv" \
  --checkpoint-dir "$checkpoint" \
  --output-dir "$stage/artifacts/exports" > "$stage/logs/vggt.log" 2>&1
rc=$?
set -e
if (( rc != 0 )); then cat "$stage/logs/vggt.log"; fi
if (( rc == 0 )); then
  shopt -s nullglob
  direct_plys=("$stage/artifacts/exports"/*_V-P.ply)
  "$python" "$stage/config/merge_direct_ply.py" --inputs "${direct_plys[@]}" \
    --output "$stage/artifacts/exports/HKairport01_E5a_V-P_N04.ply" > "$stage/logs/merge.log" 2>&1 || rc=$?
fi

if (( rc == 0 )); then
  "$python" "$stage/config/finalize_e5a.py" "$repo_root" "$stage" "$run_id" \
    > "$stage/logs/finalize.log" 2>&1 || rc=$?
else
  "$python" - "$stage" "$run_id" "$rc" <<'PY'
import json,pathlib,sys,time
stage=pathlib.Path(sys.argv[1]);run_id,rc=sys.argv[2:]
(stage/'manifest.json').write_text(json.dumps({'run_id':run_id,'stage_code':'E5a','status':'failed','exit_code':int(rc),'execution_context':'host','finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')},indent=2)+'\n')
PY
fi
if (( rc != 0 )); then cat "$stage/logs/finalize.log" 2>/dev/null || true; fi
exit "$rc"
