#!/usr/bin/env python3
"""Write E1 stage manifest and hashes for either a completed or failed attempt."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=Path, required=True)
parser.add_argument("--run-id", required=True)
parser.add_argument("--e0-stage", type=Path, required=True)
parser.add_argument("--e0a-stage", type=Path, required=True)
parser.add_argument("--status", choices=["running", "completed", "failed"], required=True)
parser.add_argument("--exit-code", type=int, default=0)
args = parser.parse_args()
stage = args.stage.resolve()

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

outputs = []
for path in sorted(stage.rglob("*")):
    if path.is_file() and path.name != "manifest.json":
        outputs.append({"path": str(path.relative_to(stage)), "size_bytes": path.stat().st_size, "sha256": sha(path)})
e0_manifest = json.loads((args.e0_stage / "manifest.json").read_text())
e0a_manifest = json.loads((args.e0a_stage / "manifest.json").read_text())
manifest = {
    "run_id": args.run_id, "stage_code": "E1", "stage_name": "Standard pointmap export",
    "status": args.status, "execution_context": "host",
    "source_git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "inputs": [
        {"name": "E0 stage", "path": str(args.e0_stage), "source_git_commit": e0_manifest.get("source_git_commit"),
         "checkpoint_hashes": e0_manifest.get("inputs", [])},
        {"name": "E0a stage", "path": str(args.e0a_stage), "source_git_commit": e0a_manifest.get("source_git_commit")},
    ],
    "settings": {"vggt_max_frames": 4, "vggt_input_size": [518, 518], "mast3r_variant": "M-F fused keyframes"},
    "outputs": outputs,
}
if args.status == "failed":
    manifest["exit_code"] = args.exit_code
(stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
