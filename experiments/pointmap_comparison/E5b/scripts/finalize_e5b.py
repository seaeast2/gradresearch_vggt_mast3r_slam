#!/usr/bin/env python3
"""Write E5b manifest, preserving completion or failure evidence."""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--repo-root", type=Path, required=True)
parser.add_argument("--stage", type=Path, required=True)
parser.add_argument("--run-id", required=True)
parser.add_argument("--status", choices=["completed", "failed"], required=True)
parser.add_argument("--exit-code", type=int, required=True)
args = parser.parse_args()
root, stage = args.repo_root.resolve(), args.stage.resolve()
e0 = root / "results/pointmap_comparison/tum-freiburg1_desk-20261105-001/stages/E0"
e5 = root / "results/pointmap_comparison/hkairport01-E5-20261105-002/stages/E5"
e0_manifest = json.loads((e0 / "manifest.json").read_text())
checkpoint_sha = next(x["sha256"] for x in e0_manifest["inputs"]
                      if x["path"] == "artifacts/checkpoints/vggt/model.pt")
files = []
for path in sorted(stage.rglob("*")):
    if not path.is_file() or path.name == "manifest.json":
        continue
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    entry = {"path": str(path.relative_to(stage)), "size_bytes": path.stat().st_size,
             "sha256": digest.hexdigest()}
    files.append(entry)

summary_path = stage / "artifacts/e5b_summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.is_file() else None
manifest = {
    "run_id": args.run_id, "stage_code": "E5b",
    "stage_name": "HKairport01 matched-keyframe VGGT and MASt3R-SLAM pointclouds",
    "status": args.status, "exit_code": args.exit_code,
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "execution_context": "host", "source_git_commit": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "input_source_stage": str(e5.relative_to(root)),
    "mast3r_source_stage": str(e0.relative_to(root)),
    "vggt_checkpoint_sha256": checkpoint_sha,
    "mast3r_checkpoint_sha256": [
        x["sha256"] for x in e0_manifest["inputs"]
        if x["path"] in ("artifacts/checkpoints/mast3r_official_hf/model.safetensors",
                          "artifacts/checkpoints/mast3r_retrieval_mirror/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth")
    ],
    "summary": summary, "outputs": files,
}
(stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({key: value for key, value in manifest.items() if key != "outputs"}, indent=2))
