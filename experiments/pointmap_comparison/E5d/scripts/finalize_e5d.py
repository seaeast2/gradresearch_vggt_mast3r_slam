#!/usr/bin/env python3
"""Summarize and hash E5d outputs, including partial model failures."""
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
parser.add_argument("--status", choices=["completed", "partial", "failed"], required=True)
parser.add_argument("--mast3r-exit-code", type=int, required=True)
parser.add_argument("--vggt-exit-code", type=int, required=True)
args = parser.parse_args()
root, stage = args.repo_root.resolve(), args.stage.resolve()
e0 = root / "results/pointmap_comparison/tum-freiburg1_desk-20261105-001/stages/E0"
e5 = root / "results/pointmap_comparison/hkairport01-E5-20261105-002/stages/E5"
e0_manifest = json.loads((e0 / "manifest.json").read_text())
vggt_sha = next(item["sha256"] for item in e0_manifest["inputs"] if item["path"] == "artifacts/checkpoints/vggt/model.pt")
e5_manifest = json.loads((e5 / "manifest.json").read_text())
map_sha = next(item["sha256"] for item in e5_manifest["outputs"] if item["path"] == "artifacts/data/HKairport01/mapping_frames.csv")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def ply_stats(path):
    raw = path.read_bytes()
    marker = b"end_header\n"
    end = raw.find(marker)
    if end < 0:
        raise ValueError(f"Invalid PLY header: {path}")
    header = raw[:end + len(marker)].decode("ascii")
    import re
    m = re.search(r"^element vertex (\d+)$", header, re.MULTILINE)
    if not m or "format binary_little_endian 1.0" not in header:
        raise ValueError(f"Unsupported PLY format: {path}")
    count = int(m.group(1))
    payload = raw[end + len(marker):]
    if len(payload) != count * 15:
        raise ValueError(f"PLY payload size mismatch: {path}")
    import numpy as np
    dtype = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                      "offsets": [0, 4, 8], "itemsize": 15})
    vertices = np.frombuffer(payload, dtype=dtype)
    xyz = np.column_stack([vertices[key] for key in ("x", "y", "z")])
    finite = np.isfinite(xyz).all(axis=1)
    return {"vertex_count": count, "finite_xyz_count": int(finite.sum()),
            "xyz_min": xyz.min(axis=0).tolist() if count else None,
            "xyz_max": xyz.max(axis=0).tolist() if count else None,
            "size_bytes": path.stat().st_size, "sha256": sha256(path)}

input_summary_path = stage / "artifacts/data/input_summary.json"
input_summary = json.loads(input_summary_path.read_text()) if input_summary_path.is_file() else None
mast3r_json = stage / "artifacts/exports/mast3r/metadata.json"
vggt_json = stage / "artifacts/exports/vggt/metadata.json"
mast_meta = json.loads(mast3r_json.read_text()) if mast3r_json.is_file() else None
vggt_meta = json.loads(vggt_json.read_text()) if vggt_json.is_file() else None
mast_npz = stage / "artifacts/exports/mast3r/mast3r_mf_pointmaps.npz"
vggt_npz = stage / "artifacts/exports/vggt/vggt_pointmaps.npz"
if mast_npz.is_file():
    import numpy as np
    with np.load(mast_npz, allow_pickle=False) as data:
        mast_ids = data["dataset_frame_id"].astype(int).tolist()
        mast_source_ids = data["source_frame_index"].astype(int).tolist()
        mast_stamps = [str(value) for value in data["rgb_timestamp"].tolist()]
        mast_fusion_count = data["fusion_count"].astype(int).tolist()
        mast_update_count = data["update_count"].astype(int).tolist()
else:
    mast_ids, mast_source_ids, mast_stamps = [], [], []
    mast_fusion_count, mast_update_count = [], []
if vggt_npz.is_file():
    import numpy as np
    with np.load(vggt_npz, allow_pickle=False) as data:
        vggt_source_ids = data["source_frame_index"].astype(int).tolist()
        vggt_stamps = [str(value) for value in data["rgb_timestamp"].tolist()]
else:
    vggt_source_ids, vggt_stamps = [], []
clouds = {}
for name, path in [
    ("MASt3R-SLAM M-F", stage / "artifacts/exports/mast3r/HKairport01_E5d_M-F_N04.ply"),
    ("VGGT V-P", stage / "artifacts/exports/vggt/HKairport01_E5d_V-P_N04.ply"),
]:
    if path.is_file():
        clouds[name] = {"path": str(path.relative_to(stage)), **ply_stats(path)}

summary = {
    "schema": "pointmap-comparison.e5d-summary.v1", "run_id": args.run_id,
    "status": args.status, "execution_context": "host",
    "input": input_summary,
    "models": {
        "MASt3R-SLAM": {"exit_code": args.mast3r_exit_code,
            "exported_keyframe_count": mast_meta.get("exported_keyframe_count") if mast_meta else 0,
            "exported_dataset_frame_ids": mast_ids,
            "exported_source_frame_indices": mast_source_ids,
            "exported_timestamps": mast_stamps,
            "keyframe_output_fraction": (mast_meta.get("exported_keyframe_count", 0) / 4) if mast_meta else 0.0,
            "fusion_count": mast_fusion_count, "update_count": mast_update_count,
            "variant": "M-F fused selected keyframes"},
        "VGGT": {"exit_code": args.vggt_exit_code,
            "exported_frame_count": vggt_meta.get("count") if vggt_meta else 0,
            "source_frame_indices": vggt_source_ids,
            "timestamps": vggt_stamps,
            "variant": "V-P direct Point Head; four views jointly inferred"},
    },
    "clouds": clouds,
    "coordinate_note": "Separate model-native pointclouds. Do not overlay without explicit pose/gauge alignment.",
    "accuracy_claim": "qualitative; HKairport01 has no geometry ground truth",
}
(stage / "artifacts/e5d_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

files = []
for path in sorted(stage.rglob("*")):
    if path.is_file() and path.name != "manifest.json":
        files.append({"path": str(path.relative_to(stage)), "size_bytes": path.stat().st_size,
                      "sha256": sha256(path)})
manifest = {
    "run_id": args.run_id, "stage_code": "E5d",
    "stage_name": "HKairport01 matched four-frame two-second pointcloud comparison",
    "status": args.status, "execution_context": "host",
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "source_git_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "e5_mapping_csv_sha256": map_sha, "vggt_checkpoint_sha256": vggt_sha,
    "input_summary": input_summary, "summary": summary, "outputs": files,
}
(stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({key: value for key, value in manifest.items() if key != "outputs"}, indent=2))
