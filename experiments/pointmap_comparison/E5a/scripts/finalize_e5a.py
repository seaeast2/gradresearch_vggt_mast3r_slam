#!/usr/bin/env python3
"""Validate E5a pointmap exports and write their run manifest."""
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

root, stage, run_id = map(Path, sys.argv[1:4])
root, stage = root.resolve(), stage.resolve()
e0 = root / "results/pointmap_comparison/tum-freiburg1_desk-20261105-001/stages/E0"
e5 = root / "results/pointmap_comparison/hkairport01-E5-20261105-002/stages/E5"
exports = stage / "artifacts/exports"
direct_ply = exports / "HKairport01_E5a_V-P_N04.ply"
npz_path = exports / "vggt_pointmaps.npz"
metadata = json.loads((exports / "metadata.json").read_text())
inputs = json.loads((stage / "artifacts/data/input_summary.json").read_text())
frames = list(__import__("csv").DictReader((stage / "artifacts/data/vggt_frames.csv").open(newline="")))
if len(frames) != 4 or metadata.get("count") != 4:
    raise ValueError("E5a must contain exactly four jointly inferred frames")
if metadata.get("gpu") != "NVIDIA GeForce RTX 5060":
    raise ValueError(f"Unexpected host GPU: {metadata.get('gpu')}")

raw = direct_ply.read_bytes()
marker = b"end_header\n"
end = raw.find(marker)
if end < 0:
    raise ValueError("Merged direct pointmap is not a valid PLY")
header = raw[:end + len(marker)].decode("ascii")
match = re.search(r"^element vertex (\d+)$", header, re.MULTILINE)
if not match or "format binary_little_endian 1.0" not in header:
    raise ValueError("Merged direct pointmap has an invalid header")
count = int(match.group(1))
payload = raw[end + len(marker):]
vertex_dtype = np.dtype({"names": ["x", "y", "z", "r", "g", "b"],
                         "formats": ["<f4", "<f4", "<f4", "u1", "u1", "u1"],
                         "offsets": [0, 4, 8, 12, 13, 14], "itemsize": 15})
vertices = np.frombuffer(payload, dtype=vertex_dtype)
finite = np.isfinite(np.column_stack([vertices[axis] for axis in ("x", "y", "z")])).all(axis=1)
if count != len(vertices) or not finite.all() or count == 0:
    raise ValueError(f"Invalid direct PLY vertices: header={count}, payload={len(vertices)}, finite={int(finite.sum())}")

with np.load(npz_path, allow_pickle=False) as data:
    if data["V_P"].shape[0] != 4 or data["V_P"].shape[-1] != 3:
        raise ValueError(f"Unexpected V-P array shape: {data['V_P'].shape}")
    valid_pixels = data["valid_mask"].reshape(4, -1).sum(axis=1).astype(int).tolist()
summary = {
    "schema": "pointmap-comparison.e5a-summary.v1",
    "run_id": str(run_id),
    "model": "facebook/VGGT-1B",
    "output_variant": "V-P direct Point Head",
    "source_mapping_frames": inputs["source_frame_count"],
    "inferred_frames": 4,
    "selected_mapping_row_indices": inputs["selected_mapping_row_indices"],
    "selected_frame_indices": inputs["selected_frame_indices"],
    "selected_timestamps": inputs["selected_timestamps"],
    "coordinate_frame": "one VGGT-predicted world frame shared by the four views in this joint inference",
    "point_count": count,
    "finite_xyz_count": int(finite.sum()),
    "valid_pixels_per_frame": valid_pixels,
    "network_shape_hw": metadata["network_shape_hw"],
    "gpu": metadata["gpu"],
    "dtype": metadata["dtype"],
    "peak_allocated_bytes": metadata["peak_allocated_bytes"],
    "peak_reserved_bytes": metadata["peak_reserved_bytes"],
    "direct_ply": str(direct_ply.relative_to(stage)),
    "whole_sequence_limit": "Only four evenly spaced images were inferred; this is not a full 240-frame VGGT reconstruction.",
}
(stage / "artifacts/e5a_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

files = []
for path in sorted(stage.rglob("*")):
    if path.is_file() and path.name != "manifest.json":
        files.append({"path": str(path.relative_to(stage)), "size_bytes": path.stat().st_size,
                      "sha256": sha256(path)})
e0_manifest = json.loads((e0 / "manifest.json").read_text())
checkpoint_sha = next(x["sha256"] for x in e0_manifest["inputs"]
                      if x["path"] == "artifacts/checkpoints/vggt/model.pt")
manifest = {
    "run_id": str(run_id), "stage_code": "E5a",
    "stage_name": "HKairport01 VGGT four-view pointmap and PLY",
    "status": "completed", "exit_code": 0, "execution_context": "host",
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "source_git_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    "model": "facebook/VGGT-1B", "checkpoint_sha256": checkpoint_sha,
    "checkpoint_source_stage": str(e0.relative_to(root)),
    "input_source_stage": str(e5.relative_to(root)), "input_selection": inputs,
    "gpu": metadata["gpu"], "dtype": metadata["dtype"],
    "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
    "gpu_compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
    "peak_allocated_bytes": metadata["peak_allocated_bytes"],
    "peak_reserved_bytes": metadata["peak_reserved_bytes"], "outputs": files,
}
(stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(summary, indent=2))
