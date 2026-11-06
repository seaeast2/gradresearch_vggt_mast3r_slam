#!/usr/bin/env python3
"""Validate same-frame inputs and separate E5d PLY/NPZ outputs."""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=Path, required=True)
args = parser.parse_args()
stage = args.stage
inputs = list(csv.DictReader((stage / "artifacts/data/frames_e5d.csv").open(newline="")))
expected_sources = [int(row["frame_index"]) for row in inputs]
expected_stamps = [row["rgb_timestamp"] for row in inputs]
candidate_ids = [int(row["candidate_grid_index"]) for row in inputs]
if len(inputs) != 4 or candidate_ids != [0, 10, 20, 30]:
    raise SystemExit(f"Unexpected E5d candidate-grid selection: {candidate_ids}")

def check_npz(path, model):
    with np.load(path, allow_pickle=False) as data:
        source = data["source_frame_index"].astype(int).tolist()
        stamps = [str(value) for value in data["rgb_timestamp"].tolist()]
        if model == "VGGT" and source != expected_sources:
            raise ValueError(f"VGGT frame IDs differ from inputs: {source}")
        for got, expected in zip(stamps, [expected_stamps[expected_sources.index(i)] for i in source]):
            if abs(float(got) - float(expected)) > 1e-6:
                raise ValueError(f"{model} timestamp mismatch: {got} vs {expected}")
        return {"exported_count": len(source), "source_frame_indices": source,
                "timestamps": stamps}

exports = {}
vggt_npz = stage / "artifacts/exports/vggt/vggt_pointmaps.npz"
mast_npz = stage / "artifacts/exports/mast3r/mast3r_mf_pointmaps.npz"
if vggt_npz.is_file():
    exports["VGGT"] = check_npz(vggt_npz, "VGGT")
if mast_npz.is_file():
    exports["MASt3R-SLAM"] = check_npz(mast_npz, "MASt3R-SLAM")
    with np.load(mast_npz, allow_pickle=False) as data:
        exports["MASt3R-SLAM"]["fusion_count"] = data["fusion_count"].astype(int).tolist()
        exports["MASt3R-SLAM"]["update_count"] = data["update_count"].astype(int).tolist()

def check_ply(path):
    raw = path.read_bytes()
    marker = b"end_header\n"
    end = raw.find(marker)
    if end < 0:
        raise ValueError(f"Invalid PLY header: {path}")
    header = raw[:end + len(marker)].decode("ascii")
    m = re.search(r"^element vertex (\d+)$", header, re.MULTILINE)
    if not m or "format binary_little_endian 1.0" not in header:
        raise ValueError(f"Invalid PLY format: {path}")
    count = int(m.group(1))
    payload = raw[end + len(marker):]
    if count == 0 or len(payload) != count * 15:
        raise ValueError(f"Invalid PLY payload: {path} vertices={count}, bytes={len(payload)}")
    dtype = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                      "offsets": [0, 4, 8], "itemsize": 15})
    points = np.frombuffer(payload, dtype=dtype)
    xyz = np.column_stack([points[key] for key in ("x", "y", "z")])
    finite = np.isfinite(xyz).all(axis=1)
    if not finite.all():
        raise ValueError(f"Non-finite PLY XYZ: {path} ({int(finite.sum())}/{count})")
    return {"vertex_count": count, "finite_xyz_count": int(finite.sum()),
            "size_bytes": path.stat().st_size}

clouds = {}
for model, path in [
    ("VGGT V-P", stage / "artifacts/exports/vggt/HKairport01_E5d_V-P_N04.ply"),
    ("MASt3R-SLAM M-F", stage / "artifacts/exports/mast3r/HKairport01_E5d_M-F_N04.ply"),
]:
    if path.is_file(): clouds[model] = check_ply(path)
if len(exports.get("VGGT", {}).get("source_frame_indices", [])) != 4:
    raise SystemExit("VGGT did not export all four shared input frames")
if not clouds.get("VGGT V-P"):
    raise SystemExit("VGGT combined PLY is missing")
status = "passed" if "MASt3R-SLAM M-F" in clouds else "partial"
result = {
    "schema": "pointmap-comparison.e5d-validation.v1", "status": status,
    "same_rgb_inputs": True, "source_frame_indices": expected_sources,
    "timestamps": expected_stamps, "model_exports": exports,
    "separate_ply_outputs": clouds,
    "note": "MASt3R-SLAM may export a subset due to automatic keyframe selection; model point clouds retain separate coordinate frames.",
}
out = stage / "artifacts/validation/e5d_validation.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
