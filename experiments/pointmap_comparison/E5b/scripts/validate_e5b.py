#!/usr/bin/env python3
"""Validate E5b matched inputs, binary PLYs, and aligned pointmap arrays."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--stage", type=Path, required=True)
args = parser.parse_args()
stage = args.stage
cloud_dir = stage / "artifacts/exports/aligned"
summary = json.loads((stage / "artifacts/e5b_summary.json").read_text())
matches = json.loads((stage / "artifacts/data/keyframe_match_summary.json").read_text())
if summary["status"] != "completed" or not matches["exact_keyframe_id_match"]:
    raise SystemExit("E5b run status/keyframe-match validation failed")

dtype = np.dtype({"names": ["x", "y", "z", "r", "g", "b"],
                  "formats": ["<f4", "<f4", "<f4", "u1", "u1", "u1"],
                  "offsets": [0, 4, 8, 12, 13, 14], "itemsize": 15})
ply_results = {}
for name, summary_key in [
    ("HKairport01_E5b_M-F.ply", "mast3r_mf_vertices"),
    ("HKairport01_E5b_V-P_on_M-F_poses.ply", "vggt_vp_aligned_vertices"),
]:
    path = cloud_dir / name
    raw = path.read_bytes()
    marker = b"end_header\n"
    end = raw.find(marker)
    if end < 0:
        raise SystemExit(f"Invalid PLY header: {path}")
    header = raw[:end + len(marker)].decode("ascii")
    match = re.search(r"^element vertex (\d+)$", header, re.MULTILINE)
    if not match or "format binary_little_endian 1.0" not in header:
        raise SystemExit(f"Invalid PLY format/count: {path}")
    count = int(match.group(1))
    payload = raw[end + len(marker):]
    if len(payload) != count * dtype.itemsize or count != summary[summary_key]:
        raise SystemExit(f"PLY payload/count mismatch: {path}")
    points = np.frombuffer(payload, dtype=dtype)
    xyz = np.column_stack([points[axis] for axis in ("x", "y", "z")])
    finite = np.isfinite(xyz).all(axis=1)
    if count == 0 or not finite.all():
        raise SystemExit(f"PLY contains empty or non-finite XYZ vertices: {path}")
    ply_results[name] = {"vertex_count": count, "finite_xyz_count": int(finite.sum()),
                         "xyz_min": xyz.min(axis=0).tolist(), "xyz_max": xyz.max(axis=0).tolist(),
                         "size_bytes": path.stat().st_size}

npz_path = cloud_dir / "e5b_aligned_pointmaps.npz"
with np.load(npz_path, allow_pickle=False) as data:
    ids = data["dataset_frame_id"]
    stamps = data["rgb_timestamp"].tolist()
    if len(ids) != 18 or ids.tolist() != matches["frame_indices"] or stamps != matches["timestamps"]:
        raise SystemExit("Aligned NPZ frame identities do not match the E5 keyframes")
    masks = data["vggt_valid_mask"].astype(bool)
    vggt = data["vggt_V_P_on_mast3r_poses"]
    mast3r = data["mast3r_points_world"]
    if not np.isfinite(vggt[masks]).all():
        raise SystemExit("Non-finite valid VGGT pointmap values")
    m_mask = data["mast3r_valid_mask"].astype(bool)
    if not np.isfinite(mast3r[m_mask]).all():
        raise SystemExit("Non-finite valid MASt3R pointmap values")
    aligned_v_count = int(masks.sum())
    aligned_m_count = int(m_mask.sum())
if aligned_v_count != ply_results["HKairport01_E5b_V-P_on_M-F_poses.ply"]["vertex_count"]:
    raise SystemExit("VGGT valid-mask count does not match PLY vertex count")
if aligned_m_count != ply_results["HKairport01_E5b_M-F.ply"]["vertex_count"]:
    raise SystemExit("MASt3R valid-mask count does not match PLY vertex count")
batches = sorted((stage / "artifacts/exports/vggt").glob("batch_*/vggt_pointmaps.npz"))
if len(batches) != 5:
    raise SystemExit(f"Expected five VGGT batches, got {len(batches)}")

result = {
    "schema": "pointmap-comparison.e5b-validation.v1", "status": "passed",
    "same_keyframe_ids": True, "matched_keyframe_count": len(ids),
    "vggt_batch_count": len(batches), "ply_outputs": ply_results,
    "valid_vggt_aligned_points": aligned_v_count,
    "valid_mast3r_points": aligned_m_count,
    "interpretation_limit": "Same frame IDs and a shared MASt3R pose frame; native pixel grids differ, so vertex counts are not a pointwise accuracy comparison.",
}
out = stage / "artifacts/validation/e5b_validation.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
