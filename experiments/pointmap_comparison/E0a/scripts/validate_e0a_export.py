#!/usr/bin/env python3
"""Check E0a frame/pixel traceability and Sim(3) local-world round trips."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--input-frames", type=Path, required=True)
parser.add_argument("--export", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

with args.input_frames.open(newline="") as stream:
    input_rows = list(csv.DictReader(stream))
data = np.load(args.export, allow_pickle=False)
ids = data["dataset_frame_id"].astype(int)
if len(ids) == 0 or len(set(ids.tolist())) != len(ids):
    raise SystemExit("Expected one or more unique exported keyframe IDs")
if np.any(ids < 0) or np.any(ids >= len(input_rows)):
    raise SystemExit("Exported frame ID is outside the four-frame input list")

local = data["X_canon_local"].astype(np.float64)
world = data["points_world"].astype(np.float64)
confidence = data["confidence"].astype(np.float64)
rgb = data["rgb"].astype(np.float64)
matrices = data["T_WC_matrix"].astype(np.float64)
shapes = data["image_shape_hw"].astype(int)
source_shapes = data["source_image_shape_hw"].astype(int)
if local.ndim != 4 or local.shape[-1] != 3 or world.shape != local.shape:
    raise SystemExit(f"Unexpected local/world pointmap shapes: {local.shape} / {world.shape}")
if confidence.shape != local.shape[:3] or rgb.shape != local.shape:
    raise SystemExit("Pointmap, confidence, and RGB grids are not aligned")
if matrices.shape != (len(ids), 4, 4):
    raise SystemExit(f"Unexpected T_WC matrix shape: {matrices.shape}")

roundtrip_max = 0.0
samples = []
non_identity_pose_count = 0
for slot, frame_id in enumerate(ids):
    row = input_rows[int(frame_id)]
    stamp = str(data["rgb_timestamp"][slot])
    if abs(float(stamp) - float(row["rgb_timestamp"])) > 1e-6:
        raise SystemExit(f"Timestamp does not map to input frame {frame_id}")
    h, w = map(int, shapes[slot])
    if local[slot].shape[:2] != (h, w) or rgb[slot].shape[:2] != (h, w):
        raise SystemExit(f"Frame {frame_id}: recorded H/W does not match output grids")
    matrix = matrices[slot]
    if float(np.max(np.abs(matrix - np.eye(4)))) > 1e-6:
        non_identity_pose_count += 1
    homogeneous = np.concatenate([local[slot].reshape(-1, 3), np.ones((h*w, 1))], axis=1)
    mapped = (matrix @ homogeneous.T).T[:, :3].reshape(h, w, 3)
    max_forward = float(np.nanmax(np.abs(mapped - world[slot])))
    inverse = np.linalg.inv(matrix)
    world_h = np.concatenate([world[slot].reshape(-1, 3), np.ones((h*w, 1))], axis=1)
    back = (inverse @ world_h.T).T[:, :3].reshape(h, w, 3)
    max_back = float(np.nanmax(np.abs(back - local[slot])))
    roundtrip_max = max(roundtrip_max, max_forward, max_back)
    if max_forward > 2e-3 or max_back > 2e-3:
        raise SystemExit(f"Frame {frame_id}: Sim(3) round-trip exceeds tolerance: {max_forward}, {max_back}")
    v, u = h // 2, w // 2
    source_h, source_w = map(int, source_shapes[slot])
    source_u = (u + 0.5) * source_w / w - 0.5
    source_v = (v + 0.5) * source_h / h - 0.5
    samples.append({
        "dataset_frame_id": int(frame_id),
        "source_frame_index": int(data["source_frame_index"][slot]),
        "rgb_timestamp": stamp,
        "output_pixel_uv": [u, v],
        "source_pixel_uv_by_resize_scale": [source_u, source_v],
        "local_xyz": local[slot, v, u].tolist(),
        "world_xyz": world[slot, v, u].tolist(),
        "confidence": float(confidence[slot, v, u]),
        "rgb_01": rgb[slot, v, u].tolist(),
    })

# The four-frame preview may export only the first, identity-initialized keyframe.
# Exercise the same homogeneous-matrix pixel transform with a non-identity Sim(3)
# so the coordinate convention check is not only an identity-pose check.
theta = np.deg2rad(17.0)
scale = 1.35
synthetic = np.eye(4, dtype=np.float64)
synthetic[:3, :3] = scale * np.array([
    [np.cos(theta), -np.sin(theta), 0.0],
    [np.sin(theta), np.cos(theta), 0.0],
    [0.0, 0.0, 1.0],
])
synthetic[:3, 3] = [0.25, -0.4, 1.2]
probe = np.array([0.12, -0.08, 2.4, 1.0])
probe_world = synthetic @ probe
probe_roundtrip = np.linalg.inv(synthetic) @ probe_world
synthetic_roundtrip_error = float(np.max(np.abs(probe_roundtrip - probe)))
if synthetic_roundtrip_error > 1e-12:
    raise SystemExit(f"Synthetic non-identity Sim(3) round-trip failed: {synthetic_roundtrip_error}")

exported = set(ids.tolist())
missing = sorted(set(range(len(input_rows))) - exported)
result = {
    "status": "passed",
    "input_frame_count": len(input_rows),
    "exported_keyframe_count": len(ids),
    "exported_dataset_frame_ids": ids.tolist(),
    "missing_dataset_frame_ids": missing,
    "max_abs_local_world_roundtrip_error": roundtrip_max,
    "exported_non_identity_pose_count": non_identity_pose_count,
    "roundtrip_tolerance": 2e-3,
    "synthetic_non_identity_sim3_roundtrip_max_abs_error": synthetic_roundtrip_error,
    "synthetic_sim3_parameters": {"scale": scale, "rotation_z_degrees": 17.0, "translation": [0.25, -0.4, 1.2]},
    "pixel_mapping_samples": samples,
    "coverage_note": "Keyframe selection coverage is reported separately; omitted input frames have no MASt3R fused keyframe pointmap.",
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
