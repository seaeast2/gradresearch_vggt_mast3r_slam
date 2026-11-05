#!/usr/bin/env python3
"""Normalize the run-local fused-keyframe hook export into the E1 M-F schema."""
import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def write_ply(path, xyz, rgb, mask):
    points = np.asarray(xyz, dtype=np.float32)[mask]
    colors = np.clip(np.asarray(rgb)[mask] * 255.0, 0, 255).astype(np.uint8)
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    with path.open("wb") as stream:
        stream.write(("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(points)}\n"
                      "property float x\nproperty float y\nproperty float z\n"
                      "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode("ascii"))
        for point, color in zip(points, colors):
            stream.write(struct.pack("<fffBBB", *point.tolist(), *color.tolist()))


parser = argparse.ArgumentParser()
parser.add_argument("--hook-export", type=Path, required=True)
parser.add_argument("--frame-list", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
source = np.load(args.hook_export, allow_pickle=False)
rows = list(csv.DictReader(args.frame_list.open(newline="")))
ids = source["dataset_frame_id"].astype(np.int32)
if np.any(ids < 0) or np.any(ids >= len(rows)):
    raise ValueError("Exported keyframe ID is outside the shared frame list")
for slot, frame_id in enumerate(ids):
    if abs(float(source["rgb_timestamp"][slot]) - float(rows[int(frame_id)]["rgb_timestamp"])) > 1e-6:
        raise ValueError(f"timestamp mismatch at dataset frame {frame_id}")
local = source["X_canon_local"].astype(np.float32)
world = source["points_world"].astype(np.float32)
confidence = source["confidence"].astype(np.float32)
rgb = source["rgb"].astype(np.float32)
mask = source["valid_mask"].astype(bool)
shape = source["image_shape_hw"].astype(np.int32)
source_shape = source["source_image_shape_hw"].astype(np.int32)
matrix = source["T_WC_matrix"].astype(np.float32)
valid = mask & np.isfinite(local).all(axis=-1) & np.isfinite(world).all(axis=-1) & np.isfinite(confidence)
args.output_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    args.output_dir / "mast3r_mf_pointmaps.npz",
    dataset_frame_id=ids, source_frame_index=source["source_frame_index"].astype(np.int32),
    rgb_timestamp=source["rgb_timestamp"], rgb_path=np.asarray([rows[int(i)]["rgb_path"] for i in ids]),
    image_shape_hw=shape, source_image_shape_hw=source_shape,
    X_canon_local=local, points_world=world, confidence=confidence, rgb=rgb,
    T_WC_matrix=matrix, fusion_count=source["fusion_count"].astype(np.int32),
    update_count=source["update_count"].astype(np.int32), valid_mask=valid,
    coordinate_frame=np.asarray("X_canon_local: canonical/camera points; points_world=T_WC.act(X_canon_local)"),
    unit=np.asarray("meter"), pixel_mapping=np.asarray("pixel-center convention; resize only for this 4:3 input"),
)
for slot, frame_id in enumerate(ids):
    stem = f"frame_{int(frame_id):06d}_{source['rgb_timestamp'][slot]}_M-F.ply"
    write_ply(args.output_dir / stem, world[slot], rgb[slot], valid[slot])
missing = sorted(set(range(len(rows))) - set(ids.tolist()))
meta = {
    "schema": "pointmap-comparison.v1", "model": "MASt3R-SLAM", "variant": "M-F fused keyframe",
    "input_frame_count": len(rows), "exported_keyframe_count": len(ids),
    "dataset_frame_ids": ids.tolist(), "missing_dataset_frame_ids": missing,
    "image_shape_hw": shape.tolist(), "source_image_shape_hw": source_shape.tolist(),
    "inverse_pixel_center_formula": "u=(u_net+0.5)*W/W_net-0.5; v=(v_net+0.5)*H/H_net-0.5",
    "fusion_count": source["fusion_count"].astype(int).tolist(), "update_count": source["update_count"].astype(int).tolist(),
    "outputs": ["mast3r_mf_pointmaps.npz", "*_M-F.ply"],
}
(args.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
