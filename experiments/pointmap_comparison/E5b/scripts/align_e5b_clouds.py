#!/usr/bin/env python3
"""Re-express batched VGGT V-P maps using matched MASt3R camera poses."""
import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--mast3r-npz", type=Path, required=True)
parser.add_argument("--vggt-batch-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
m = np.load(args.mast3r_npz, allow_pickle=False)
m_ids = m["dataset_frame_id"].astype(np.int32)
m_stamps = [str(value) for value in m["rgb_timestamp"].tolist()]
m_slot = {stamp: i for i, stamp in enumerate(m_stamps)}
m_world = m["points_world"].astype(np.float32)
m_local = m["X_canon_local"].astype(np.float32)
m_rgb = m["rgb"].astype(np.float32)
m_valid = m["valid_mask"].astype(bool)
m_pose = m["T_WC_matrix"].astype(np.float64)

v_points, v_colors, v_masks = [], [], []
seen = set()
for batch_path in sorted(args.vggt_batch_root.glob("batch_*/vggt_pointmaps.npz")):
    with np.load(batch_path, allow_pickle=False) as batch:
        stamps = [str(value) for value in batch["rgb_timestamp"].tolist()]
        xyz_world = batch["V_P"].astype(np.float32)
        extrinsics = batch["extrinsic"].astype(np.float64)
        images = batch["input_rgb"].astype(np.float32)
        masks = batch["valid_mask"].astype(bool)
    for j, stamp in enumerate(stamps):
        if stamp not in m_slot or stamp in seen:
            raise ValueError(f"Unexpected or duplicate VGGT timestamp {stamp}")
        seen.add(stamp)
        slot = m_slot[stamp]
        # VGGT extrinsic is world-to-camera; invert it to camera-local points.
        r = extrinsics[j, :, :3]
        t = extrinsics[j, :, 3]
        xyz_camera = np.einsum("ij,hwj->hwi", r.T, xyz_world[j] - t)
        # Apply the matched MASt3R Sim(3) camera pose to both maps.
        pose = m_pose[slot]
        xyz_aligned = np.einsum("ij,hwj->hwi", pose[:3, :3], xyz_camera) + pose[:3, 3]
        valid = masks[j] & np.isfinite(xyz_aligned).all(axis=-1)
        v_points.append(xyz_aligned.astype(np.float32))
        v_colors.append(np.transpose(images[j], (1, 2, 0)))
        v_masks.append(valid)
if seen != set(m_stamps):
    raise ValueError(f"VGGT/MASt3R keyframe timestamp coverage differs: missing={sorted(set(m_stamps)-seen)}")

args.output_dir.mkdir(parents=True, exist_ok=True)
def write_ply(path, xyz, rgb, mask):
    pts = np.asarray(xyz, np.float32)[mask]
    cols = np.clip(np.asarray(rgb)[mask] * 255.0, 0, 255).astype(np.uint8)
    finite = np.isfinite(pts).all(axis=1)
    pts, cols = pts[finite], cols[finite]
    with path.open("wb") as stream:
        header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(pts)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        stream.write(header.encode("ascii"))
        for p, c in zip(pts, cols):
            stream.write(struct.pack("<fffBBB", *p.tolist(), *c.tolist()))
    return int(len(pts))

mast3r_count = write_ply(args.output_dir / "HKairport01_E5b_M-F.ply", m_world, m_rgb, m_valid)
vggt_count = write_ply(args.output_dir / "HKairport01_E5b_V-P_on_M-F_poses.ply", np.stack(v_points), np.stack(v_colors), np.stack(v_masks))
np.savez_compressed(
    args.output_dir / "e5b_aligned_pointmaps.npz",
    dataset_frame_id=m_ids, rgb_timestamp=np.asarray(m_stamps),
    mast3r_X_canon_local=m_local, mast3r_points_world=m_world,
    mast3r_confidence=m["confidence"], mast3r_valid_mask=m_valid,
    vggt_V_P_on_mast3r_poses=np.stack(v_points), vggt_valid_mask=np.stack(v_masks),
    rgb=m_rgb, mast3r_T_WC_matrix=m_pose.astype(np.float32),
    coordinate_note=np.asarray("V-P converted by inverse VGGT world-to-camera pose into camera-local coordinates, then transformed with matched MASt3R T_WC Sim(3); M-F points are T_WC.act(X_canon_local)"),
)
summary = {
    "schema": "pointmap-comparison.e5b-clouds.v1",
    "matched_keyframes": len(m_stamps),
    "frame_indices": m_ids.tolist(), "timestamps": m_stamps,
    "mast3r_mf_vertices": mast3r_count,
    "vggt_vp_aligned_vertices": vggt_count,
    "mast3r_valid_pixels_per_frame": m_valid.reshape(len(m_stamps), -1).sum(axis=1).astype(int).tolist(),
    "vggt_valid_pixels_per_frame": np.stack(v_masks).reshape(len(v_masks), -1).sum(axis=1).astype(int).tolist(),
    "shared_pose_frame": "MASt3R-SLAM T_WC Sim(3) for the matched keyframe; visualization registration only",
    "outputs": ["HKairport01_E5b_M-F.ply", "HKairport01_E5b_V-P_on_M-F_poses.ply", "e5b_aligned_pointmaps.npz"],
}
(args.output_dir / "cloud_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
