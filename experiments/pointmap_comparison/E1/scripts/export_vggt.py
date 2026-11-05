#!/usr/bin/env python3
"""Run pinned VGGT and export direct/depth pointmaps with source-frame identity."""
import argparse
import csv
import json
import struct
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def write_ply(path, xyz, rgb, mask):
    points = np.asarray(xyz, dtype=np.float32)[mask]
    colors = np.clip(np.asarray(rgb)[mask] * 255.0, 0, 255).astype(np.uint8)
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        header = ("ply\nformat binary_little_endian 1.0\n" f"element vertex {len(points)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        stream.write(header.encode("ascii"))
        for point, color in zip(points, colors):
            stream.write(struct.pack("<fffBBB", *point.tolist(), *color.tolist()))


parser = argparse.ArgumentParser()
parser.add_argument("--frame-list", type=Path, required=True)
parser.add_argument("--checkpoint-dir", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
rows = list(csv.DictReader(args.frame_list.open(newline="")))
if not rows:
    raise ValueError("Frame list is empty")
if len(rows) > 4:
    raise ValueError("E0b established N=4 as the safe maximum for this GPU and input size")
paths = [Path(row["rgb_path"]).resolve() for row in rows]
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; run E1 on the host outside the sandbox")
device = torch.device("cuda:0")
dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
model = VGGT()
checkpoint = args.checkpoint_dir / "model.pt"
state = torch.load(checkpoint, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval().to(device)
torch.cuda.reset_peak_memory_stats(device)
images = load_and_preprocess_images([str(p) for p in paths], mode="pad").to(device)
with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
    pred = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    depth = pred["depth"]
    v_d = unproject_depth_map_to_point_map(depth.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))
torch.cuda.synchronize(device)

def cpu_float(value):
    return value.detach().to(device="cpu", dtype=torch.float32).numpy()

v_p = cpu_float(pred["world_points"].squeeze(0))
v_p_conf = cpu_float(pred["world_points_conf"].squeeze(0))
v_d = np.asarray(v_d, dtype=np.float32)
depth_np = cpu_float(depth.squeeze(0)).squeeze(-1)
depth_conf = cpu_float(pred["depth_conf"].squeeze(0))
extrinsic_np, intrinsic_np = cpu_float(extrinsic.squeeze(0)), cpu_float(intrinsic.squeeze(0))
images_np = images.detach().to(device="cpu", dtype=torch.float32).numpy()
network_h, network_w = images_np.shape[-2:]
source_hw = np.asarray([[Image.open(p).height, Image.open(p).width] for p in paths], dtype=np.int32)
resized_hw_values = []
for h, w in source_hw:
    if w >= h:
        resized_hw_values.append([round(h * (518 / w) / 14) * 14, 518])
    else:
        resized_hw_values.append([518, round(w * (518 / h) / 14) * 14])
resized_hw = np.asarray(resized_hw_values, dtype=np.int32)
pad_tblr = np.asarray([[(network_h-rh)//2, network_h-rh-(network_h-rh)//2, (network_w-rw)//2, network_w-rw-(network_w-rw)//2] for rh, rw in resized_hw], dtype=np.int32)
valid = np.zeros((len(rows), network_h, network_w), dtype=bool)
for i, (rh, rw) in enumerate(resized_hw):
    top, bottom, left, right = pad_tblr[i]
    valid[i, top:network_h-bottom if bottom else network_h, left:network_w-right if right else network_w] = True
valid &= np.isfinite(v_p).all(axis=-1) & np.isfinite(v_d).all(axis=-1) & np.isfinite(depth_np) & (depth_np > 0)
args.output_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    args.output_dir / "vggt_pointmaps.npz",
    dataset_frame_id=np.arange(len(rows), dtype=np.int32),
    source_frame_index=np.asarray([int(r["frame_index"]) for r in rows], dtype=np.int32),
    rgb_timestamp=np.asarray([r["rgb_timestamp"] for r in rows]),
    rgb_path=np.asarray([str(p) for p in paths]),
    source_image_shape_hw=source_hw, image_shape_hw=np.tile([network_h, network_w], (len(rows), 1)),
    resized_image_shape_hw=resized_hw, padding_tblr=pad_tblr,
    input_rgb=images_np, V_P=v_p, V_P_conf=v_p_conf, V_D=v_d,
    depth=depth_np, depth_conf=depth_conf, extrinsic=extrinsic_np, intrinsic=intrinsic_np,
    valid_mask=valid,
    coordinate_frame=np.asarray("V_P and V_D are VGGT predicted world-coordinate pointmaps; camera tensors use VGGT convention"),
    unit=np.asarray("meter"), pixel_mapping=np.asarray("pixel-center convention; see metadata.json"),
)
for i, row in enumerate(rows):
    stem = f"frame_{int(row['frame_index']):06d}_{row['rgb_timestamp']}"
    write_ply(args.output_dir / f"{stem}_V-P.ply", v_p[i], images_np[i].transpose(1, 2, 0), valid[i])
    write_ply(args.output_dir / f"{stem}_V-D.ply", v_d[i], images_np[i].transpose(1, 2, 0), valid[i])
metadata = {
    "schema": "pointmap-comparison.v1", "model": "facebook/VGGT-1B", "count": len(rows),
    "network_shape_hw": [network_h, network_w], "source_shape_hw": source_hw.tolist(),
    "padding_tblr": pad_tblr.tolist(), "resize_mode": "mode=pad; bicubic; white padding",
    "inverse_pixel_center_formula": "u=(u_net-pad_left+0.5)*W/resized_W-0.5; v=(v_net-pad_top+0.5)*H/resized_H-0.5",
    "dtype": str(dtype), "gpu": torch.cuda.get_device_name(device),
    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    "outputs": ["vggt_pointmaps.npz", "*_V-P.ply", "*_V-D.ply"],
}
(args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
