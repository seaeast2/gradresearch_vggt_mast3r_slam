#!/usr/bin/env python3
"""Minimal, reproducible E0 VGGT inference on the shared four-frame RGB subset."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

parser = argparse.ArgumentParser()
parser.add_argument("--frame-list", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--checkpoint-dir", required=True, type=Path)
args = parser.parse_args()

with args.frame_list.open(newline="") as f:
    rows = list(csv.DictReader(f))
image_paths = [Path(row["rgb_path"]) for row in rows]
frame_ids = [row["rgb_timestamp"] for row in rows]
if len(image_paths) != 4:
    raise ValueError(f"E0 smoke run expects exactly four shared frames; got {len(image_paths)}")

if not torch.cuda.is_available():
    raise RuntimeError("E0 inference must run on the host CUDA device; torch.cuda.is_available() is false")

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
device = torch.device("cuda:0")
dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
model = VGGT()
checkpoint = args.checkpoint_dir / "model.pt"
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
state = torch.load(checkpoint, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval().to(device)
images = load_and_preprocess_images([str(p) for p in image_paths], mode="pad").to(device)

with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
    pred = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    depth = pred["depth"]
    v_d = unproject_depth_map_to_point_map(depth.squeeze(0), extrinsic.squeeze(0), intrinsic.squeeze(0))

args.output.parent.mkdir(parents=True, exist_ok=True)
def to_numpy(tensor):
    return tensor.detach().to(device="cpu", dtype=torch.float32).numpy()

np.savez_compressed(
    args.output,
    frame_ids=np.asarray(frame_ids),
    original_rgb_paths=np.asarray([str(p) for p in image_paths]),
    input_images=images.cpu().numpy(),
    V_P=to_numpy(pred["world_points"].squeeze(0)),
    V_P_conf=to_numpy(pred["world_points_conf"].squeeze(0)),
    V_D=np.asarray(v_d, dtype=np.float32),
    depth=to_numpy(depth.squeeze(0)),
    depth_conf=to_numpy(pred["depth_conf"].squeeze(0)),
    extrinsic=to_numpy(extrinsic.squeeze(0)),
    intrinsic=to_numpy(intrinsic.squeeze(0)),
)
print(json.dumps({
    "device": torch.cuda.get_device_name(device),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "dtype": str(dtype),
    "frames": frame_ids,
    "input_shape": list(images.shape),
    "V_P_shape": list(pred["world_points"].shape),
    "V_D_shape": list(v_d.shape),
    "output": str(args.output),
    "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
}, indent=2))
