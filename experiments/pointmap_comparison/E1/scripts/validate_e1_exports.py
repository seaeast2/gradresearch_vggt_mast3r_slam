#!/usr/bin/env python3
"""Validate E1 export schemas, pixel correspondence, and Sim(3) round trips."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--vggt", type=Path, required=True)
parser.add_argument("--mast3r", type=Path, required=True)
parser.add_argument("--frame-list", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
v = np.load(args.vggt, allow_pickle=False)
m = np.load(args.mast3r, allow_pickle=False)
rows = list(csv.DictReader(args.frame_list.open(newline="")))
assert v["V_P"].shape == v["V_D"].shape and v["V_P"].shape[-1] == 3
assert v["V_P_conf"].shape == v["depth"].shape == v["depth_conf"].shape == v["valid_mask"].shape
assert v["V_P"].shape[:3] == v["input_rgb"].shape[:1] + v["input_rgb"].shape[2:]
assert m["X_canon_local"].shape == m["points_world"].shape == m["rgb"].shape
assert m["confidence"].shape == m["valid_mask"].shape == m["X_canon_local"].shape[:-1]
assert np.allclose(v["source_image_shape_hw"], [[480, 640]] * len(v["dataset_frame_id"]))

# Verify the VGGT pixel-center mapping and inverse mapping at corners and center.
vmeta = json.loads((args.vggt.parent / "metadata.json").read_text())
top, bottom, left, right = vmeta["padding_tblr"][0]
resized_h, resized_w = v["resized_image_shape_hw"][0]
source_h, source_w = v["source_image_shape_hw"][0]
pixel_roundtrip = 0.0
for u_net, v_net in [(left, top), (left + resized_w/2, top + resized_h/2), (left + resized_w-1, top + resized_h-1)]:
    u_src = (u_net-left+0.5)*source_w/resized_w-0.5
    v_src = (v_net-top+0.5)*source_h/resized_h-0.5
    u_back = (u_src+0.5)*resized_w/source_w-0.5+left
    v_back = (v_src+0.5)*resized_h/source_h-0.5+top
    pixel_roundtrip = max(pixel_roundtrip, abs(u_back-u_net), abs(v_back-v_net))
assert pixel_roundtrip < 1e-9

# Recreate the source image transforms and check exported RGB at corresponding pixels.
vggt_rgb_error = 0.0
mast3r_rgb_error = 0.0
for slot, image_path in enumerate(v["rgb_path"]):
    source = Image.open(str(image_path)).convert("RGB")
    rh, rw = map(int, v["resized_image_shape_hw"][slot])
    resized = source.resize((rw, rh), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (int(v["V_P"].shape[2]), int(v["V_P"].shape[1])), (255,255,255))
    left, top = map(int, [v["padding_tblr"][slot][2], v["padding_tblr"][slot][0]])
    canvas.paste(resized, (left,top))
    expected = np.asarray(canvas, dtype=np.float32) / 255.0
    observed = v["input_rgb"][slot].transpose(1,2,0)
    vggt_rgb_error = max(vggt_rgb_error, float(np.max(np.abs(expected-observed))))
    assert vggt_rgb_error <= 1e-6
for slot, image_path in enumerate(m["rgb_path"]):
    source = Image.open(str(image_path)).convert("RGB")
    h, w = map(int, m["image_shape_hw"][slot])
    resampling = Image.Resampling.LANCZOS if max(source.size) > max(w,h) else Image.Resampling.BICUBIC
    expected = np.asarray(source.resize((w,h), resampling), dtype=np.float32)/255.0
    mast3r_rgb_error = max(mast3r_rgb_error, float(np.max(np.abs(expected-m["rgb"][slot]))))
    assert mast3r_rgb_error <= 1e-6

mast3r_pixel_roundtrip = 0.0
for net_u, net_v in [(0,0),(w/2,h/2),(w-1,h-1)]:
    src_u = (net_u+0.5)*source_w/w-0.5
    src_v = (net_v+0.5)*source_h/h-0.5
    net_u_back = (src_u+0.5)*w/source_w-0.5
    net_v_back = (src_v+0.5)*h/source_h-0.5
    mast3r_pixel_roundtrip=max(mast3r_pixel_roundtrip,abs(net_u_back-net_u),abs(net_v_back-net_v))
assert mast3r_pixel_roundtrip < 1e-9

sample_records = []
for slot, frame_id in enumerate(m["dataset_frame_id"].astype(int)):
    frame_id = int(frame_id)
    h, w = map(int, m["image_shape_hw"][slot])
    assert m["X_canon_local"][slot].shape[:2] == (h, w)
    assert m["rgb"][slot].shape[:2] == (h, w)
    assert m["rgb_timestamp"][slot] == rows[frame_id]["rgb_timestamp"]
    matrix = m["T_WC_matrix"][slot].astype(np.float64)
    local = m["X_canon_local"][slot, h//2, w//2].astype(np.float64)
    expected = (matrix @ np.r_[local, 1.0])[:3]
    observed = m["points_world"][slot, h//2, w//2].astype(np.float64)
    assert np.max(np.abs(expected-observed)) < 2e-3
    back = (np.linalg.inv(matrix) @ np.r_[observed, 1.0])[:3]
    assert np.max(np.abs(back-local)) < 2e-3
    source_h, source_w = map(int, m["source_image_shape_hw"][slot])
    source_u = (w//2+0.5)*source_w/w-0.5
    source_v = (h//2+0.5)*source_h/h-0.5
    sample_records.append({"dataset_frame_id": frame_id, "rgb_timestamp": str(m["rgb_timestamp"][slot]),
                           "pixel_uv": [w//2,h//2], "source_pixel_uv": [source_u,source_v],
                           "local_xyz": local.tolist(), "world_xyz": observed.tolist(),
                           "confidence": float(m["confidence"][slot,h//2,w//2]),
                           "rgb_01": m["rgb"][slot,h//2,w//2].astype(float).tolist()})

vggt_samples = []
for slot, frame_id in enumerate(v["dataset_frame_id"].astype(int)):
    frame_id = int(frame_id)
    assert 0 <= frame_id < len(rows)
    assert str(v["rgb_timestamp"][slot]) == rows[frame_id]["rgb_timestamp"]
    h, w = map(int, v["image_shape_hw"][slot])
    u, vv = w//2, h//2
    pad_top, _, pad_left, _ = map(int, v["padding_tblr"][slot])
    resized_h, resized_w = map(int, v["resized_image_shape_hw"][slot])
    src_h, src_w = map(int, v["source_image_shape_hw"][slot])
    src_u=(u-pad_left+0.5)*src_w/resized_w-0.5
    src_v=(vv-pad_top+0.5)*src_h/resized_h-0.5
    assert v["valid_mask"][slot,vv,u]
    vggt_samples.append({"dataset_frame_id":frame_id,"rgb_timestamp":str(v["rgb_timestamp"][slot]),
        "net_pixel_uv":[u,vv],"source_pixel_uv":[src_u,src_v],
        "V_P_xyz":v["V_P"][slot,vv,u].astype(float).tolist(),
        "V_D_xyz":v["V_D"][slot,vv,u].astype(float).tolist(),
        "V_P_confidence":float(v["V_P_conf"][slot,vv,u]),
        "depth_m":float(v["depth"][slot,vv,u]),
        "depth_confidence":float(v["depth_conf"][slot,vv,u]),
        "rgb_01":v["input_rgb"][slot,:,vv,u].astype(float).tolist()})
paired_pixel_differences = []
vggt_by_id = {sample["dataset_frame_id"]: sample for sample in vggt_samples}
for sample in sample_records:
    frame_id = sample["dataset_frame_id"]
    if frame_id in vggt_by_id:
        vg = vggt_by_id[frame_id]
        delta = np.asarray(vg["source_pixel_uv"]) - np.asarray(sample["source_pixel_uv"])
        difference = float(np.linalg.norm(delta))
        assert difference < 0.02, f"Representative source pixels diverge by {difference:.6f} px"
        paired_pixel_differences.append({"dataset_frame_id":frame_id,"rgb_timestamp":sample["rgb_timestamp"],
            "vggt_source_pixel_uv":vg["source_pixel_uv"],"mast3r_source_pixel_uv":sample["source_pixel_uv"],
            "euclidean_difference_px":difference})
assert paired_pixel_differences, "No overlapping VGGT/M-F frame to verify common source pixel"

# Exercise non-identity scale/rotation/translation, independent of the identity E0a keyframe pose.
theta = np.deg2rad(23.0); scale = 1.7
sim3 = np.eye(4); sim3[:3,:3] = scale*np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
sim3[:3,3] = [0.4,-0.2,1.1]
probe = np.array([0.2,-0.3,2.0,1.0]); sim3_error = float(np.max(np.abs(np.linalg.inv(sim3)@(sim3@probe)-probe)))
assert sim3_error < 1e-12
result = {"status":"passed", "vggt_input_frames":len(v["dataset_frame_id"]),
          "mast3r_input_frames":len(rows), "mast3r_exported_keyframes":len(m["dataset_frame_id"]),
          "mast3r_missing_dataset_frame_ids":sorted(set(range(len(rows)))-set(m["dataset_frame_id"].astype(int).tolist())),
          "vggt_pointmap_grid":list(v["V_P"].shape[1:3]), "mast3r_pointmap_grids":m["image_shape_hw"].tolist(),
          "pixel_mapping_roundtrip_max_error_px":{"vggt":pixel_roundtrip,"mast3r":mast3r_pixel_roundtrip},
          "rgb_transform_max_abs_error":{"vggt":vggt_rgb_error,"mast3r":mast3r_rgb_error},
          "paired_source_pixel_correspondence":paired_pixel_differences,
          "synthetic_nonidentity_sim3_roundtrip_max_error":sim3_error,
          "vggt_center_pixel_samples":vggt_samples,
          "mast3r_center_pixel_samples":sample_records,
          "note":"Coverage is keyframe-selected subset; this check does not claim dense per-input MASt3R output."}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
