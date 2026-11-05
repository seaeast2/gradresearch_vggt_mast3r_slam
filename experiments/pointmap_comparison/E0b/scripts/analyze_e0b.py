#!/usr/bin/env python3
"""Measure E0b input transforms, TUM development prefix geometry, and depth support."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--e0-stage", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
e0 = args.e0_stage.resolve()
associated = list(csv.DictReader((e0 / "artifacts/data/frames_associated.csv").open(newline="")))
valid = [row for row in associated if row["within_tolerance"].lower() == "true"]
centers = np.asarray([list(map(float, row["groundtruth_pose_xyzw"].split()[:3])) for row in valid])
prefixes = {}
for fraction in (0.05, 0.10, 0.20, 0.25):
    n = math.ceil(len(valid) * fraction)
    points = centers[:n]
    eigenvalues = np.linalg.eigvalsh(np.cov(points.T))[::-1]
    span = np.linalg.norm(points - points[0], axis=1)
    prefixes[str(n)] = {
        "fraction": fraction,
        "first_rgb_timestamp": valid[0]["rgb_timestamp"],
        "last_rgb_timestamp": valid[n - 1]["rgb_timestamp"],
        "max_displacement_from_start_m": float(span.max()),
        "center_covariance_eigenvalues_m2_desc": eigenvalues.tolist(),
        "lambda2_over_lambda1": float(eigenvalues[1] / eigenvalues[0]),
        "lambda3_over_lambda1": float(eigenvalues[2] / eigenvalues[0]),
    }

# Verify transformations against the pinned, saved source and E0 output shapes.
vggt_source = e0 / "artifacts/sources/vggt/vggt/utils/load_fn.py"
mast3r_source = e0 / "artifacts/sources/mast3r_slam/mast3r_slam/mast3r_utils.py"
e0_npz = np.load(e0 / "artifacts/demo_outputs/vggt_e0_predictions.npz")
vggt_hw = list(e0_npz["input_images"].shape[-2:])
dataset_dir = e0 / "artifacts/data/rgbd_dataset_freiburg1_desk"
sample_rgb = Image.open(dataset_dir / valid[0]["rgb_file"])
original_w, original_h = sample_rgb.size
vggt_scale = 518 / max(original_w, original_h)
vggt_resized_w = 518 if original_w >= original_h else round(original_w * vggt_scale / 14) * 14
vggt_resized_h = 518 if original_h >= original_w else round(original_h * vggt_scale / 14) * 14
vggt_pad_left = (518 - vggt_resized_w) // 2
vggt_pad_top = (518 - vggt_resized_h) // 2
e0a_stage = e0.parents[2] / "tum-freiburg1_desk-20261105-005/stages/E0a"
mast3r_npz = np.load(e0a_stage / "artifacts/demo_outputs/mast3r_e0a/mast3r_e0a_fused_keyframes.npz")
mast3r_hw = list(mast3r_npz["X_canon_local"].shape[1:3])
result = {
    "dataset": "TUM RGB-D freiburg1_desk development sequence",
    "valid_rgb_depth_pose_matches": len(valid),
    "prefix_geometry": prefixes,
    "input_mapping": {
        "original_wh": [original_w, original_h],
        "vggt": {
            "source_file": str(vggt_source), "mode": "pad", "tensor_hw": vggt_hw,
            "resized_wh": [vggt_resized_w, vggt_resized_h],
            "padding_ltrb": [vggt_pad_left, 518-vggt_resized_w-vggt_pad_left, vggt_pad_top, 518-vggt_resized_h-vggt_pad_top],
            "inverse_pixel_center": "u=(u_net-pad_left+0.5)*W/resized_W-0.5; v=(v_net-pad_top+0.5)*H/resized_H-0.5",
            "implementation_note": "bicubic resize (PIL); white pad in pinned source",
        },
        "mast3r_slam": {
            "source_file": str(mast3r_source), "pointmap_hw": mast3r_hw,
            "scale_xy": [original_w/mast3r_hw[1], original_h/mast3r_hw[0]],
            "inverse_pixel_center": "u=(u_net+0.5)*W/W_net-0.5; v=(v_net+0.5)*H/H_net-0.5",
            "implementation_note": "long-side resize to 512 then crop to multiples of 8; this 4:3 input is unchanged by crop",
        },
    },
    "gt_depth": {"encoding": "uint16 PNG, metric_m=raw/5000 per TUM benchmark format", "valid_range_m": [0.3, 5.0]},
    "protocol_proposal": {
        "alignment_prefix": "first 20% of RGB frames with depth+pose association within 20ms; ceil(N*0.20), minimum 30",
        "degenerate_if": "fewer than 30 matches, max center displacement <0.5m, or lambda2/lambda1 <0.01",
        "primary_point_error_tolerance_m": 0.05,
        "surface_outlier_tau_m": 0.05,
        "free_space_gt_uncertainty_margin_m": 0.02,
        "common_voxel_size_m": 0.02,
        "gt_valid_depth_m": [0.3, 5.0],
        "sensitivity": "also report point/F-score thresholds at 0.02m and 0.10m; retain GT valid-pixel counts",
    },
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
