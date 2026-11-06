#!/usr/bin/env python3
"""Validate replayed MASt3R keyframes against E5 and create VGGT frame CSV."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--mast3r-npz", type=Path, required=True)
parser.add_argument("--baseline-csv", type=Path, required=True)
parser.add_argument("--all-frames-csv", type=Path, required=True)
parser.add_argument("--output-csv", type=Path, required=True)
parser.add_argument("--summary", type=Path, required=True)
args = parser.parse_args()
baseline = list(csv.DictReader(args.baseline_csv.open(newline="")))
all_frames = list(csv.DictReader(args.all_frames_csv.open(newline="")))
data = np.load(args.mast3r_npz, allow_pickle=False)
ids = data["dataset_frame_id"].astype(int).tolist()
stamps = [str(value) for value in data["rgb_timestamp"].tolist()]
expected_ids = [int(row["frame_index"]) for row in baseline]
expected_stamps = [row["rgb_timestamp"] for row in baseline]
if ids != expected_ids or len(stamps) != len(expected_stamps):
    raise SystemExit(f"Replayed keyframe IDs differ from E5: expected {expected_ids}; got {ids}")
for actual, expected in zip(stamps, expected_stamps):
    if abs(float(actual) - float(expected)) > 1e-6:
        raise SystemExit(f"Replayed keyframe timestamp differs from E5: expected {expected}; got {actual}")
rows_by_id = {int(row["frame_index"]): row for row in all_frames}
selected = [rows_by_id[index] for index in ids]
args.output_csv.parent.mkdir(parents=True, exist_ok=True)
with args.output_csv.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
    writer.writeheader()
    writer.writerows(selected)
summary = {
    "schema": "pointmap-comparison.e5b-matched-keyframes.v1",
    "baseline_e5_keyframe_count": len(expected_ids),
    "replayed_mast3r_keyframe_count": len(ids),
    "exact_keyframe_id_match": ids == expected_ids,
    "frame_indices": ids,
    "timestamps": stamps,
    "rgb_paths": [row["rgb_path"] for row in selected],
}
args.summary.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
