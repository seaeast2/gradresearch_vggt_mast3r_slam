#!/usr/bin/env python3
"""Select four E5 mapping frames at one-second intervals and build TUM input."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--mapping-csv", type=Path, required=True)
parser.add_argument("--selected-csv", type=Path, required=True)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--summary", type=Path, required=True)
args = parser.parse_args()
with args.mapping_csv.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 240:
    raise SystemExit(f"Expected the pinned 240-frame E5 mapping split, got {len(rows)}")
candidate_indices = [0, 5, 10, 15]
candidate_to_row = {int(row["candidate_grid_index"]): i for i, row in enumerate(rows)}
if not set(candidate_indices).issubset(candidate_to_row):
    raise SystemExit(f"Required one-second candidate grid indices are missing: {candidate_indices}")
row_indices = [candidate_to_row[index] for index in candidate_indices]
selected = [rows[index] for index in row_indices]
for row in selected:
    if not Path(row["rgb_path"]).is_file():
        raise FileNotFoundError(row["rgb_path"])
intervals = [float(selected[i+1]["rgb_timestamp"]) - float(selected[i]["rgb_timestamp"]) for i in range(3)]
if any(abs(delta - 1.0) > 0.01 for delta in intervals):
    raise SystemExit(f"Selected timestamps are not one second apart: {intervals}")
args.selected_csv.parent.mkdir(parents=True, exist_ok=True)
with args.selected_csv.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
    writer.writeheader()
    writer.writerows(selected)
args.dataset.mkdir(parents=True, exist_ok=False)
rgb = args.dataset / "rgb"
rgb.mkdir()
lines = ["# timestamp filename", "# E5c same four inputs for VGGT and MASt3R-SLAM"]
for row in selected:
    (rgb / row["filename"]).symlink_to(Path(row["rgb_path"]).resolve())
    lines.append(f"{row['rgb_timestamp']} rgb/{row['filename']}")
(args.dataset / "rgb.txt").write_text("\n".join(lines) + "\n")
result = {
    "schema": "pointmap-comparison.e5c-input.v1",
    "source_mapping_csv_sha256": hashlib.sha256(args.mapping_csv.read_bytes()).hexdigest(),
    "source_mapping_frame_count": len(rows), "input_frame_count": 4,
    "selection_rule": "first frame and subsequent mapping frames at 1 s intervals",
    "selected_mapping_row_indices": row_indices,
    "selected_candidate_grid_indices": candidate_indices,
    "source_frame_indices": [int(row["frame_index"]) for row in selected],
    "timestamps": [row["rgb_timestamp"] for row in selected],
    "interval_seconds": intervals,
    "rgb_paths": [row["rgb_path"] for row in selected],
    "rgb_sha256": [row["sha256"] for row in selected],
}
args.summary.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
