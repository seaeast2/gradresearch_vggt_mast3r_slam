#!/usr/bin/env python3
"""Select the four user-specified E5 mapping images and build TUM input."""
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
filenames = ["000050.jpg", "000070.jpg", "000090.jpg", "000110.jpg"]
filename_to_row = {row["filename"]: i for i, row in enumerate(rows)}
if not set(filenames).issubset(filename_to_row):
    raise SystemExit(f"Required mapping filenames are missing: {filenames}")
row_indices = [filename_to_row[name] for name in filenames]
selected = [rows[index] for index in row_indices]
for row in selected:
    if not Path(row["rgb_path"]).is_file():
        raise FileNotFoundError(row["rgb_path"])
intervals = [float(selected[i+1]["rgb_timestamp"]) - float(selected[i]["rgb_timestamp"]) for i in range(3)]
args.selected_csv.parent.mkdir(parents=True, exist_ok=True)
with args.selected_csv.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
    writer.writeheader()
    writer.writerows(selected)
args.dataset.mkdir(parents=True, exist_ok=False)
rgb = args.dataset / "rgb"
rgb.mkdir()
lines = ["# timestamp filename", "# E5e same four inputs for VGGT and MASt3R-SLAM"]
for row in selected:
    (rgb / row["filename"]).symlink_to(Path(row["rgb_path"]).resolve())
    lines.append(f"{row['rgb_timestamp']} rgb/{row['filename']}")
(args.dataset / "rgb.txt").write_text("\n".join(lines) + "\n")
result = {
    "schema": "pointmap-comparison.e5e-input.v1",
    "source_mapping_csv_sha256": hashlib.sha256(args.mapping_csv.read_bytes()).hexdigest(),
    "source_mapping_frame_count": len(rows), "input_frame_count": 4,
    "selection_rule": "explicit user-specified mapping filenames",
    "selected_filenames": filenames,
    "selected_mapping_row_indices": row_indices,
    "selected_candidate_grid_indices": [int(row["candidate_grid_index"]) for row in selected],
    "source_frame_indices": [int(row["frame_index"]) for row in selected],
    "timestamps": [row["rgb_timestamp"] for row in selected],
    "interval_seconds": intervals,
    "rgb_paths": [row["rgb_path"] for row in selected],
    "rgb_sha256": [row["sha256"] for row in selected],
}
args.summary.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
