#!/usr/bin/env python3
"""Select four deterministic, evenly spaced images from the E5 mapping split."""
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--mapping-csv", type=Path, required=True)
parser.add_argument("--output-csv", type=Path, required=True)
parser.add_argument("--summary", type=Path, required=True)
args = parser.parse_args()

with args.mapping_csv.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 240:
    raise SystemExit(f"Expected the pinned E5 240-frame mapping split, got {len(rows)}")
indices = [round(i * (len(rows) - 1) / 3) for i in range(4)]
selected = [rows[i] for i in indices]
if len(set(indices)) != 4:
    raise SystemExit(f"Sampling did not produce four unique indices: {indices}")
for row in selected:
    if not Path(row["rgb_path"]).is_file():
        raise FileNotFoundError(row["rgb_path"])

args.output_csv.parent.mkdir(parents=True, exist_ok=True)
with args.output_csv.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
    writer.writeheader()
    writer.writerows(selected)
summary = {
    "schema": "pointmap-comparison.e5a-input.v1",
    "source_mapping_csv": str(args.mapping_csv.resolve()),
    "source_frame_count": len(rows),
    "input_budget": len(selected),
    "selection": "four evenly spaced mapping frames including first and last",
    "selected_mapping_row_indices": indices,
    "selected_frame_indices": [int(row["frame_index"]) for row in selected],
    "selected_timestamps": [row["rgb_timestamp"] for row in selected],
    "selected_rgb_paths": [row["rgb_path"] for row in selected],
}
args.summary.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
