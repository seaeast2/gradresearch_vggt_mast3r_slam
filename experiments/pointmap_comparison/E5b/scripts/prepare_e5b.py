#!/usr/bin/env python3
"""Create E5b's full MASt3R dataset and the baseline E5 keyframe list."""
import argparse
import csv
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--mapping-csv", type=Path, required=True)
parser.add_argument("--trajectory", type=Path, required=True)
parser.add_argument("--dataset", type=Path, required=True)
parser.add_argument("--all-frames-csv", type=Path, required=True)
parser.add_argument("--baseline-keys-csv", type=Path, required=True)
parser.add_argument("--summary", type=Path, required=True)
args = parser.parse_args()

with args.mapping_csv.open(newline="") as stream:
    rows = list(csv.DictReader(stream))
if len(rows) != 240:
    raise SystemExit(f"Expected 240 E5 mapping frames, got {len(rows)}")
trajectory = []
for line in args.trajectory.read_text().splitlines():
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    trajectory.append(line.split()[0])
if not trajectory:
    raise SystemExit("E5 trajectory has no keyframes")

def match_stamp(stamp):
    distances = [abs(float(row["rgb_timestamp"]) - float(stamp)) for row in rows]
    index = min(range(len(rows)), key=distances.__getitem__)
    if distances[index] > 1e-6:
        raise ValueError(f"E5 keyframe timestamp {stamp} has no exact mapping input")
    return rows[index]

baseline_keys = [match_stamp(stamp) for stamp in trajectory]
if len({int(row["frame_index"]) for row in baseline_keys}) != len(baseline_keys):
    raise SystemExit("E5 trajectory maps multiple keyframes to one input frame")
for row in rows:
    if not Path(row["rgb_path"]).is_file():
        raise FileNotFoundError(row["rgb_path"])

args.dataset.mkdir(parents=True, exist_ok=False)
rgb_dir = args.dataset / "rgb"
rgb_dir.mkdir()
lines = ["# timestamp filename", "# E5b replay of the pinned E5 mapping sequence"]
for row in rows:
    target = rgb_dir / row["filename"]
    target.symlink_to(Path(row["rgb_path"]).resolve())
    lines.append(f"{row['rgb_timestamp']} rgb/{row['filename']}")
(args.dataset / "rgb.txt").write_text("\n".join(lines) + "\n")

args.all_frames_csv.parent.mkdir(parents=True, exist_ok=True)
with args.all_frames_csv.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
with args.baseline_keys_csv.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(baseline_keys)
summary = {
    "schema": "pointmap-comparison.e5b-input.v1",
    "mapping_frame_count": len(rows),
    "baseline_e5_keyframe_count": len(baseline_keys),
    "baseline_frame_indices": [int(row["frame_index"]) for row in baseline_keys],
    "baseline_timestamps": [row["rgb_timestamp"] for row in baseline_keys],
    "mapping_csv_sha256": __import__("hashlib").sha256(args.mapping_csv.read_bytes()).hexdigest(),
    "baseline_trajectory": str(args.trajectory.resolve()),
}
args.summary.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
