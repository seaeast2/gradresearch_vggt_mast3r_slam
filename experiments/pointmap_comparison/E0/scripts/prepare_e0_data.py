#!/usr/bin/env python3
"""Create E0's shared four-frame TUM input list and MASt3R RGB sequence."""
import argparse
import bisect
import csv
import json
from pathlib import Path


def read_pairs(path):
    pairs = []
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stamp, filename, *_ = line.split()
            pairs.append((float(stamp), filename))
    return pairs


def nearest(entries, stamps, target):
    index = bisect.bisect_left(stamps, target)
    candidates = entries[max(0, index - 1):min(len(entries), index + 1)]
    return min(candidates, key=lambda item: abs(item[0] - target))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerance-ms", type=float, default=20.0)
    parser.add_argument("--preview-frames", type=int, default=4)
    args = parser.parse_args()
    dataset = args.dataset_dir.resolve()
    out = args.output_dir.resolve()
    tolerance = args.tolerance_ms / 1000.0

    rgb = read_pairs(dataset / "rgb.txt")
    depth = read_pairs(dataset / "depth.txt")
    poses = []
    with (dataset / "groundtruth.txt").open() as stream:
        for line in stream:
            line = line.strip()
            if line and not line.startswith("#"):
                fields = line.split()
                poses.append((float(fields[0]), " ".join(fields[1:])))
    if not rgb or not depth or not poses:
        raise RuntimeError("TUM sequence is incomplete; expected rgb.txt, depth.txt, and groundtruth.txt")

    depth_stamps = [item[0] for item in depth]
    pose_stamps = [item[0] for item in poses]
    associated = []
    for frame_index, (rgb_stamp, rgb_file) in enumerate(rgb):
        depth_stamp, depth_file = nearest(depth, depth_stamps, rgb_stamp)
        pose_stamp, pose = nearest(poses, pose_stamps, rgb_stamp)
        depth_delta, pose_delta = abs(depth_stamp-rgb_stamp), abs(pose_stamp-rgb_stamp)
        associated.append({
            "frame_index": frame_index,
            "rgb_timestamp": f"{rgb_stamp:.6f}",
            "rgb_path": str((dataset / rgb_file).resolve()),
            "depth_timestamp": f"{depth_stamp:.6f}",
            "depth_path": str((dataset / depth_file).resolve()),
            "depth_abs_delta_s": f"{depth_delta:.6f}",
            "groundtruth_timestamp": f"{pose_stamp:.6f}",
            "groundtruth_pose_xyzw": pose,
            "groundtruth_abs_delta_s": f"{pose_delta:.6f}",
            "within_tolerance": depth_delta <= tolerance and pose_delta <= tolerance,
        })
    matched = [row for row in associated if row["within_tolerance"]]
    preview = matched[:args.preview_frames]
    if len(preview) != args.preview_frames:
        raise RuntimeError(f"Expected {args.preview_frames} matched frames; found {len(preview)}")

    out.mkdir(parents=True, exist_ok=True)
    with (out / "frames_associated.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(associated[0]))
        writer.writeheader()
        writer.writerows(associated)
    with (out / "frames_e0.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(preview[0]))
        writer.writeheader()
        writer.writerows(preview)

    preview_dir = out / "tum" / "rgbd_dataset_freiburg1_desk_E0"
    (preview_dir / "rgb").mkdir(parents=True, exist_ok=True)
    lines = ["# E0 shared RGB preview subset", "# timestamp filename"]
    for row in preview:
        source = Path(row["rgb_path"])
        target = preview_dir / "rgb" / source.name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.symlink_to(source)
        lines.append(f"{row['rgb_timestamp']} rgb/{source.name}")
    (preview_dir / "rgb.txt").write_text("\n".join(lines) + "\n")
    summary = {
        "dataset": "TUM RGB-D freiburg1_desk",
        "sync_rule": "nearest timestamp, independently match depth and ground truth to RGB",
        "max_abs_timestamp_difference_s": tolerance,
        "counts": {"rgb": len(rgb), "depth": len(depth), "groundtruth_pose": len(poses), "rgb_with_depth_and_pose_within_tolerance": len(matched)},
        "preview_frame_indices": [row["frame_index"] for row in preview],
        "preview_rgb_timestamps": [row["rgb_timestamp"] for row in preview],
        "preview_dataset_dir": str(preview_dir),
    }
    (out / "synchronization_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
