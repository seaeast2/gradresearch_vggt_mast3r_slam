#!/usr/bin/env python3
"""Extract the source repository's 240-frame HKairport01 mapping split."""
import argparse
import csv
import hashlib
import json
import pathlib

import cv2
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore


TOPIC = "/left_camera/image/compressed"
GRID_START = 1671606500.1999009
GRID_STEP_S = 0.2
GRID_COUNT = 300
HELDOUT_MODULUS = 5
TIME_TOLERANCE_S = 0.035


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", required=True, type=pathlib.Path)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    args = parser.parse_args()
    mcap = args.mcap.resolve()
    output = args.output.resolve()
    if not mcap.is_file():
        raise SystemExit(f"MCAP not found: {mcap}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to overwrite existing extraction: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rgb_dir = output / "mapping" / "rgb"
    rgb_dir.mkdir(parents=True)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    candidates = {}
    matching_topic_messages = 0
    with AnyReader([mcap], default_typestore=typestore) as reader:
        connections = [c for c in reader.connections if c.topic == TOPIC]
        if not connections:
            raise RuntimeError(f"MCAP does not contain required topic {TOPIC}")
        for connection, bag_time_ns, raw in reader.messages(connections=connections):
            matching_topic_messages += 1
            message = reader.deserialize(raw, connection.msgtype)
            stamp = float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1e-9
            grid_index = round((stamp - GRID_START) / GRID_STEP_S)
            if not 0 <= grid_index < GRID_COUNT:
                continue
            target_stamp = GRID_START + grid_index * GRID_STEP_S
            if abs(stamp - target_stamp) > TIME_TOLERANCE_S:
                continue
            if grid_index in candidates:
                raise RuntimeError(f"duplicate camera image for 5 Hz grid index {grid_index}")
            candidates[grid_index] = {
                "message": message,
                "bag_time_ns": int(bag_time_ns),
                "timestamp": stamp,
            }

    if len(candidates) != GRID_COUNT:
        missing = sorted(set(range(GRID_COUNT)) - set(candidates))
        raise RuntimeError(
            f"expected {GRID_COUNT} timestamp-grid candidates, got {len(candidates)}; "
            f"missing indices (first 20): {missing[:20]}"
        )

    rows = []
    retained_index = 0
    for grid_index in range(GRID_COUNT):
        if grid_index % HELDOUT_MODULUS == HELDOUT_MODULUS - 1:
            continue
        entry = candidates[grid_index]
        message = entry["message"]
        payload = bytes(message.data)
        if not payload:
            raise RuntimeError(f"empty compressed image at grid index {grid_index}")
        fmt = str(message.format).lower()
        suffix = ".png" if "png" in fmt else ".jpg"
        filename = f"{retained_index:06d}{suffix}"
        image_path = rgb_dir / filename
        image_path.write_bytes(payload)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (2048, 2448):
            raise RuntimeError(
                f"unexpected decoded image at {image_path}: "
                f"{None if image is None else image.shape}"
            )
        rows.append({
            "frame_index": retained_index,
            "candidate_grid_index": grid_index,
            "rgb_timestamp": f"{entry['timestamp']:.9f}",
            "bag_timestamp_ns": entry["bag_time_ns"],
            "filename": filename,
            "rgb_path": str(image_path),
            "width": image.shape[1],
            "height": image.shape[0],
            "format": fmt,
            "sha256": sha256_file(image_path),
        })
        retained_index += 1

    if retained_index != 240:
        raise RuntimeError(f"expected 240 retained frames, got {retained_index}")
    csv_path = output / "mapping_frames.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rgb_lines = ["# timestamp filename"]
    rgb_lines.extend(f"{r['rgb_timestamp']} rgb/{r['filename']}" for r in rows)
    (output / "mapping" / "rgb.txt").write_text("\n".join(rgb_lines) + "\n")
    summary = {
        "source_mcap": str(mcap),
        "source_mcap_sha256": sha256_file(mcap),
        "topic": TOPIC,
        "topic_message_count": matching_topic_messages,
        "grid_start_timestamp": GRID_START,
        "grid_step_seconds": GRID_STEP_S,
        "candidate_grid_frames": len(candidates),
        "heldout_rule": f"candidate_grid_index % {HELDOUT_MODULUS} == 4",
        "heldout_frames": GRID_COUNT - retained_index,
        "mapping_frames": retained_index,
        "first_mapping_timestamp": rows[0]["rgb_timestamp"],
        "last_mapping_timestamp": rows[-1]["rgb_timestamp"],
        "mapping_csv_sha256": sha256_file(csv_path),
        "execution_context": "host",
    }
    (output / "extraction_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
