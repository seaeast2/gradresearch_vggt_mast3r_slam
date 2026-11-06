#!/usr/bin/env python3
"""Concatenate same-inference binary PLYs while retaining the shared VGGT frame."""
import argparse
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", nargs="+", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

body_chunks = []
total_vertices = 0
for path in args.inputs:
    data = path.read_bytes()
    marker = b"end_header\n"
    end = data.find(marker)
    if end < 0:
        raise ValueError(f"Missing PLY header terminator: {path}")
    header = data[:end + len(marker)].decode("ascii")
    if "format binary_little_endian 1.0" not in header:
        raise ValueError(f"Unsupported PLY encoding: {path}")
    match = re.search(r"^element vertex (\d+)$", header, re.MULTILINE)
    if not match:
        raise ValueError(f"Missing vertex count: {path}")
    count = int(match.group(1))
    payload = data[end + len(marker):]
    if len(payload) != count * 15:
        raise ValueError(f"Unexpected binary vertex payload size in {path}")
    total_vertices += count
    body_chunks.append(payload)

args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("wb") as stream:
    stream.write(("ply\nformat binary_little_endian 1.0\n"
                  f"element vertex {total_vertices}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  "end_header\n").encode("ascii"))
    for chunk in body_chunks:
        stream.write(chunk)
print(f"{args.output}: {total_vertices} vertices from {len(args.inputs)} frames")
