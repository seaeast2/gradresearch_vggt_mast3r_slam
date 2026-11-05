#!/usr/bin/env python3
"""Validate and summarize the fused PLY and trajectory from an E5 run."""
import argparse
import datetime
import hashlib
import json
import pathlib

import numpy as np
from plyfile import PlyData


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, type=pathlib.Path)
    args = parser.parse_args()
    stage = args.stage.resolve()
    manifest = json.loads((stage / "manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise SystemExit(f"E5 stage is not completed: {manifest.get('status')}")
    ply_files = list((stage / "artifacts" / "outputs").glob("*/*.ply"))
    trajectory_files = list((stage / "artifacts" / "outputs").glob("*/*.txt"))
    if len(ply_files) != 1 or len(trajectory_files) != 1:
        raise SystemExit(f"expected one PLY and one trajectory; got {len(ply_files)}, {len(trajectory_files)}")
    ply = ply_files[0]
    vertices = PlyData.read(ply)["vertex"].data
    xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")])
    finite_count = int(np.isfinite(xyz).all(axis=1).sum())
    if finite_count != len(vertices) or finite_count == 0:
        raise SystemExit(f"invalid PLY coordinates: {finite_count}/{len(vertices)} finite")
    trajectory = trajectory_files[0]
    rows = [line.split() for line in trajectory.read_text().splitlines() if line.strip()]
    if not rows:
        raise SystemExit("trajectory has no keyframes")
    extraction = json.loads((stage / "artifacts/data/HKairport01/extraction_summary.json").read_text())
    summary = {
        "run_id": manifest["run_id"],
        "stage_code": "E5",
        "source_mcap_sha256": extraction["source_mcap_sha256"],
        "mapping_frames": extraction["mapping_frames"],
        "heldout_frames": extraction["heldout_frames"],
        "first_input_timestamp": extraction["first_mapping_timestamp"],
        "last_input_timestamp": extraction["last_mapping_timestamp"],
        "ply_path": str(ply.relative_to(stage)),
        "ply_sha256": digest(ply),
        "ply_bytes": ply.stat().st_size,
        "vertex_count": len(vertices),
        "finite_xyz_count": finite_count,
        "xyz_min": xyz.min(axis=0).tolist(),
        "xyz_max": xyz.max(axis=0).tolist(),
        "trajectory_path": str(trajectory.relative_to(stage)),
        "trajectory_sha256": digest(trajectory),
        "keyframe_count": len(rows),
        "first_keyframe_timestamp": rows[0][0],
        "last_keyframe_timestamp": rows[-1][0],
        "seconds_from_last_keyframe_to_last_input": float(extraction["last_mapping_timestamp"]) - float(rows[-1][0]),
        "wall_time_seconds": manifest["wall_time_seconds"],
        "peak_gpu_memory_mib_sampled": manifest["peak_gpu_memory_mib_sampled"],
        "execution_context": manifest["execution_context"],
    }
    path = stage / "artifacts" / "e5_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    outputs = manifest.setdefault("outputs", [])
    outputs[:] = [item for item in outputs if item.get("path") != "artifacts/e5_summary.json"]
    outputs.append({
        "path": "artifacts/e5_summary.json",
        "size_bytes": path.stat().st_size,
        "sha256": digest(path),
    })
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    run_manifest_path = stage.parent.parent / "run_manifest.json"
    if run_manifest_path.exists():
        run_manifest = json.loads(run_manifest_path.read_text())
        run_manifest["status"] = "completed"
        run_manifest["finished_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        run_manifest["primary_result"] = "stages/E5/artifacts/e5_summary.json"
        run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
