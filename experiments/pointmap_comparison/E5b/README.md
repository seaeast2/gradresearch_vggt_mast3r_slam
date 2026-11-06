# E5b — matched-keyframe VGGT / MASt3R-SLAM pointclouds

E5b replays the pinned E5 240-image mapping sequence with the same MASt3R-SLAM
configuration and exports every fused keyframe's canonical pointmap and pose.
The run verifies that the selected keyframe IDs and timestamps match E5. VGGT
then jointly infers those exact RGB frames in batches of at most four (the
validated memory-safe budget). Its predicted world points are converted back
to each VGGT camera frame and re-expressed using the corresponding MASt3R-SLAM
Sim(3) pose. The resulting merged V-P and M-F PLYs therefore use the same
keyframe images and MASt3R-SLAM world frame for qualitative visualization.

This pose-conditioned overlay compares pointmap geometry at matched keyframes;
it is not an independent comparison of the two systems' camera trajectories.
HKairport01 has no geometry GT, so the result is qualitative.

Run on the host with CUDA visible:

```bash
bash experiments/pointmap_comparison/E5b/scripts/run_e5b.sh \
  hkairport01-E5b-20261106-001
```

The run refuses to overwrite an existing stage and stores inputs, keyframe
exports, batch predictions, aligned clouds, logs, and provenance under
`results/pointmap_comparison/<run_id>/stages/E5b/`.
