# E5c — matched four-frame, one-second comparison

Select four images from the beginning of E5's 5 Hz mapping split at 1-second
intervals (mapping frame indices 0, 5, 10, 15). Feed the same timestamped RGB
list to VGGT-1B jointly and to MASt3R-SLAM in chronological order. Preserve
separate VGGT V-P and MASt3R-SLAM M-F PLY files so they can be inspected
independently. MASt3R-SLAM keyframe selection and tracking coverage are reported
as observed; this stage does not force every input to become a keyframe.

Run on the host with CUDA visible:

```bash
bash experiments/pointmap_comparison/E5c/scripts/run_e5c.sh \
  hkairport01-E5c-20261106-001
```

Each model keeps its own predicted coordinates. Do not overlay the two PLYs
without an explicit coordinate-alignment step. The pointclouds are qualitative;
HKairport01 has no geometry ground truth.
