# E5e — user-selected four-frame comparison

Select `000050.jpg`, `000070.jpg`, `000090.jpg`, and `000110.jpg` from E5's
mapping sequence. Feed this same timestamped RGB list to VGGT-1B jointly and to
MASt3R-SLAM in chronological order. Preserve separate VGGT V-P and MASt3R-SLAM
M-F PLY files so they can be inspected independently. The original run uses
MASt3R-SLAM's automatic keyframe selection. A follow-up run can force each
successfully tracked selected input to become a keyframe, then transform each
keyframe pointmap by its estimated `T_WC` into the SLAM world frame and merge
those four clouds into one PLY. Frames that fail tracking are not force-added.

Run on the host with CUDA visible:

```bash
bash experiments/pointmap_comparison/E5e/scripts/run_e5e.sh \
  hkairport01-E5e-20261106-001
```

The canonical E5e result keeps VGGT's four-frame output and uses MASt3R-SLAM's
forced-keyframe output, with each successfully tracked input represented as a
keyframe. The original automatic-keyframe MASt3R result and imported run
provenance remain archived under the canonical run folder.

To create a fresh run of this forced-keyframe variant, choose a new run ID and
reuse the desired VGGT source run:

```bash
bash experiments/pointmap_comparison/E5e/scripts/run_e5e_force_keyframes.sh \
  hkairport01-E5e-20261106-004 hkairport01-E5e-20261106-001
```

Each model keeps its own predicted coordinates. Do not overlay the two PLYs
without an explicit coordinate-alignment step. The pointclouds are qualitative;
HKairport01 has no geometry ground truth.
