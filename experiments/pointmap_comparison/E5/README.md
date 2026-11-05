# E5 — HKairport01 continuous MASt3R-SLAM

Run a timestamped 60-second HKairport01 mapping window from the public MARS-LVIG MCAP used to reconstruct the source repository's 240-frame mapping split. The source repository records 240 mapping images plus 60 held-out candidates from a 5 Hz grid; E5 preserves that 4:1 split and feeds all 240 mapping images in timestamp order to MASt3R-SLAM. This is a qualitative map export, not a trajectory-accuracy evaluation.

The MCAP is a large dataset resource and remains under the ignored `results/pointmap_comparison/<run_id>/stages/E5/` tree. `prepare_e5_data.sh` downloads a pinned Hugging Face revision and checks the published LFS SHA-256 before extracting. The extraction script uses the pinned MASt3R environment's `rosbags` and OpenCV packages. It reads the `/left_camera/image/compressed` ROS 2 topic, validates the 300 candidate-grid frames, excludes the preregistered held-out grid indices, and hashes each retained JPEG.

Run on the host with CUDA visible:

```bash
bash experiments/pointmap_comparison/E5/scripts/prepare_e5_data.sh <run_id>
bash experiments/pointmap_comparison/E5/scripts/run_e5.sh \
  results/pointmap_comparison/<run_id>/stages/E5
```

This experiment uses the CC BY-NC 4.0 MCAP distribution of MARS-LVIG. The original repository's ignored frame archive is not present in this checkout, so the public MCAP is the data source; exact equality with the old local PNG hashes is not claimed.

The runner uses a run-local overlay with a 32-keyframe GPU buffer, matching the 8 GB configuration previously used by the source repository. It refuses to overwrite a previous run, samples host GPU memory, keeps the stock fused-map PLY/trajectory, and validates the PLY header and vertex count. If more than 32 keyframes are selected, upstream raises an explicit capacity error and the failed run remains intact.
