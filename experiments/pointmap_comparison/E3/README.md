# E3: Pilot matrix and repeatability

E3 evaluates contiguous P1 prefixes (4/8/16/32) on TUM `freiburg1_desk` and `freiburg1_room`. VGGT inputs above the E0b safe maximum of four are attempted as explicitly labeled OOM probes; no tiling is substituted for the preregistered joint-input condition. MASt3R-SLAM receives the same selected RGB frame list in timestamp order. The C1 run is a MASt3R-only calibration auxiliary; VGGT has no matching external-calibration input path.

Each condition writes under a distinct model/condition/repeat directory in one unique E3 run. Keep failed attempts and logs. Run model inference on the host; CPU evaluation can use the pinned MASt3R Python environment:

```bash
stage=results/pointmap_comparison/<run_id>/stages/E3
python3 experiments/pointmap_comparison/E3/scripts/prepare_e3.py \
  --dataset desk \
  --dataset-dir results/pointmap_comparison/tum-freiburg1_desk-20261105-001/stages/E0/artifacts/data/rgbd_dataset_freiburg1_desk \
  --output-dir "$stage/artifacts/data/desk"
curl --fail --location --output "$stage/artifacts/data/rgbd_dataset_freiburg1_room.tgz" \
  https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz
mkdir -p "$stage/artifacts/data/tum_room"
tar -xzf "$stage/artifacts/data/rgbd_dataset_freiburg1_room.tgz" -C "$stage/artifacts/data/tum_room"
python3 experiments/pointmap_comparison/E3/scripts/prepare_e3.py \
  --dataset room \
  --dataset-dir "$stage/artifacts/data/tum_room/rgbd_dataset_freiburg1_room" \
  --output-dir "$stage/artifacts/data/room"

for scene in desk room; do
  for n in 4 8 16 32; do
    bash experiments/pointmap_comparison/E3/scripts/run_vggt_attempt.sh "$stage" "$scene" "$n" 1 P1
    bash experiments/pointmap_comparison/E3/scripts/run_mast3r_attempt.sh "$stage" "$scene" "$n" C0 1 P1
  done
done

# Pair metrics are available only where both systems exported outputs.
bash experiments/pointmap_comparison/E3/scripts/evaluate_pair.sh "$stage" desk 4 1 1
bash experiments/pointmap_comparison/E3/scripts/evaluate_pair.sh "$stage" room 4 1 1

# P2 uses all valid MASt3R frames and a fixed four-view VGGT sample.
bash experiments/pointmap_comparison/E3/scripts/run_mast3r_attempt.sh "$stage" room 1362 C0 1 P2
bash experiments/pointmap_comparison/E3/scripts/run_vggt_attempt.sh "$stage" room 4 1 P2

python3 experiments/pointmap_comparison/E3/scripts/summarize_e3.py \
  --stage "$stage" --output-dir "$stage/artifacts/summary"
python3 experiments/pointmap_comparison/E3/scripts/finalize_e3.py \
  --stage "$stage" --run-id "<run_id>" --status completed_with_failures
```

N>4 VGGT commands attempt the preregistered joint input and preserve CUDA OOM as a condition result; they do not silently switch to chunks. Add repeats under new `repeatNN` directories. `evaluate_mast3r_retention.py` computes confidence retention on each exported M-F keyframe. The parent run is immutable once complete. Input archives, model checkpoints, and dense results remain ignored by Git; commit the scripts and concise tables/summary only.
