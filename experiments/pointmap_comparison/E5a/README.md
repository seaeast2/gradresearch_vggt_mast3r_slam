# E5a — HKairport01 VGGT pointmap and PLY

Run one joint VGGT-1B inference on four evenly spaced images selected from the
240-frame E5 mapping split. E0b established four views as the safe input budget
for the available 8 GB GPU. The direct VGGT Point Head outputs share one
predicted world coordinate frame within this inference. This is a qualitative
sample of the sequence, not a pointmap for all 240 images or a geometry-accuracy
result.

Run on the host with CUDA visible:

```bash
bash experiments/pointmap_comparison/E5a/scripts/run_e5a.sh \
  hkairport01-E5a-20261106-001
```

The run reuses the pinned E5 mapping-frame CSV and the E0 VGGT checkpoint and
environment. It writes the sampled input list, per-frame direct/depth PLYs, the
full prediction NPZ, a merged direct-Point-Head PLY, logs, hashes, and a
manifest under `results/pointmap_comparison/<run_id>/stages/E5a/`.
