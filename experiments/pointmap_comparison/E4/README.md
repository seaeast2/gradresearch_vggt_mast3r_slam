# E4: Outlier cases and uncertainty indicators

E4 uses the two E3 C0 N=4 paired pointmap exports (pixel-local 3D metrics) and the four-view VGGT sample from room P2. Semantic case labels are manual visual tags; geometric outlier labels are fixed as camera-local 3D error `>0.05 m` on registered GT-valid pixels. Confidence retention always reports both outlier rate and retained coverage.

```bash
python3 experiments/pointmap_comparison/E4/scripts/analyze_e4.py \
  --e3-stage results/pointmap_comparison/tum-freiburg1_desk-E3-20261105-016/stages/E3 \
  --output-dir results/pointmap_comparison/tum-freiburg1_desk-E4-20261105-019/stages/E4/artifacts
```

All E4 files go to a new ignored run directory. Figures use same-view camera-frame points; no E4 result applies a sequence-global alignment.
