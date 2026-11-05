# E2: GT-based evaluation pipeline

E2 associates TUM RGB, depth, and mocap poses; samples model pointmaps back onto original RGB pixels; computes depth and camera-local 3D metrics; and reports cloud and outlier metrics on the observed surface. Global Sim(3) alignment refuses prefixes shorter than the E0b minimum, instead of silently fitting an underdetermined transform.

Run on the host (CPU evaluation is sufficient; no model inference occurs):

```bash
bash experiments/pointmap_comparison/E2/scripts/run_e2.sh
```

All outputs, configs, logs, hashes, and a machine-readable manifest go under one new `results/pointmap_comparison/<run_id>/stages/E2/`. The E1 pilot covers four images and one fused MASt3R keyframe, so E2 marks sequence-global scores unavailable and reports the exact shared per-pixel/local subset.
