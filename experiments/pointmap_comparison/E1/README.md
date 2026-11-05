# E1: Standard pointmap export

E1 exports the same four RGB inputs through pinned VGGT and MASt3R-SLAM pipelines. Run it on the host because the sandbox does not expose CUDA. Each run is isolated at `results/pointmap_comparison/<run_id>/stages/E1/`; the runner refuses to overwrite an existing run.

```bash
bash experiments/pointmap_comparison/E1/scripts/run_e1.sh
```

The VGGT NPZ retains V-P, V-D, raw depth/confidence, camera parameters, image tensors, frame identity, and input-to-source pixel mapping. Separate V-P and V-D PLYs are written for visualization. MASt3R-SLAM's run-local final hook exports fused keyframes after backend work drains; a schema adapter writes M-F NPZ and PLY with selected and missing input IDs reported. It does not claim pointmaps for frames that were not selected as keyframes.

The included validator checks both exports, point/RGB/confidence grid alignment, source pixel transform round trips, and local-to-world Sim(3) round trips. `N=4` is the E0b-safe VGGT input limit for the recorded RTX 5060 configuration.
