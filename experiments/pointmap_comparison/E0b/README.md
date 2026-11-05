# E0b: E1 implementation preflight

E0b closes the implementation decisions in section 13 of the comparison plan. It reuses the completed E0 TUM sequence, pinned source snapshots, and checkpoints without modifying them. CPU-side source/geometry/depth checks are reproducible with `analyze_e0b.py`; CUDA memory measurements use `probe_vram_e0b.py` on the host because the sandbox does not expose the GPU.

```bash
bash experiments/pointmap_comparison/E0b/scripts/run_e0b.sh
```

The runner creates an isolated `results/pointmap_comparison/<run_id>/stages/E0b/` tree and copies inputs/configuration plus logs, machine-readable measurements, and a manifest there. It never overwrites the E0/E0a outputs. Run the CUDA probe outside a GPU-hiding sandbox. Any OOM is a recorded measurement, not a failed experiment.
