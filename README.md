# VGGT / MASt3R-SLAM Pointmap Comparison

This repository keeps experiment scripts, configurations, source patches, and documentation in Git. Downloaded datasets and checkpoints, external source checkouts, and generated run outputs stay local under ignored directories. The directory policy follows the companion [`gradresearch_mast3r_slam` `.gitignore`](https://github.com/seaeast2/gradresearch_mast3r_slam/blob/main/.gitignore), including `.gitkeep` placeholders for resource directories.

## E0 reproduction

See [E0 instructions](experiments/pointmap_comparison/E0/README.md) for host CUDA requirements, pinned source checkouts, dataset/model downloads, shared-frame preparation, and both demo commands. Downloaded files are checksum-verified by `scripts/pointmap_comparison/download_e0_resources.sh`.

Each run writes only to `results/pointmap_comparison/<run_id>/stages/E0/`; use a fresh `RUN_ID` for every execution. The E0 plan and observed result are in [the experiment plan](docs/POINTMAP_COMPARISON_EXPERIMENT_PLAN.md).

## Resource layout

| Path | Git policy | Contents |
|---|---|---|
| `experiments/`, `scripts/`, `docs/` | tracked | Reproduction code, configs, patches, and documentation |
| `data/` | ignored except `.gitkeep` | Downloaded benchmark data |
| `checkpoints/` | ignored except `.gitkeep` | Downloaded model weights |
| `thirdparty/` | ignored except `.gitkeep` | Pinned upstream source checkouts |
| `results/` | ignored except `.gitkeep` | Per-run artifacts, logs, and manifests |
