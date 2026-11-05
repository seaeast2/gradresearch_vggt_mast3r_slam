# E0: TUM RGB-D environment and baseline demo

This folder contains the small, version-controlled code and configuration needed to recreate E0. Downloaded datasets, checkpoints, external source checkouts, and generated run outputs are stored under ignored `data/`, `checkpoints/`, `thirdparty/`, and `results/` directories. The directory layout and placeholder-file pattern follow the companion [`gradresearch_mast3r_slam` repository](https://github.com/seaeast2/gradresearch_mast3r_slam/blob/main/.gitignore).

## Download inputs

From the repository root:

```bash
bash scripts/pointmap_comparison/download_e0_resources.sh
mkdir -p data/pointmap_comparison/tum
tar -xzf data/pointmap_comparison/rgbd_dataset_freiburg1_desk.tgz \
  -C data/pointmap_comparison/tum
```

The download script verifies SHA-256 before accepting each file. It downloads the TUM `freiburg1_desk` archive, VGGT-1B checkpoint, official NAVER MASt3R model and config, and MASt3R training-free retrieval assets from the recorded vslamlab Hugging Face mirror. Model files remain local and are not committed.

## Pin external source code

```bash
bash scripts/pointmap_comparison/checkout_e0_sources.sh
```

This checks out VGGT commit `a288dd0f14786c93483e45524328726ab7b1b4ce` and MASt3R-SLAM commit `e6f4e3d474fad0e11f561482012be864ba8c3f17` with recursive submodules, then applies the tracked E0 patch. The source clones are disposable local dependencies; our scripts, config, and compatibility patch remain tracked in this repository.

## Prepare the shared input frames

Choose a fresh run ID for each execution so outputs cannot overwrite a previous run:

```bash
export RUN_ID="tum-freiburg1_desk-$(date +%Y%m%d-%H%M%S)"
export STAGE_DIR="results/pointmap_comparison/$RUN_ID/stages/E0"
python experiments/pointmap_comparison/E0/scripts/prepare_e0_data.py \
  --dataset-dir data/pointmap_comparison/tum/rgbd_dataset_freiburg1_desk \
  --output-dir "$STAGE_DIR/artifacts/data"
```

The synchronizer independently matches depth and ground-truth poses to RGB timestamps using a nearest-neighbor rule and 20 ms limit. It writes a four-frame shared list (`frames_e0.csv`) and an RGB sequence for the MASt3R-SLAM demo.

## Install and run

Run GPU checks and inference on the host, outside any sandbox that hides CUDA. Use separate Python 3.11 environments for the two projects. Install a CUDA-enabled PyTorch build compatible with the host driver/toolkit; E0 used PyTorch `2.14.1+cu132` and `torchvision 0.29.1+cu132`.

For VGGT, install its pinned checkout and dependencies into its environment:

```bash
python -m pip install -r thirdparty/pointmap_comparison/vggt/requirements.txt
python -m pip install -e thirdparty/pointmap_comparison/vggt
```

For MASt3R-SLAM, install the pinned checkout following its upstream installation steps (`thirdparty/mast3r` and `thirdparty/in3d`, then the root package with `--no-build-isolation`). Set `MAST3R_PYTHON` to that environment's interpreter. E0's patch includes RTX 5060 / compute capability 12.0 build compatibility and a four-frame keyframe buffer; adapt the CUDA architecture setting for a different GPU.

Run VGGT on the shared frame list (using the VGGT environment's Python):

```bash
python experiments/pointmap_comparison/E0/scripts/run_vggt_e0.py \
  --frame-list "$STAGE_DIR/artifacts/data/frames_e0.csv" \
  --output "$STAGE_DIR/artifacts/demo_outputs/vggt_e0_predictions.npz" \
  --checkpoint-dir checkpoints/pointmap_comparison/vggt
```

Run MASt3R-SLAM on the same four RGB frames:

```bash
MAST3R_PYTHON=/path/to/mast3r-slam-venv/bin/python \
RUN_ID="$RUN_ID" bash experiments/pointmap_comparison/E0/scripts/run_mast3r_e0.sh
```

MASt3R-SLAM selects keyframes automatically; a successful E0 run may therefore produce fewer point clouds than input frames. Preserve the stage manifest, config copy, logs, and artifact checksums alongside each run's outputs.

## Provenance and limitations

The E0 experiment run, hashes, outputs, and observed keyframe coverage are documented under the local ignored run directory and in `docs/POINTMAP_COMPARISON_EXPERIMENT_PLAN.md`. The retrieval checkpoint is from a third-party mirror; its source URL and SHA-256 are explicit in `download_e0_resources.sh`. Check and comply with the respective upstream code and model licenses before use.
