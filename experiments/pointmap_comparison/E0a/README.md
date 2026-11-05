# E0a: MASt3R-SLAM fused pointmap extraction preflight

E0a checks whether MASt3R-SLAM's final fused keyframe state can be exported with reliable input-frame and pixel identity before the standard E1 exporter is implemented. It uses a run-local generated `main_e0a.py`, drains single-thread backend work before termination, and exports only after the backend exits. The pinned upstream checkout stays unchanged.

## Inputs and execution

Run GPU inference on the host, outside a sandbox that hides CUDA. E0a reuses a completed E0 stage by default. To use a different E0 run, set `E0_STAGE_DIR` to its `stages/E0` directory. On a clean checkout, download and extract the TUM sequence and checkpoints, pin the upstream sources with `scripts/pointmap_comparison/checkout_e0_sources.sh`, install the MASt3R-SLAM environment following the E0 instructions, then run:

```bash
MAST3R_PYTHON=/path/to/mast3r-slam-venv/bin/python \
RUN_ID="tum-freiburg1_desk-E0a-$(date +%Y%m%d-%H%M%S)" \
bash experiments/pointmap_comparison/E0a/scripts/run_e0a.sh
```

If `E0_STAGE_DIR` has no `frames_e0.csv`, the script prepares the shared four-frame mapping from `data/pointmap_comparison/tum/rgbd_dataset_freiburg1_desk`. Set `TUM_DATASET_DIR` if the extracted sequence is elsewhere. `MAST3R_SOURCE_DIR` can select a pinned source checkout; by default it uses E0's checkout or `thirdparty/pointmap_comparison/mast3r_slam`.

Every run is isolated under `results/pointmap_comparison/<run_id>/stages/E0a/`, including the copied frame map, source overlay, NPZ, validation JSON, PLY/trajectory, logs, and manifest. A new `RUN_ID` is required after a failed attempt. The validator reports selected keyframes and missing inputs rather than assuming every input frame becomes a keyframe.

## E0a result

The completed host run is `tum-freiburg1_desk-20261105-005`. With four RGB inputs, MASt3R-SLAM selected one keyframe (25% coverage). Its center pixel maps to source TUM frame 17 / timestamp `1305031453.359684`, with local/world XYZ, confidence, and RGB available in the export. The measured pose for that keyframe was identity-initialized, so E0a also ran a separate synthetic non-identity Sim(3) round trip. This establishes the extraction path; E1 must still validate a real non-identity trajectory and implement the reusable export interface.
