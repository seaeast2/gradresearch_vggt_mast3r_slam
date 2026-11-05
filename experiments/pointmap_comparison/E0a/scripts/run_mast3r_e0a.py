#!/usr/bin/env python3
"""Run pinned MASt3R-SLAM with a run-local export hook after backend shutdown."""
import pathlib
import sys


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_mast3r_e0a.py /path/to/pinned/main.py /path/to/run-local/main_e0a.py")

    upstream_main = pathlib.Path(sys.argv[1]).resolve()
    output_main = pathlib.Path(sys.argv[2]).resolve()
    upstream_code = upstream_main.read_text()

    import_marker = "import datetime\n"
    if upstream_code.count(import_marker) != 1:
        raise RuntimeError("Pinned upstream main.py no longer matches the expected E0a import hook")
    upstream_code = upstream_code.replace(
        import_marker,
        import_marker + "import csv\nimport os\nimport numpy as np\n",
        1,
    )

    backend_join = "    backend.join()\n"
    if upstream_code.count(backend_join) != 1:
        raise RuntimeError("Pinned upstream main.py no longer has one backend.join() export point")

    terminate_loop = "        if i == len(dataset):\n            states.set_mode(Mode.TERMINATED)\n"
    if upstream_code.count(terminate_loop) != 1:
        raise RuntimeError("Pinned upstream main.py no longer matches the expected final-frame drain point")
    drain_queue = '''        if i == len(dataset):
            if config["single_thread"]:
                while True:
                    with states.lock:
                        pending_global_optimization = len(states.global_optimizer_tasks)
                    if pending_global_optimization == 0:
                        break
                    time.sleep(0.01)
            states.set_mode(Mode.TERMINATED)
'''
    upstream_code = upstream_code.replace(terminate_loop, drain_queue, 1)

    export_hook = r'''    # E0a: export fused keyframes after backend shutdown, without editing the checkout.
    e0a_out = pathlib.Path(os.environ["E0A_EXPORT_DIR"])
    e0a_out.mkdir(parents=True, exist_ok=True)
    with open(os.environ["E0A_FRAME_MAP_CSV"], newline="") as e0a_stream:
        e0a_rows = list(csv.DictReader(e0a_stream))
    e0a_local, e0a_world, e0a_conf, e0a_rgb = [], [], [], []
    e0a_matrices, e0a_ids, e0a_source_ids, e0a_stamps = [], [], [], []
    e0a_shapes, e0a_source_shapes, e0a_fusion_count, e0a_update_count = [], [], [], []
    for e0a_slot in range(len(keyframes)):
        e0a_kf = keyframes[e0a_slot]
        e0a_id = int(e0a_kf.frame_id)
        if not 0 <= e0a_id < len(e0a_rows):
            raise RuntimeError(f"keyframe frame_id {e0a_id} is outside the shared input list")
        e0a_row = e0a_rows[e0a_id]
        e0a_stamp = str(dataset.timestamps[e0a_id])
        if abs(float(e0a_stamp) - float(e0a_row["rgb_timestamp"])) > 1e-6:
            raise RuntimeError(f"keyframe timestamp mismatch at shared frame {e0a_id}")
        e0a_h, e0a_w = [int(v) for v in e0a_kf.img_shape.flatten().tolist()]
        e0a_x = e0a_kf.X_canon.reshape(e0a_h, e0a_w, 3).detach().cpu().numpy().astype(np.float32)
        e0a_c = e0a_kf.get_average_conf().reshape(e0a_h, e0a_w).detach().cpu().numpy().astype(np.float32)
        e0a_p = e0a_kf.T_WC.act(e0a_kf.X_canon).reshape(e0a_h, e0a_w, 3).detach().cpu().numpy().astype(np.float32)
        e0a_color = e0a_kf.uimg.reshape(e0a_h, e0a_w, 3).detach().cpu().numpy().astype(np.float32)
        e0a_matrix = e0a_kf.T_WC.matrix().detach().cpu().numpy().reshape(4, 4).astype(np.float32)
        e0a_source_bgr = cv2.imread(e0a_row["rgb_path"], cv2.IMREAD_COLOR)
        if e0a_source_bgr is None:
            raise RuntimeError(f"cannot read source RGB image: {e0a_row['rgb_path']}")
        e0a_source_h, e0a_source_w = e0a_source_bgr.shape[:2]
        e0a_local.append(e0a_x)
        e0a_world.append(e0a_p)
        e0a_conf.append(e0a_c)
        e0a_rgb.append(e0a_color)
        e0a_matrices.append(e0a_matrix)
        e0a_ids.append(e0a_id)
        e0a_source_ids.append(int(e0a_row["frame_index"]))
        e0a_stamps.append(e0a_stamp)
        e0a_shapes.append([e0a_h, e0a_w])
        e0a_source_shapes.append([e0a_source_h, e0a_source_w])
        e0a_fusion_count.append(int(e0a_kf.N))
        e0a_update_count.append(int(e0a_kf.N_updates))
    if not e0a_ids:
        raise RuntimeError("MASt3R-SLAM finished without any fused keyframes")
    np.savez_compressed(
        e0a_out / "mast3r_e0a_fused_keyframes.npz",
        dataset_frame_id=np.asarray(e0a_ids, dtype=np.int32),
        source_frame_index=np.asarray(e0a_source_ids, dtype=np.int32),
        rgb_timestamp=np.asarray(e0a_stamps),
        image_shape_hw=np.asarray(e0a_shapes, dtype=np.int32),
        source_image_shape_hw=np.asarray(e0a_source_shapes, dtype=np.int32),
        X_canon_local=np.stack(e0a_local),
        points_world=np.stack(e0a_world),
        confidence=np.stack(e0a_conf),
        valid_mask=np.isfinite(np.stack(e0a_local)).all(axis=-1) & np.isfinite(np.stack(e0a_conf)) & (np.stack(e0a_conf) > 0),
        rgb=np.stack(e0a_rgb),
        T_WC_matrix=np.stack(e0a_matrices),
        fusion_count=np.asarray(e0a_fusion_count, dtype=np.int32),
        update_count=np.asarray(e0a_update_count, dtype=np.int32),
        coordinate_note=np.asarray("X_canon_local is per-keyframe canonical/camera pointmap; points_world = T_WC.act(X_canon_local)"),
        color_note=np.asarray("RGB is the resized MASt3R input image in [0,1], HxWx3"),
    )
    print(f"E0a exported {len(e0a_ids)} fused keyframes to {e0a_out}")
'''
    upstream_code = upstream_code.replace(backend_join, backend_join + export_hook, 1)
    output_main.parent.mkdir(parents=True, exist_ok=True)
    output_main.write_text(upstream_code)
    print(f"Created run-local patched launcher: {output_main}")


if __name__ == "__main__":
    main()
