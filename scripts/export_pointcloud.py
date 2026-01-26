import argparse, os, json, subprocess, sys
import numpy as np
from devo.config import cfg
from utils.eval_utils import run_voxel
from utils.load_utils import fpv_evs_iterator, rpg_evs_iterator, vector_evs_iterator


def build_arg_parser(datapath_required=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/eval_fpv.yaml")
    parser.add_argument("--datapath", required=datapath_required)
    parser.add_argument("--weights", default="DEVO.pth")
    parser.add_argument("--out", default="point_cloud.npy")
    parser.add_argument(
        "--export_edge_cloud",
        action="store_true",
        help="Export an edge-aligned point cloud by densifying depths onto event edges (uses scorer for VO accuracy).",
    )
    parser.add_argument(
        "--edge_cloud_out",
        type=str,
        default=None,
        help="Path for the edge point cloud .npy (defaults to <out>_edges.npy).",
    )
    parser.add_argument("--edge_topk", type=int, default=6000, help="Number of edge pixels per keyframe (quarter-res).")
    parser.add_argument("--edge_border", type=int, default=2, help="Border to suppress in quarter-res pixels.")
    parser.add_argument("--edge_knn", type=int, default=4, help="kNN for inverse-depth interpolation from patch centers.")
    parser.add_argument("--edge_max_dist", type=float, default=6.0, help="Max kNN radius (quarter-res pixels).")
    parser.add_argument("--viz", action="store_true", help="Enable live DPViewer visualization")
    parser.add_argument("--viz-flow", action="store_true", help="Enable flow computation/output")
    parser.add_argument('--save_per_frame_cloud', action="store_true", help="Save point cloud for each frame")
    parser.add_argument('--save_per_frame_cloud_path', type=str, default="results/clouds", help="Path to save per-frame point clouds")
    parser.add_argument('--export_frame_data', action="store_true", help="Export per-frame sparse point clouds, depths, and metadata to disk")
    parser.add_argument('--frame_data_out', type=str, default=None, help="Path for the per-frame .npz bundle (defaults to <out>_frames.npz)")
    parser.add_argument('--save_dpviewer_cloud', action="store_true", help="Accumulate DPViewer-style point cloud and save to disk")
    parser.add_argument('--dataset', choices=["fpv", "rpg", "vector"], default="fpv", help="Dataset type to choose iterator/geometry")
    parser.add_argument('--stride', type=int, default=1, help="Frame/event stride for the iterator")
    parser.add_argument('--side', type=str, default="left", help="Camera side (used for stereo datasets such as RPG)")
    parser.add_argument('--start_us', type=float, default=None, help="Start timestamp (microseconds) for FPV sequences")
    parser.add_argument('--stop_us', type=float, default=None, help="Stop timestamp (microseconds) for FPV sequences")
    parser.add_argument('--debug', action='store_true', help="Print detailed per-frame debug info during DEVO processing")
    parser.add_argument('--convert_ply', action='store_true', help="Convert the output .npy point cloud to .ply format using npy2ply.py")
    parser.add_argument('--cleanup', action='store_true', help="Run cleanup on the generated .ply file (implies --convert_ply)")
    parser.add_argument('--cleanup_algo', choices=['sor', 'ror'], default='sor', help="Cleanup algorithm: 'sor' or 'ror'")
    parser.add_argument('--sor_nb_neighbors', type=int, default=20, help="[SOR] Number of neighbors (default: 20)")
    parser.add_argument('--sor_std_ratio', type=float, default=2.0, help="[SOR] Standard deviation ratio (default: 2.0)")
    parser.add_argument('--ror_radius', type=float, default=0.05, help="[ROR] Radius (default: 0.05)")
    parser.add_argument('--ror_min_neighbors', type=int, default=16, help="[ROR] Min neighbors (default: 16)")

    return parser


def run_export_pointcloud(args):
    cfg.merge_from_file(args.config)

    base, ext = os.path.splitext(args.out)
    if ext == "":
        ext = ".npy"
    pointcloud_path = base + ext
    out_dir = os.path.dirname(os.path.abspath(pointcloud_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    edge_cloud_path = None
    if args.export_edge_cloud:
        edge_cloud_path = args.edge_cloud_out or (base + "_edges" + ext)
    dense_points_path = None
    if args.save_dpviewer_cloud:
        dense_points_path = base + "_dpviewer" + ext

    dataset = args.dataset.lower()
    if dataset == "fpv":
        iterator = fpv_evs_iterator(
            args.datapath,
            stride=args.stride,
            t_start_us=args.start_us,
            t_stop_us=args.stop_us,
        )
        H, W = 260, 346
    elif dataset == "rpg":
        iterator = rpg_evs_iterator(args.datapath, side=args.side, stride=args.stride)
        H, W = 180, 240
    elif dataset == "vector":
        intr_path = os.path.join(args.datapath, f"calib_undist_evs_{args.side}.txt")
        intrinsics = None
        if os.path.exists(intr_path):
            try:
                intrinsics = np.loadtxt(intr_path)
            except Exception as exc:
                print(f"Warning: failed to load intrinsics from {intr_path}: {exc}")
        iterator = vector_evs_iterator(
            args.datapath,
            side=args.side,
            stride=args.stride,
            t_start_us=args.start_us,
            t_stop_us=args.stop_us,
            H=480,
            W=640,
        )
        H, W = 480, 640
    else:
        raise ValueError(f"Unsupported dataset '{args.dataset}'")

    results = run_voxel(
        voxeldir=args.datapath,
        cfg=cfg,
        network=args.weights,
        iterator=iterator,
        H=H,
        W=W,
        viz=args.viz,
        viz_flow=args.viz_flow,
        debug=getattr(args, "debug", False),
        return_observables=True,
        return_frame_observables=args.export_frame_data,
        export_edge_cloud=args.export_edge_cloud,
        edge_cloud_out=edge_cloud_path,
        edge_topk=args.edge_topk,
        edge_border=args.edge_border,
        edge_knn=args.edge_knn,
        edge_max_dist=args.edge_max_dist,
        save_per_frame_cloud=args.save_per_frame_cloud,
        save_per_frame_cloud_path=args.save_per_frame_cloud_path,
        accumulate_map=args.save_dpviewer_cloud,
        full_map_out=dense_points_path,
    )

    if args.export_frame_data:
        poses, tstamps, flow, point_cloud, depths, frame_data = results
    else:
        poses, tstamps, flow, point_cloud, depths = results

    np.save(pointcloud_path, point_cloud)
    np.save(base + "_depths" + ext, depths)
    np.save(base + "_poses" + ext, poses)
    np.save(base + "_tstamps" + ext, tstamps)

    meta = {
        "dataset": dataset,
        "datapath": args.datapath,
        "side": getattr(args, "side", None),
        "config": args.config,
        "weights": args.weights,
        "edge_cloud_exported": bool(args.export_edge_cloud),
        "edge_cloud_path": edge_cloud_path,
        "edge_topk": int(args.edge_topk),
        "edge_border": int(args.edge_border),
        "edge_knn": int(args.edge_knn),
        "edge_max_dist": float(args.edge_max_dist),
        "start_us": args.start_us,
        "stop_us": args.stop_us,
        "H": H,
        "W": W,
        "intrinsics_file": intr_path if dataset == "vector" else None,
        "intrinsics_fx_fy_cx_cy": intrinsics.tolist() if dataset == "vector" and intrinsics is not None else None,
        "start_pose_xyz_quat_xyzw": poses[0].tolist() if poses is not None and len(poses) > 0 else None,
        "start_timestamp_us": float(tstamps[0]) if tstamps is not None and len(tstamps) > 0 else None,
        "num_poses": int(len(poses)) if poses is not None else 0,
        "point_cloud_shape": list(point_cloud.shape) if point_cloud is not None else None,
        "depths_shape": list(depths.shape) if depths is not None else None,
        "flow_saved": flow is not None,
        "export_frame_data": args.export_frame_data,
        "save_per_frame_cloud": args.save_per_frame_cloud,
    }
    with open(base + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if flow is not None:
        np.save(base + "_flow" + ext, flow, allow_pickle=True)

    if args.export_frame_data:
        frame_ids = np.array([f["frame_id"] for f in frame_data], dtype=np.int64)
        timestamps = np.array([f["timestamp"] for f in frame_data], dtype=np.float64)
        poses_per_frame = np.stack([f["pose"] for f in frame_data], axis=0) if frame_data else np.zeros((0, 7), dtype=np.float32)
        intrinsics_per_frame = np.stack([f["intrinsics"] for f in frame_data], axis=0) if frame_data else np.zeros((0, 4), dtype=np.float32)
        counts = np.array([len(f["points"]) for f in frame_data], dtype=np.int32)

        if counts.size > 0 and counts.sum() > 0:
            points_concat = np.concatenate([f["points"] for f in frame_data], axis=0)
            depths_concat = np.concatenate([f["depths"] for f in frame_data], axis=0)
        else:
            points_concat = np.zeros((0, 3), dtype=np.float32)
            depths_concat = np.zeros((0,), dtype=np.float32)

        # Store centers per frame (no concatenation), padded to max points per frame.
        max_count = int(counts.max()) if counts.size > 0 else 0
        centers_per_frame = np.full((len(frame_data), max_count, 2), np.nan, dtype=np.float32)
        for i, entry in enumerate(frame_data):
            c = entry.get("centers", None)
            if c is None:
                continue
            c = np.asarray(c, dtype=np.float32)
            n = min(len(c), max_count)
            if n > 0:
                centers_per_frame[i, :n] = c[:n]

        offsets = np.zeros_like(counts, dtype=np.int64)
        if counts.size > 0:
            offsets[1:] = np.cumsum(counts[:-1], dtype=np.int64)

        frame_out_path = args.frame_data_out or base + "_frames.npz"
        np.savez(
            frame_out_path,
            frame_ids=frame_ids,
            timestamps=timestamps,
            poses=poses_per_frame,
            intrinsics=intrinsics_per_frame,
            counts=counts,
            offsets=offsets,
            points=points_concat,
            depths=depths_concat,
            centers=centers_per_frame,
        )

    if args.cleanup or args.convert_ply:
        print("\n--- Starting Post-processing ---")
        script_dir = os.path.dirname(os.path.realpath(__file__))
        npy2ply_script = os.path.join(script_dir, "npy2ply.py")
        
        ply_paths = []
        for npy_path in [pointcloud_path, edge_cloud_path] if args.export_edge_cloud else [pointcloud_path]:
            if npy_path is None:
                continue
            cmd_convert = [sys.executable, npy2ply_script, npy_path]
            print(f"Running conversion: {' '.join(cmd_convert)}")
            try:
                subprocess.check_call(cmd_convert)
                ply_paths.append(npy_path[:-4] + ".ply")
            except subprocess.CalledProcessError as e:
                print(f"Error during npy2ply conversion: {e}")
                return

        if args.cleanup:
            cleanup_script = os.path.join(script_dir, "cleanup_pointcloud.py")
            for ply_path in ply_paths:
                cmd_cleanup = [
                    sys.executable,
                    cleanup_script,
                    "--input_file",
                    ply_path,
                    "--algorithm",
                    args.cleanup_algo,
                ]
                if args.cleanup_algo == "sor":
                    cmd_cleanup.extend(["--nb_neighbors", str(args.sor_nb_neighbors)])
                    cmd_cleanup.extend(["--std_ratio", str(args.sor_std_ratio)])
                elif args.cleanup_algo == "ror":
                    cmd_cleanup.extend(["--radius", str(args.ror_radius)])
                    cmd_cleanup.extend(["--min_neighbors", str(args.ror_min_neighbors)])

                print(f"Running cleanup: {' '.join(cmd_cleanup)}")
                try:
                    subprocess.check_call(cmd_cleanup)
                except subprocess.CalledProcessError as e:
                    print(f"Error during cleanup: {e}")


if __name__ == "__main__":
    parser = build_arg_parser()
    run_export_pointcloud(parser.parse_args())
