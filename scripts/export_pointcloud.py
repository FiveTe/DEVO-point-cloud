import argparse, os, json, subprocess, sys
import numpy as np
import torch
import torch.nn.functional as F
from devo.config import cfg
from utils.eval_utils import run_voxel
from utils.load_utils import fpv_evs_iterator, rpg_evs_iterator, vector_evs_iterator
from utils.focus_edge_cloud import (
    backproject_edge_points_world,
    extract_edge_pixels_qres,
    inv_depth_prior_knn,
    refine_edge_inv_depth_focus,
)


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
    parser.add_argument(
        "--export_focus_edge_cloud",
        action="store_true",
        help="Export an edge-aligned point cloud with multi-view focus refinement (Pipeline A-style).",
    )
    parser.add_argument(
        "--focus_edge_cloud_out",
        type=str,
        default=None,
        help="Path for the focus-refined edge point cloud .npy (defaults to <out>_focus_edges.npy).",
    )
    parser.add_argument("--focus_downsample", type=int, default=4, help="Downsample factor for focus edge pixels (default: 4).")
    parser.add_argument("--focus_topk", type=int, default=6000, help="Edge pixels per reference keyframe (qres).")
    parser.add_argument("--focus_border", type=int, default=2, help="Border to suppress in focus qres pixels.")
    parser.add_argument("--focus_knn", type=int, default=4, help="kNN for inverse-depth prior from DEVO patch centers.")
    parser.add_argument("--focus_max_dist", type=float, default=6.0, help="Max kNN radius for inverse-depth prior (qres).")
    parser.add_argument("--focus_support", type=int, default=4, help="Support frames on each side of reference.")
    parser.add_argument("--focus_support_stride", type=int, default=1, help="Stride between support frames.")
    parser.add_argument("--focus_rho_samples", type=int, default=25, help="Inverse-depth samples per edge pixel.")
    parser.add_argument("--focus_rho_rel", type=float, default=0.2, help="Relative inverse-depth band (+/- rho_rel * rho0).")
    parser.add_argument("--focus_rho_abs", type=float, default=None, help="Absolute inverse-depth band (+/- rho_abs). Overrides rho_rel.")
    parser.add_argument("--focus_patch_radius", type=int, default=1, help="Patch radius for event evidence (0=bilinear at point).")
    parser.add_argument("--focus_min_peak_ratio", type=float, default=1.2, help="Min peak-to-second ratio for accepting a depth.")
    parser.add_argument("--focus_min_score", type=float, default=0.0, help="Min focus score for accepting a depth.")
    parser.add_argument("--focus_min_baseline", type=float, default=0.0, help="Min camera baseline (world units) to include a support frame.")
    parser.add_argument("--focus_keyframe_stride", type=int, default=1, help="Only refine every Nth recorded keyframe.")
    parser.add_argument("--focus_max_keyframes", type=int, default=0, help="Cap number of keyframes to refine (0=all).")
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
    focus_edge_cloud_path = None
    if args.export_focus_edge_cloud:
        focus_edge_cloud_path = args.focus_edge_cloud_out or (base + "_focus_edges" + ext)
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

    # Wrap the iterator to cache a compact quarter-res event image per slice.
    # We keep only a collapsed 2D map (abs-sum across bins), downsampled by focus_downsample,
    # plus a matching intrinsics scaled into the same qres coordinate system.
    focus_cache = []

    def _cache_iterator(it):
        for voxel, intr, ts_us in it:
            if args.export_focus_edge_cloud:
                with torch.no_grad():
                    img = voxel.abs().sum(dim=0, keepdim=False)  # (H,W)
                    ds = int(max(args.focus_downsample, 1))
                    img_q = F.avg_pool2d(img[None, None], kernel_size=ds, stride=ds).squeeze(0).squeeze(0)
                    focus_cache.append(
                        {
                            "timestamp": float(ts_us),
                            "event_img_qres": img_q.detach().cpu().numpy().astype(np.float16),
                            "intr_qres": (intr.detach().cpu().numpy().astype(np.float32) / float(ds)),
                        }
                    )
            yield voxel, intr, ts_us

    iterator_wrapped = _cache_iterator(iterator)

    results = run_voxel(
        voxeldir=args.datapath,
        cfg=cfg,
        network=args.weights,
        iterator=iterator_wrapped,
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

    # Focus-refined edge cloud (Pipeline A-style)
    if args.export_focus_edge_cloud:
        if not args.export_frame_data:
            raise ValueError("--export_focus_edge_cloud requires --export_frame_data to access sparse depth seeds.")
        if focus_edge_cloud_path is None:
            focus_edge_cloud_path = base + "_focus_edges" + ext

        cache_ts = np.array([c["timestamp"] for c in focus_cache], dtype=np.float64)
        slam_ts = np.asarray(tstamps, dtype=np.float64).reshape(-1)

        def _nearest_index(sorted_ts: np.ndarray, t: float) -> int:
            if sorted_ts.size == 0:
                return -1
            j = int(np.searchsorted(sorted_ts, t))
            if j <= 0:
                return 0
            if j >= sorted_ts.size:
                return int(sorted_ts.size - 1)
            return j if abs(sorted_ts[j] - t) < abs(sorted_ts[j - 1] - t) else (j - 1)

        # Map cache frames to nearest pose index (timestamp-based, robust to skipped slices).
        pose_for_cache = []
        for t in cache_ts:
            pi = _nearest_index(slam_ts, float(t))
            pose_for_cache.append(np.asarray(poses[pi], dtype=np.float32))
        pose_for_cache = pose_for_cache

        focus_points = []
        keyframes = frame_data[:: max(int(args.focus_keyframe_stride), 1)]
        if int(args.focus_max_keyframes) > 0:
            keyframes = keyframes[: int(args.focus_max_keyframes)]

        for kf in keyframes:
            t0 = float(kf["timestamp"])
            ref_idx = _nearest_index(cache_ts, t0)
            if ref_idx < 0 or ref_idx >= len(focus_cache):
                continue

            ref_event_img = np.asarray(focus_cache[ref_idx]["event_img_qres"], dtype=np.float32)
            ref_intr = np.asarray(focus_cache[ref_idx]["intr_qres"], dtype=np.float32)
            ref_pose = pose_for_cache[ref_idx]

            edge_uv = extract_edge_pixels_qres(
                ref_event_img,
                topk=int(args.focus_topk),
                border=int(args.focus_border),
            )

            centers = np.asarray(kf.get("centers", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
            inv_depth_seeds = np.asarray(kf.get("depths", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
            if int(args.focus_downsample) != 4 and centers.size > 0:
                # DEVO patch centers are in quarter-res (downsample=4) coordinates.
                # Scale them into the focus image's coordinate system (downsample=focus_downsample).
                centers = centers * (4.0 / float(max(int(args.focus_downsample), 1)))
            inv0, valid = inv_depth_prior_knn(
                edge_uv,
                centers,
                inv_depth_seeds,
                knn=int(args.focus_knn),
                max_dist=float(args.focus_max_dist),
            )
            if not np.any(valid):
                continue

            # Support frames around reference (index-based window).
            sup = []
            s_each = int(max(args.focus_support, 0))
            s_stride = int(max(args.focus_support_stride, 1))
            for d in range(-s_each * s_stride, (s_each * s_stride) + 1, s_stride):
                if d == 0:
                    continue
                j = ref_idx + d
                if j < 0 or j >= len(focus_cache):
                    continue
                sup.append(
                    {
                        "pose": pose_for_cache[j],
                        "intr": focus_cache[j]["intr_qres"],
                        "event_img": focus_cache[j]["event_img_qres"],
                    }
                )

            inv_star, conf = refine_edge_inv_depth_focus(
                edge_uv,
                inv0,
                ref_pose=ref_pose,
                ref_intr=ref_intr,
                support_frames=sup,
                rho_samples=int(args.focus_rho_samples),
                rho_rel=float(args.focus_rho_rel),
                rho_abs=None if args.focus_rho_abs is None else float(args.focus_rho_abs),
                patch_radius=int(args.focus_patch_radius),
                min_peak_ratio=float(args.focus_min_peak_ratio),
                min_score=float(args.focus_min_score),
                min_baseline=float(args.focus_min_baseline),
            )

            pw = backproject_edge_points_world(edge_uv, inv_star, ref_pose, ref_intr)
            if pw.size > 0:
                focus_points.append(pw)

        if focus_points:
            focus_cloud = np.concatenate(focus_points, axis=0).astype(np.float32)
        else:
            focus_cloud = np.zeros((0, 3), dtype=np.float32)
        np.save(focus_edge_cloud_path, focus_cloud)

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
        "focus_edge_cloud_exported": bool(args.export_focus_edge_cloud),
        "focus_edge_cloud_path": focus_edge_cloud_path,
        "focus_topk": int(args.focus_topk),
        "focus_border": int(args.focus_border),
        "focus_knn": int(args.focus_knn),
        "focus_max_dist": float(args.focus_max_dist),
        "focus_support": int(args.focus_support),
        "focus_support_stride": int(args.focus_support_stride),
        "focus_rho_samples": int(args.focus_rho_samples),
        "focus_rho_rel": float(args.focus_rho_rel),
        "focus_rho_abs": args.focus_rho_abs,
        "focus_patch_radius": int(args.focus_patch_radius),
        "focus_min_peak_ratio": float(args.focus_min_peak_ratio),
        "focus_min_score": float(args.focus_min_score),
        "focus_min_baseline": float(args.focus_min_baseline),
        "focus_keyframe_stride": int(args.focus_keyframe_stride),
        "focus_max_keyframes": int(args.focus_max_keyframes),
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
        npy_paths = [pointcloud_path]
        if args.export_edge_cloud and edge_cloud_path is not None:
            npy_paths.append(edge_cloud_path)
        if args.export_focus_edge_cloud and focus_edge_cloud_path is not None:
            npy_paths.append(focus_edge_cloud_path)

        for npy_path in npy_paths:
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
