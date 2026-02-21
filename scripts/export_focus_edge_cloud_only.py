import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from utils.load_utils import fpv_evs_iterator, rpg_evs_iterator, vector_evs_iterator
from utils.focus_edge_cloud_torch import (
    backproject_world_torch,
    focus_refine_inv_depth_torch,
    knn_inv_depth_prior_torch,
    sobel_edges_topk_qres_torch,
)


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Export Pipeline A focus-refined edge cloud without running DEVO.")
    ap.add_argument("--datapath", required=True, help="Dataset folder (same as export_pointcloud.py)")
    ap.add_argument("--frames_npz", required=True, help="Path to *_frames.npz produced by export_pointcloud.py --export_frame_data")
    ap.add_argument("--out", required=True, help="Output .npy path for the focus edge cloud")

    ap.add_argument("--dataset", choices=["fpv", "rpg", "vector"], default="fpv")
    ap.add_argument("--stride", type=int, default=1, help="Frame/event stride for the iterator")
    ap.add_argument("--side", type=str, default="left", help="Camera side (used for stereo datasets such as RPG/Vector)")
    ap.add_argument("--start_us", type=float, default=None, help="Start timestamp (microseconds) for FPV/Vector sequences")
    ap.add_argument("--stop_us", type=float, default=None, help="Stop timestamp (microseconds) for FPV/Vector sequences")

    ap.add_argument("--focus_downsample", type=int, default=4, help="Downsample factor for focus edge pixels (default: 4).")
    ap.add_argument("--focus_topk", type=int, default=6000, help="Edge pixels per reference keyframe (qres).")
    ap.add_argument("--focus_border", type=int, default=2, help="Border to suppress in focus qres pixels.")
    ap.add_argument("--focus_knn", type=int, default=4, help="kNN for inverse-depth prior from DEVO patch centers.")
    ap.add_argument("--focus_max_dist", type=float, default=6.0, help="Max kNN radius for inverse-depth prior (qres).")
    ap.add_argument("--focus_support", type=int, default=4, help="Support frames on each side of reference.")
    ap.add_argument("--focus_support_stride", type=int, default=1, help="Stride between support frames.")
    ap.add_argument("--focus_rho_samples", type=int, default=25, help="Inverse-depth samples per edge pixel.")
    ap.add_argument("--focus_rho_rel", type=float, default=0.2, help="Relative inverse-depth band (+/- rho_rel * rho0).")
    ap.add_argument("--focus_rho_abs", type=float, default=None, help="Absolute inverse-depth band (+/- rho_abs). Overrides rho_rel.")
    ap.add_argument("--focus_patch_radius", type=int, default=1, help="Patch radius for event evidence (0=bilinear at point).")
    ap.add_argument("--focus_min_peak_ratio", type=float, default=1.2, help="Min peak-to-second ratio for accepting a depth.")
    ap.add_argument("--focus_min_score", type=float, default=0.0, help="Min focus score for accepting a depth.")
    ap.add_argument("--focus_min_baseline", type=float, default=0.0, help="Min camera baseline (world units) to include a support frame.")
    ap.add_argument("--focus_keyframe_stride", type=int, default=1, help="Only refine every Nth keyframe from frames_npz.")
    ap.add_argument("--focus_max_keyframes", type=int, default=0, help="Cap number of keyframes to refine (0=all).")
    ap.add_argument("--cache_dtype", choices=["fp16", "fp32"], default="fp16", help="Event image cache dtype (speed/memory tradeoff).")
    ap.add_argument("--progress_every", type=int, default=200, help="Print progress every N slices/keyframes (0=disable).")
    ap.add_argument("--device", type=str, default="cuda", help="Torch device (e.g. cuda, cuda:0, cpu).")
    return ap


def _nearest_index(sorted_ts: np.ndarray, t: float) -> int:
    if sorted_ts.size == 0:
        return -1
    j = int(np.searchsorted(sorted_ts, t))
    if j <= 0:
        return 0
    if j >= sorted_ts.size:
        return int(sorted_ts.size - 1)
    return j if abs(sorted_ts[j] - t) < abs(sorted_ts[j - 1] - t) else (j - 1)


def _load_frames_npz(path: str) -> dict:
    data = np.load(path, allow_pickle=False)
    required = ["timestamps", "poses", "intrinsics", "counts", "offsets", "depths", "centers"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{path} missing keys: {missing}")
    return {k: data[k] for k in data.files}


def _iter_dataset(args):
    dataset = args.dataset.lower()
    if dataset == "fpv":
        return fpv_evs_iterator(
            args.datapath,
            stride=args.stride,
            t_start_us=args.start_us,
            t_stop_us=args.stop_us,
        )
    if dataset == "rpg":
        return rpg_evs_iterator(args.datapath, side=args.side, stride=args.stride)
    if dataset == "vector":
        # Vector iterator expects calib files directly under `indir`.
        # Some datasets are nested as `<root>/<seq_name>/...`.
        indir = args.datapath
        calib_name = f"calib_undist_evs_{args.side}.txt"
        if not os.path.exists(os.path.join(indir, calib_name)):
            base = os.path.basename(os.path.normpath(indir))
            candidates = []
            candidate_same_name = os.path.join(indir, base)
            if os.path.isdir(candidate_same_name):
                candidates.append(candidate_same_name)
            # If there's exactly one subdir, try it as well.
            try:
                subdirs = [os.path.join(indir, d) for d in os.listdir(indir) if os.path.isdir(os.path.join(indir, d))]
            except Exception:
                subdirs = []
            if len(subdirs) == 1:
                candidates.append(subdirs[0])
            for cand in candidates:
                if os.path.exists(os.path.join(cand, calib_name)):
                    print(f"[focus-only] Using Vector sequence folder: {cand}")
                    indir = cand
                    break

        return vector_evs_iterator(
            indir,
            side=args.side,
            stride=args.stride,
            t_start_us=args.start_us,
            t_stop_us=args.stop_us,
            H=480,
            W=640,
        )
    raise ValueError(f"Unsupported dataset '{args.dataset}'")


def main(args):
    frames = _load_frames_npz(args.frames_npz)
    frame_ts = np.asarray(frames["timestamps"], dtype=np.float64).reshape(-1)
    frame_poses = np.asarray(frames["poses"], dtype=np.float32)

    counts = np.asarray(frames["counts"], dtype=np.int64).reshape(-1)
    offsets = np.asarray(frames["offsets"], dtype=np.int64).reshape(-1)
    depths_concat = np.asarray(frames["depths"], dtype=np.float32).reshape(-1)
    centers_padded = np.asarray(frames["centers"], dtype=np.float32)

    ds = int(max(args.focus_downsample, 1))
    cache_dtype = torch.float16 if args.cache_dtype == "fp16" else torch.float32
    progress_every = int(max(args.progress_every, 0))
    device = torch.device(args.device)

    # 1) Build (timestamp -> cached event image + intrinsics in qres coords)
    focus_cache = []
    t_cache0 = time.time()
    last_print = t_cache0
    for voxel, intr, ts_us in _iter_dataset(args):
        with torch.no_grad():
            img = voxel.abs().sum(dim=0, keepdim=False)  # (H,W)
            img_q = F.avg_pool2d(img[None, None], kernel_size=ds, stride=ds).squeeze(0).squeeze(0)
            focus_cache.append(
                {
                    "timestamp": float(ts_us),
                    "event_img_qres": img_q.to(device=device, dtype=cache_dtype, non_blocking=True),
                    "intr_qres": (intr.to(device=device, dtype=torch.float32, non_blocking=True) / float(ds)),
                }
            )
        if progress_every > 0 and (len(focus_cache) % progress_every == 0):
            now = time.time()
            dt = now - last_print
            total_dt = now - t_cache0
            hz = (progress_every / max(dt, 1e-6))
            print(f"[focus-only] cached {len(focus_cache)} slices ({hz:.1f} slices/s, {total_dt:.1f}s elapsed)")
            last_print = now

    cache_ts = np.array([c["timestamp"] for c in focus_cache], dtype=np.float64)
    print(f"[focus-only] finished caching {len(focus_cache)} slices in {time.time() - t_cache0:.1f}s")

    # Map each cached slice to a pose by nearest timestamp in frames_npz.
    pose_for_cache = []
    for t in cache_ts:
        pi = _nearest_index(frame_ts, float(t))
        if pi < 0:
            pose_for_cache.append(torch.zeros((7,), device=device, dtype=torch.float32))
        else:
            pose_for_cache.append(torch.as_tensor(frame_poses[pi], device=device, dtype=torch.float32))

    # 2) Per-keyframe focus refinement using sparse inverse-depth seeds from frames_npz.
    focus_points = []
    kstride = max(int(args.focus_keyframe_stride), 1)
    max_k = int(args.focus_max_keyframes)
    keyframe_indices = list(range(0, len(frame_ts), kstride))
    if max_k > 0:
        keyframe_indices = keyframe_indices[:max_k]

    for fi in keyframe_indices:
        if progress_every > 0 and (fi % progress_every == 0):
            print(f"[focus-only] refining keyframe idx {fi}/{len(frame_ts)}")
        t0 = float(frame_ts[fi])
        ref_idx = _nearest_index(cache_ts, t0)
        if ref_idx < 0 or ref_idx >= len(focus_cache):
            continue

        ref_event_img = focus_cache[ref_idx]["event_img_qres"]
        ref_intr = focus_cache[ref_idx]["intr_qres"]
        ref_pose = pose_for_cache[ref_idx]

        edge_uv = sobel_edges_topk_qres_torch(
            ref_event_img.to(torch.float32),
            topk=int(args.focus_topk),
            border=int(args.focus_border),
        )
        if edge_uv.numel() == 0:
            continue

        # Seeds for this keyframe
        c = np.asarray(centers_padded[fi], dtype=np.float32)
        if c.ndim != 2 or c.shape[1] != 2:
            continue
        n = int(counts[fi]) if fi < counts.size else int(np.isfinite(c).all(axis=1).sum())
        n = max(min(n, c.shape[0]), 0)
        centers = c[:n]
        inv_depth_seeds = depths_concat[int(offsets[fi]) : int(offsets[fi] + n)].astype(np.float32, copy=False)

        good = np.isfinite(centers).all(axis=1) & np.isfinite(inv_depth_seeds) & (inv_depth_seeds > 0)
        centers = centers[good]
        inv_depth_seeds = inv_depth_seeds[good]
        if centers.size == 0:
            continue

        if ds != 4:
            # frames_npz stores DEVO patch centers in quarter-res (downsample=4) coords.
            centers = centers * (4.0 / float(ds))

        inv0, valid = knn_inv_depth_prior_torch(
            edge_uv,
            torch.as_tensor(centers, device=device, dtype=torch.float32),
            torch.as_tensor(inv_depth_seeds, device=device, dtype=torch.float32),
            knn=int(args.focus_knn),
            max_dist=float(args.focus_max_dist),
        )
        if not bool(valid.any().item()):
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

        inv_star, _conf = focus_refine_inv_depth_torch(
            edge_uv,
            inv0,
            ref_pose,
            ref_intr,
            sup,
            rho_samples=int(args.focus_rho_samples),
            rho_rel=float(args.focus_rho_rel),
            rho_abs=None if args.focus_rho_abs is None else float(args.focus_rho_abs),
            patch_radius=int(args.focus_patch_radius),
            min_peak_ratio=float(args.focus_min_peak_ratio),
            min_score=float(args.focus_min_score),
            min_baseline=float(args.focus_min_baseline),
        )

        pw = backproject_world_torch(edge_uv, inv_star, ref_pose, ref_intr)
        if pw.numel() > 0:
            focus_points.append(pw)

    if focus_points:
        focus_cloud = torch.cat(focus_points, dim=0).detach().cpu().numpy().astype(np.float32)
    else:
        focus_cloud = np.zeros((0, 3), dtype=np.float32)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.out, focus_cloud)
    print(f"Saved {focus_cloud.shape[0]} points to {args.out}")


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())
