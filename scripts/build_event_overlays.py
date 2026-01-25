"""
Offline overlay builder that does NOT require torch.

Given a DEVO frames.npz (sparse map + poses) and the Vector event data,
this script:
  - slices rectified events per frame time window (using ms_to_idx)
  - rasterizes a rectified event image (sum of polarities)
  - projects the per-frame sparse points onto the event image
  - saves overlay PNGs and per-frame npz blobs (pixels + projected points)

Example:
python scripts/build_event_overlays.py \
  --frames results/corridor_first40s_fullPF256/frames.npz \
  --scene data/Vector_Cor/corridors_dolly \
  --side left \
  --out results/corridor_first40s_fullPF256/overlays \
  --window-us 70000
"""

import argparse
import os
import pathlib
import warnings

import h5py
import numpy as np

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


def _load_events_file(scene_dir: str, side: str):
    seq = os.path.basename(os.path.normpath(scene_dir))
    evt_path = os.path.join(scene_dir, f"{seq}1.synced.{side}_event.hdf5")
    if not os.path.isfile(evt_path):
        raise FileNotFoundError(f"Event file not found: {evt_path}")

    f = h5py.File(evt_path, "r")
    prefix = "events/" if "events/x" in f else ""

    xs = f[prefix + "x"]
    ys = f[prefix + "y"]
    ts = f[prefix + "t"]
    ps = f[prefix + "p"]

    if "ms_to_idx" not in f:
        raise KeyError("ms_to_idx dataset missing in event file; cannot slice efficiently.")
    ms_to_idx = np.asarray(f["ms_to_idx"], dtype=np.int64)
    t_offset = int(f["t_offset"][()]) if "t_offset" in f else 0

    return f, xs, ys, ts, ps, ms_to_idx, t_offset


def _slice_event_indices(ts_ds, ms_to_idx, t0_us, t1_us, t_offset):
    """Return [start, end) indices into event datasets covering [t0_us, t1_us)."""
    if t1_us <= t0_us:
        return None, None

    t0_adj = max(0, int(t0_us - t_offset))
    t1_adj = max(t0_adj + 1, int(t1_us - t_offset))

    start_ms = max(0, int(np.floor(t0_adj / 1000.0)))
    end_ms = int(np.ceil(t1_adj / 1000.0))
    end_ms = min(end_ms, len(ms_to_idx) - 1)

    start_idx = ms_to_idx[start_ms]
    end_idx = ts_ds.shape[0] if end_ms + 1 >= len(ms_to_idx) else ms_to_idx[end_ms + 1]
    if end_idx <= start_idx:
        return None, None

    ts_slice = np.asarray(ts_ds[start_idx:end_idx])
    local_start = np.searchsorted(ts_slice, t0_adj, side="left")
    local_end = np.searchsorted(ts_slice, t1_adj, side="left")
    return start_idx + local_start, start_idx + local_end


def _rectified_event_image(xs, ys, ts, ps, rect_map, t0_us, t1_us, nbins=5):
    """Rasterize events into a voxel grid, then collapse to a single image."""
    H, W, _ = rect_map.shape
    if xs.size == 0:
        return np.zeros((H, W), dtype=np.float32), np.zeros((nbins, H, W), dtype=np.float32)

    xr = rect_map[ys, xs, 0]
    yr = rect_map[ys, xs, 1]
    bins = ((ts - t0_us) / float(t1_us - t0_us) * nbins).astype(np.int64)
    bins = np.clip(bins, 0, nbins - 1)
    pol = np.where(ps > 0, 1.0, -1.0).astype(np.float32)

    voxel = np.zeros((nbins, H, W), dtype=np.float32)
    xr_i = np.clip(np.rint(xr).astype(np.int64), 0, W - 1)
    yr_i = np.clip(np.rint(yr).astype(np.int64), 0, H - 1)
    np.add.at(voxel, (bins, yr_i, xr_i), pol)

    event_img = np.abs(voxel).sum(axis=0)
    return event_img, voxel


def _project_points(frame_idx, frames_npz, res_scale=4.0):
    from scipy.spatial.transform import Rotation as R

    fx, fy, cx, cy = frames_npz["intrinsics"][frame_idx] * res_scale
    t = frames_npz["poses"][frame_idx, :3]
    q = frames_npz["poses"][frame_idx, 3:]
    R_c2w = R.from_quat(q).as_matrix()
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ t

    start = frames_npz["offsets"][frame_idx]
    count = frames_npz["counts"][frame_idx]
    pts_w = frames_npz["points"][start : start + count]

    pts_c = (R_w2c @ pts_w.T + t_w2c[:, None]).T
    z = pts_c[:, 2]
    valid = z > 0
    pts_c = pts_c[valid]
    u = fx * (pts_c[:, 0] / pts_c[:, 2]) + cx
    v = fy * (pts_c[:, 1] / pts_c[:, 2]) + cy
    return u, v, pts_c


def build_overlays(frames_path, scene_dir, side, out_dir, window_us=None, max_frames=None, res_scale=4.0):
    frames_npz = np.load(frames_path)
    timestamps = frames_npz["timestamps"]

    if window_us is None:
        if len(timestamps) > 1:
            window_us = float(np.median(np.diff(timestamps)))
        else:
            window_us = 70000.0
        warnings.warn(f"window_us not provided; using {window_us:.1f} from timestamps.")

    rect_path = os.path.join(scene_dir, f"rectify_map_{side}.h5")
    with h5py.File(rect_path, "r") as rmap:
        rect_map = np.array(rmap["rectify_map"])

    evt_file, xs_ds, ys_ds, ts_ds, ps_ds, ms_to_idx, t_offset = _load_events_file(scene_dir, side)

    out_dir = pathlib.Path(out_dir)
    overlays_dir = out_dir / "overlays"
    data_dir = out_dir / "npz_frames"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    total_frames = len(frames_npz["frame_ids"])
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    for i in range(total_frames):
        ts_mid = float(timestamps[i])
        half = window_us * 0.5
        t0 = ts_mid - half
        t1 = ts_mid + half

        idx0, idx1 = _slice_event_indices(ts_ds, ms_to_idx, t0, t1, t_offset)
        if idx0 is None or idx1 is None or idx1 <= idx0:
            print(f"[frame {i}] No events in window; skipping.")
            continue

        xs = np.asarray(xs_ds[idx0:idx1], dtype=np.int64)
        ys = np.asarray(ys_ds[idx0:idx1], dtype=np.int64)
        ts = np.asarray(ts_ds[idx0:idx1], dtype=np.int64) + t_offset
        ps = np.asarray(ps_ds[idx0:idx1])

        event_img, voxel = _rectified_event_image(xs, ys, ts, ps, rect_map, t0, t1)
        u, v, pts_c = _project_points(i, frames_npz, res_scale=res_scale)

        np.savez_compressed(
            data_dir / f"frame_{i:05d}.npz",
            frame_id=int(frames_npz["frame_ids"][i]),
            timestamp=ts_mid,
            event_image=event_img.astype(np.float32),
            voxel=voxel.astype(np.float32),
            u=u.astype(np.float32),
            v=v.astype(np.float32),
            points_cam=pts_c.astype(np.float32),
        )

        if _HAS_CV2:
            vis = cv2.normalize(event_img, None, 0, 255, cv2.NORM_MINMAX)
            vis = vis.astype(np.uint8)
            vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
            for x, y in zip(u, v):
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < vis.shape[1] and 0 <= yi < vis.shape[0]:
                    cv2.circle(vis, (xi, yi), 1, (0, 0, 255), -1)
            cv2.imwrite(str(overlays_dir / f"overlay_{i:05d}.png"), vis)
        else:
            if i == 0:
                warnings.warn("OpenCV not available; skipping overlay PNGs. npz data still saved.")

        print(f"[frame {i}] saved overlay + data (events {idx1 - idx0}).")

    evt_file.close()


def main():
    ap = argparse.ArgumentParser(description="Build event overlays without torch.")
    ap.add_argument("--frames", required=True, help="Path to frames.npz from DEVO export.")
    ap.add_argument("--scene", required=True, help="Scene directory (e.g., data/Vector_Cor/corridors_dolly).")
    ap.add_argument("--side", default="left", choices=["left", "right"], help="Camera side.")
    ap.add_argument("--out", required=True, help="Output directory for overlays and npz per frame.")
    ap.add_argument("--window-us", type=float, default=None, help="Event integration window in microseconds.")
    ap.add_argument("--max-frames", type=int, default=None, help="Process only the first N frames.")
    ap.add_argument("--res-scale", type=float, default=4.0, help="Scale intrinsics from frames.npz to event image size.")
    args = ap.parse_args()

    build_overlays(
        frames_path=args.frames,
        scene_dir=args.scene,
        side=args.side,
        out_dir=args.out,
        window_us=args.window_us,
        max_frames=args.max_frames,
        res_scale=args.res_scale,
    )


if __name__ == "__main__":
    main()
