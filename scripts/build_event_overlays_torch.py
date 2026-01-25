"""
Build per-frame event overlays using the torch-based Vector iterator.

For each frame in a DEVO frames.npz:
  - pull the matching rectified event voxel from vector_evs_iterator
  - collapse to a 2D event image
  - project the sparse points onto the event image (intrinsics upscaled by RES)
  - save overlay PNGs (if OpenCV is installed) and per-frame npz blobs

Example:
python scripts/build_event_overlays_torch.py \
  --frames results/corridor_first40s_fullPF256/frames.npz \
  --scene data/Vector_Cor/corridors_dolly \
  --side left \
  --out results/corridor_first40s_fullPF256/overlays_torch \
  --res-scale 4.0
"""

import argparse
import os
import pathlib
import warnings

import numpy as np
import torch

try:
    import cv2

    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

from utils.load_utils import vector_evs_iterator


def quat_to_rotmat(q):
    """Quaternion (qx,qy,qz,qw) -> 3x3 rotation matrix."""
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def project_frame(frames_npz, frame_idx, res_scale=4.0):
    """Project sparse points of a frame onto the event image plane."""
    fx, fy, cx, cy = frames_npz["intrinsics"][frame_idx] * res_scale
    t = frames_npz["poses"][frame_idx, :3]
    q = frames_npz["poses"][frame_idx, 3:]
    R_c2w = quat_to_rotmat(q)
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


def build_overlays_torch(frames_path, scene_dir, side, out_dir, res_scale=4.0, max_frames=None):
    frames_npz = np.load(frames_path)
    total_frames = len(frames_npz["frame_ids"])
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    out_dir = pathlib.Path(out_dir)
    overlays_dir = out_dir / "overlays"
    data_dir = out_dir / "npz_frames"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    iterator = vector_evs_iterator(scene_dir, side=side, H=480, W=640, stride=1, dT_ms=None)

    for i, (voxel, _, ts_us) in enumerate(iterator):
        if i >= total_frames:
            break
        event_img = voxel.abs().sum(0).cpu().numpy()
        u, v, pts_c = project_frame(frames_npz, i, res_scale=res_scale)

        np.savez_compressed(
            data_dir / f"frame_{i:05d}.npz",
            frame_id=int(frames_npz["frame_ids"][i]),
            timestamp=float(frames_npz["timestamps"][i]),
            event_image=event_img.astype(np.float32),
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

        print(f"[frame {i}] ts={ts_us:.0f} saved overlay + data ({len(u)} points)")


def main():
    ap = argparse.ArgumentParser(description="Build event overlays with torch (Vector iterator).")
    ap.add_argument("--frames", required=True, help="Path to frames.npz from DEVO export.")
    ap.add_argument("--scene", required=True, help="Scene directory (e.g., data/Vector_Cor/corridors_dolly).")
    ap.add_argument("--side", default="left", choices=["left", "right"], help="Camera side.")
    ap.add_argument("--out", required=True, help="Output directory for overlays and per-frame npz.")
    ap.add_argument("--res-scale", type=float, default=4.0, help="Scale intrinsics from frames.npz to event image size.")
    ap.add_argument("--max-frames", type=int, default=None, help="Process only the first N frames.")
    args = ap.parse_args()

    build_overlays_torch(
        frames_path=args.frames,
        scene_dir=args.scene,
        side=args.side,
        out_dir=args.out,
        res_scale=args.res_scale,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
