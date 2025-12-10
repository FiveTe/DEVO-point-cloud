#!/usr/bin/env python3
import argparse
import numpy as np
import torch


try:
    from dpviewer import Viewer
except ImportError as exc:  # pragma: no cover
    raise SystemExit("dpviewer module not found. Build DPViewer first.") from exc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a saved DEVO/DPVO map in the Pangolin viewer."
    )
    parser.add_argument("--poses", required=True, help="Path to poses.npy (N x 7)")
    parser.add_argument("--points", required=True, help="Path to point_cloud.npy (M x 3)")
    parser.add_argument(
        "--colors",
        default=None,
        help="Optional path to colors.npy (M x 3, floats in [0,1] or uint8).",
    )
    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs=4,
        metavar=("FX", "FY", "CX", "CY"),
        default=(320.0, 320.0, 320.0, 240.0),
        help="Camera intrinsics fx fy cx cy (default: 320 320 320 240)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"Loading poses from {args.poses}")

    poses = np.load(args.poses)
    points = np.load(args.points)

    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError("Poses must have shape (N, 7)")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points must have shape (M, 3)")

    colors = None
    if args.colors is not None:
        print(f"Loading colors from {args.colors}")
        colors = np.load(args.colors)
        if colors.shape != points.shape:
            raise ValueError("Colors must have same shape as points")
        if colors.dtype != np.uint8:
            colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    else:
        colors = np.full(points.shape, 255, dtype=np.uint8)

    device = torch.device("cuda")

    print(f"Initializing viewer with {poses.shape[0]} poses and {points.shape[0]} points")
    poses_t = torch.from_numpy(poses).to(device=device, dtype=torch.float32)
    points_t = torch.from_numpy(points).to(device=device, dtype=torch.float32)
    colors_t = torch.from_numpy(colors).to(device=device, dtype=torch.uint8)

    intrinsics = np.asarray(args.intrinsics, dtype=np.float32)
    intrinsics_t = torch.from_numpy(np.tile(intrinsics, (poses.shape[0], 1))).to(
        device=device
    )

    dummy_image = torch.zeros(args.height, args.width, 3, dtype=torch.uint8, device="cpu")

    print("Starting DPViewer...")
    viewer = Viewer(dummy_image, poses_t, points_t, colors_t, intrinsics_t)
    print("Viewer running; close the window to exit.")
    viewer.join()


if __name__ == "__main__":
    main()
