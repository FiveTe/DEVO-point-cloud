#!/usr/bin/env python3
"""
Quick checks for Vector datasets:
- Inspect intrinsics and rectify map shape/stats.
- Summarize point cloud vs pose trajectory scale and suggest a scale factor.
- Optionally dump simple plots.
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def load_intrinsics(path: Path):
    if not path.exists():
        return None
    try:
        return np.loadtxt(path)
    except Exception as exc:  # pragma: no cover
        print(f"Failed to load intrinsics from {path}: {exc}")
        return None


def summarize_rectify_map(path: Path):
    if not path.exists():
        print(f"Rectify map missing: {path}")
        return None

    with h5py.File(path, "r") as f:
        rect = f["rectify_map"][:]
    stats = {
        "shape": rect.shape,
        "min": rect.min(),
        "max": rect.max(),
        "mean": rect.mean(),
        "std": rect.std(),
    }
    print(f"Rectify map {path}: {stats}")
    return rect


def summarize_intrinsics(intrinsics: np.ndarray, path: Path):
    if intrinsics is None:
        print(f"Intrinsics not found at {path}")
        return
    if intrinsics.shape[0] >= 4:
        fx, fy, cx, cy = intrinsics[:4]
        print(f"Intrinsics {path}: fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    else:
        print(f"Intrinsics {path}: {intrinsics}")


def summarize_geometry(pc: np.ndarray, poses: np.ndarray):
    if pc is None or pc.size == 0:
        print("Point cloud is empty or missing.")
        return
    if poses is None or poses.size == 0:
        print("Poses are empty or missing.")
        return

    pc_centroid = pc.mean(axis=0)
    pc_norms = np.linalg.norm(pc - pc_centroid, axis=1)
    traj = poses[:, :3]
    traj_centered = traj - traj[0]
    traj_norms = np.linalg.norm(traj_centered, axis=1)

    pc_range = np.percentile(pc_norms, [5, 50, 95])
    traj_range = np.percentile(traj_norms, [5, 50, 95])

    best_scale = (traj_norms.max() + 1e-9) / (pc_norms.max() + 1e-9)

    disp = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    print(f"Point cloud norms (5/50/95 pct): {pc_range}")
    print(f"Trajectory norms (5/50/95 pct): {traj_range}")
    print(f"Pose step median: {np.median(disp):.4f}, mean: {disp.mean():.4f}")
    print(f"Approx scale to match pc->traj range: {best_scale:.6f}")


def maybe_plot(pc, poses, rect, outdir: Path):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"matplotlib not available, skipping plots: {exc}")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    if rect is not None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].imshow(rect[..., 0], cmap="coolwarm")
        axs[0].set_title("Rectify map x")
        axs[1].imshow(rect[..., 1], cmap="coolwarm")
        axs[1].set_title("Rectify map y")
        for ax in axs:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(outdir / "rectify_map.png", dpi=200)
        plt.close(fig)

    if pc is not None and poses is not None and pc.size > 0 and poses.size > 0:
        fig, ax = plt.subplots(figsize=(5, 5))
        sample = pc
        if sample.shape[0] > 200000:
            idx = np.random.choice(sample.shape[0], 200000, replace=False)
            sample = sample[idx]
        ax.scatter(sample[:, 0], sample[:, 1], s=0.1, alpha=0.3, label="pc_xy")
        traj = poses[:, :3]
        ax.plot(traj[:, 0], traj[:, 1], "r-", linewidth=2, label="pose_xy")
        ax.legend()
        ax.set_aspect("equal")
        ax.set_title("XY view: pose vs point cloud")
        fig.tight_layout()
        fig.savefig(outdir / "pose_vs_pc_xy.png", dpi=200)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Debug Vector scale issues.")
    parser.add_argument("--scene", type=Path, default=Path("data/Vector_Cor/corridors_dolly"), help="Scene directory.")
    parser.add_argument("--pc", type=Path, default=Path("results/point_cloud.npy"), help="Point cloud .npy")
    parser.add_argument("--poses", type=Path, default=Path("results/point_cloud_poses.npy"), help="Poses .npy")
    parser.add_argument("--intrinsics", type=Path, default=None, help="Intrinsics file (fx fy cx cy).")
    parser.add_argument("--rectify", type=Path, default=None, help="Rectify map .h5")
    parser.add_argument("--save-plots", type=Path, default=None, help="Save simple diagnostic plots to this dir.")
    args = parser.parse_args()

    intr_path = args.intrinsics or args.scene / "calib_undist_evs_left.txt"
    rectify_path = args.rectify or args.scene / "rectify_map_left.h5"

    intrinsics = load_intrinsics(intr_path)
    summarize_intrinsics(intrinsics, intr_path)
    rect = summarize_rectify_map(rectify_path)

    pc = None
    poses = None
    if args.pc.exists():
        pc = np.load(args.pc)
    else:
        print(f"Point cloud file missing: {args.pc}")
    if args.poses.exists():
        poses = np.load(args.poses)
    else:
        print(f"Poses file missing: {args.poses}")

    summarize_geometry(pc, poses)

    if args.save_plots:
        maybe_plot(pc, poses, rect, args.save_plots)


if __name__ == "__main__":
    main()
