#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description="Plot DEVO trajectory (optionally with sparse map)")
    parser.add_argument("--poses", required=True, help="Path to poses.npy (N x 7)")
    parser.add_argument("--points", help="Path to point_cloud.npy (M x 3) to overlay")
    parser.add_argument("--radius", type=float, default=None, help="Optional radius around the trajectory center to keep points")
    parser.add_argument("--out", default=None, help="Optional output PNG path")
    args = parser.parse_args()

    poses = np.load(args.poses)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError("poses array must have shape (N,7)")

    traj = poses[:, :3]
    cloud = None
    if args.points:
        cloud = np.load(args.points)
        if cloud.ndim != 2 or cloud.shape[1] != 3:
            raise ValueError("points array must have shape (M,3)")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    if cloud is not None and len(cloud) > 0:
        if args.radius is not None:
            center = traj.mean(axis=0)
            mask = np.linalg.norm(cloud - center, axis=1) <= args.radius
            cloud = cloud[mask]
        ax.scatter(cloud[:,0], cloud[:,1], cloud[:,2], s=0.5, c="gray", alpha=0.3)
    ax.plot(traj[:,0], traj[:,1], traj[:,2], color="red", linewidth=2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("DEVO Camera Trajectory")
    ax.set_xlim(traj[:,0].min()-1, traj[:,0].max()+1)
    ax.set_ylim(traj[:,1].min()-1, traj[:,1].max()+1)
    ax.set_zlim(traj[:,2].min()-1, traj[:,2].max()+1)
    ax.view_init(elev=25, azim=-70)
    plt.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=200)
        print(f"Saved trajectory to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
