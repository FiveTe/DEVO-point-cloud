#!/usr/bin/env python3
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def pose_to_tf(pose):
    t = pose[:3]
    q = pose[3:]
    rot = R.from_quat(q).as_matrix()
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = rot
    tf[:3, 3] = t
    return tf


def main():
    parser = argparse.ArgumentParser(
        description="Offline viewer for DEVO/DPVO sparse maps",
    )
    parser.add_argument("--points", required=True, help="point cloud .npy file (N x 3)")
    parser.add_argument("--poses", required=True, help="poses .npy file (M x 7)")
    parser.add_argument(
        "--intrinsics",
        type=float,
        nargs=4,
        metavar=("fx", "fy", "cx", "cy"),
        default=[458.0, 458.0, 319.5, 239.5],
        help="Camera intrinsics (fx fy cx cy)",
    )
    parser.add_argument("--width", type=int, default=640, help="Image width")
    parser.add_argument("--height", type=int, default=480, help="Image height")
    parser.add_argument("--scale", type=float, default=0.15, help="Frustum length scale")
    args = parser.parse_args()

    points = np.load(args.points)
    poses = np.load(args.poses)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points.npy must have shape (N, 3)")
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError("poses.npy must have shape (M, 7)")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    traj = o3d.geometry.LineSet()
    traj.points = o3d.utility.Vector3dVector(poses[:, :3])
    traj.lines = o3d.utility.Vector2iVector([[i, i + 1] for i in range(len(poses) - 1)])
    traj.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]] * (len(poses) - 1))

    fx, fy, cx, cy = args.intrinsics
    frustums = []
    base_corners = np.array([
        [0.0, 0.0, 0.0],
        [(0 - cx) / fx, (0 - cy) / fy, 1.0],
        [(args.width - cx) / fx, (0 - cy) / fy, 1.0],
        [(args.width - cx) / fx, (args.height - cy) / fy, 1.0],
        [(0 - cx) / fx, (args.height - cy) / fy, 1.0],
    ]) * args.scale

    line_indices = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]

    for pose in poses:
        tf = pose_to_tf(pose)
        corners_h = np.hstack([base_corners, np.ones((base_corners.shape[0], 1))])
        world = (tf @ corners_h.T).T[:, :3]

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(world)
        ls.lines = o3d.utility.Vector2iVector(line_indices)
        ls.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 1.0]] * len(line_indices))
        frustums.append(ls)

    o3d.visualization.draw_geometries([cloud, traj, *frustums])


if __name__ == "__main__":
    main()
