#!/usr/bin/env python3
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def pose_to_matrix(pose):
    t = pose[:3]
    q = pose[3:]
    rot = R.from_quat(q).as_matrix()
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = t
    return mat


def build_frustum_lines(poses, intrinsics, width, height, scale):
    fx, fy, cx, cy = intrinsics
    corners = np.array([
        [0.0, 0.0, 0.0],
        [(0 - cx) / fx, (0 - cy) / fy, 1.0],
        [(width - cx) / fx, (0 - cy) / fy, 1.0],
        [(width - cx) / fx, (height - cy) / fy, 1.0],
        [(0 - cx) / fx, (height - cy) / fy, 1.0],
    ]) * scale

    line_indices = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
    frusta = []
    for pose in poses:
        tf = pose_to_matrix(pose)
        corners_h = np.hstack([corners, np.ones((corners.shape[0], 1))])
        world_pts = (tf @ corners_h.T).T[:, :3]

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(world_pts)
        ls.lines = o3d.utility.Vector2iVector(line_indices)
        ls.colors = o3d.utility.Vector3dVector([[0, 0, 1]] * len(line_indices))
        frusta.append(ls)
    return frusta


def main():
    parser = argparse.ArgumentParser(description="Open3D viewer for DEVO point clouds and trajectory")
    parser.add_argument('--points', required=True, help='Path to point cloud .npy (N x 3)')
    parser.add_argument('--poses', required=False, help='Optional poses .npy (M x 7) to overlay trajectory')
    parser.add_argument('--downsample', type=float, default=None, help='Optional voxel size for downsampling')
    parser.add_argument('--radius', type=float, default=None, help='Optional radius around trajectory center to keep points')
    parser.add_argument('--intrinsics', type=float, nargs=4, metavar=('fx','fy','cx','cy'),
                        default=[458.0, 458.0, 319.5, 239.5], help='Camera intrinsics for frustums')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--frustum-scale', type=float, default=0.15)
    args = parser.parse_args()

    pts = np.load(args.points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('points array must have shape (N,3)')

    if args.poses:
        poses = np.load(args.poses)
        traj = poses[:, :3]
        center = traj.mean(axis=0)
        if args.radius is not None:
            mask = np.linalg.norm(pts - center, axis=1) <= args.radius
            pts = pts[mask]
    else:
        traj = None
        center = pts.mean(axis=0)
        if args.radius is not None:
            mask = np.linalg.norm(pts - center, axis=1) <= args.radius
            pts = pts[mask]

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    cloud.paint_uniform_color([0, 0, 0])

    if args.downsample is not None and args.downsample > 0:
        cloud = cloud.voxel_down_sample(args.downsample)

    geometries = [cloud]
    if traj is not None:
        traj_ls = o3d.geometry.LineSet()
        traj_ls.points = o3d.utility.Vector3dVector(traj)
        traj_ls.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(traj)-1)])
        traj_ls.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * (len(traj)-1))
        geometries.append(traj_ls)
        frusta = build_frustum_lines(poses, args.intrinsics, args.width, args.height, args.frustum_scale)
        geometries.extend(frusta)

    o3d.visualization.draw_geometries(geometries)


if __name__ == '__main__':
    main()
