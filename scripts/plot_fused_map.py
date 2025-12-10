#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_fused_points(frame_npz):
    data = np.load(frame_npz)
    points = data.get('points')
    counts = data.get('counts')
    offsets = data.get('offsets')

    if points is None or counts is None or offsets is None:
        raise ValueError("frames npz must contain 'points', 'counts', and 'offsets'")

    if counts.ndim != 1 or offsets.ndim != 1:
        raise ValueError("counts and offsets must be 1-D arrays")

    if points.size == 0 or counts.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    total_points = offsets[-1] + counts[-1]
    if total_points != points.shape[0]:
        raise ValueError("counts/offsets do not align with points length")

    return points


def main():
    parser = argparse.ArgumentParser(description="Fuse per-frame sparse maps and plot with trajectory")
    parser.add_argument('--poses', required=True, help='Path to poses.npy (N x 7)')
    parser.add_argument('--frames', required=True, help='Path to <out>_frames.npz generated with --export_frame_data')
    parser.add_argument('--out', default='fused_map.png', help='Output PNG path')
    parser.add_argument('--points-out', default=None, help='Optional .npy file to save fused points')
    parser.add_argument('--radius', type=float, default=None, help='Optional radius around the trajectory center to filter points')
    args = parser.parse_args()

    poses = np.load(args.poses)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError('poses array must have shape (N,7)')

    traj = poses[:, :3]
    fused_points = load_fused_points(args.frames)
    if args.radius is not None and fused_points.size > 0:
        center = traj.mean(axis=0)
        mask = np.linalg.norm(fused_points - center, axis=1) <= args.radius
        fused_points = fused_points[mask]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if fused_points.size > 0:
        ax.scatter(fused_points[:,0], fused_points[:,1], fused_points[:,2], s=0.5, c='gray', alpha=0.3)
    ax.plot(traj[:,0], traj[:,1], traj[:,2], color='red', linewidth=2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Fused Sparse Map + Camera Trajectory')
    ax.set_xlim(traj[:,0].min()-1, traj[:,0].max()+1)
    ax.set_ylim(traj[:,1].min()-1, traj[:,1].max()+1)
    ax.set_zlim(traj[:,2].min()-1, traj[:,2].max()+1)
    ax.view_init(elev=25, azim=-70)
    plt.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f'Saved fused map figure to {args.out}')

    if args.points_out is not None and fused_points.size > 0:
        np.save(args.points_out, fused_points)
        print(f'Saved fused points to {args.points_out}')


if __name__ == '__main__':
    main()
