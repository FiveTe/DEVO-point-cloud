#!/usr/bin/env python3
import argparse
import numpy as np
from plyfile import PlyData, PlyElement


def main():
    parser = argparse.ArgumentParser(description='Convert DEVO point cloud to PLY')
    parser.add_argument('--points', required=True, help='Path to point cloud .npy (N x 3)')
    parser.add_argument('--out', default=None, help='Output .ply path (defaults to <points>.ply)')
    args = parser.parse_args()

    pts = np.load(args.points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('points array must have shape (N,3)')

    out = args.out or args.points.replace('.npy', '.ply')
    vertex = np.array([tuple(p) for p in pts], dtype=[('x','f4'),('y','f4'),('z','f4')])
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(out)
    print(f'Saved {len(pts)} points to {out}')


if __name__ == '__main__':
    main()
