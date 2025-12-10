#!/usr/bin/env python3
import argparse
import numpy as np
from plyfile import PlyData, PlyElement


def main():
    parser = argparse.ArgumentParser(description="Save fused sparse map as .ply")
    parser.add_argument('--points', required=True, help='Numpy file with fused points (N x 3)')
    parser.add_argument('--out', default='fused_map.ply', help='Output PLY filename')
    args = parser.parse_args()

    pts = np.load(args.points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('points.npy must have shape (N, 3)')

    if pts.size == 0:
        raise SystemExit('No points to save')

    vertex = np.array([tuple(p) for p in pts], dtype=[('x','f4'),('y','f4'),('z','f4')])
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(args.out)
    print(f'Saved {len(pts)} points to {args.out}')


if __name__ == '__main__':
    main()
