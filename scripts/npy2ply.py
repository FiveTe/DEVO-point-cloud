#!/usr/bin/env python3
import argparse
import numpy as np
import os
import sys
from plyfile import PlyData, PlyElement

def main():
    parser = argparse.ArgumentParser(description='Convert .npy point cloud to .ply')
    parser.add_argument('input', help='Path to input .npy file')
    args = parser.parse_args()

    input_path = args.input
    if not input_path.endswith('.npy'):
        print(f"Error: Input file '{input_path}' does not have .npy extension")
        sys.exit(1)

    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found")
        sys.exit(1)

    try:
        pts = np.load(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        sys.exit(1)

    # Check shape
    if pts.ndim != 2 or pts.shape[1] != 3:
        print(f"Error: Expected shape (N, 3), got {pts.shape}")
        sys.exit(1)

    output_path = input_path[:-4] + '.ply'
    
    # Create structured array for plyfile
    try:
        vertex = np.array([tuple(p) for p in pts], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        PlyData([PlyElement.describe(vertex, 'vertex')]).write(output_path)
        print(f"Successfully converted '{input_path}' to '{output_path}' ({len(pts)} points)")
    except Exception as e:
        print(f"Error writing to {output_path}: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
