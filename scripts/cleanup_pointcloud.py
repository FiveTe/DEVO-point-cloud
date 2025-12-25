#!/usr/bin/env python3
import open3d as o3d
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Cleanup a .ply point cloud using SOR or ROR methods.")
    
    parser.add_argument("--input_file", "-i", required=True, help="Path to the input .ply file")
    parser.add_argument("--algorithm", "-a", choices=["sor", "ror"], required=True, 
                        help="Outlier removal algorithm: 'sor' (Statistical Outlier Removal) or 'ror' (Radius Outlier Removal)")
    parser.add_argument("--output_file", "-o", help="Path to the output .ply file. If not provided, appends '_cleaned' to input filename.")
    
    # SOR parameters
    parser.add_argument("--nb_neighbors", type=int, default=20, 
                        help="[SOR] Number of neighbors to analyze for each point (default: 20)")
    parser.add_argument("--std_ratio", type=float, default=2.0, 
                        help="[SOR] Standard deviation ratio. Points with average distance larger than this ratio * std_dev will be removed. (default: 2.0)")
    
    # ROR parameters
    parser.add_argument("--radius", type=float, default=0.05, 
                        help="[ROR] Radius to search for neighbors (default: 0.05)")
    parser.add_argument("--min_neighbors", type=int, default=16, 
                        help="[ROR] Minimum number of neighbors required within the radius (default: 16)")
    
    parser.add_argument("--display", action="store_true", help="Visualize the result after cleaning (requires a display)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed information")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    print(f"Loading point cloud from {args.input_file}...")
    pcd = o3d.io.read_point_cloud(args.input_file)
    
    if pcd.is_empty():
        print("Error: The point cloud is empty or could not be read.")
        sys.exit(1)

    original_count = len(pcd.points)
    if args.verbose:
        print(f"Original point cloud has {original_count} points.")

    cl = None
    ind = None

    if args.algorithm == "sor":
        print(f"Applying Statistical Outlier Removal (nb_neighbors={args.nb_neighbors}, std_ratio={args.std_ratio})...")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=args.nb_neighbors,
                                                 std_ratio=args.std_ratio)
    elif args.algorithm == "ror":
        print(f"Applying Radius Outlier Removal (nb_points={args.min_neighbors}, radius={args.radius})...")
        cl, ind = pcd.remove_radius_outlier(nb_points=args.min_neighbors,
                                            radius=args.radius)

    cleaned_pcd = pcd.select_by_index(ind)
    cleaned_count = len(cleaned_pcd.points)
    removed_count = original_count - cleaned_count
    
    print(f"Cleanup complete. Removed {removed_count} points ({removed_count/original_count:.2%}). Remaining: {cleaned_count} points.")

    output_path = args.output_file
    if not output_path:
        base, ext = os.path.splitext(args.input_file)
        output_path = f"{base}_cleaned{ext}"

    print(f"Saving cleaned point cloud to {output_path}...")
    o3d.io.write_point_cloud(output_path, cleaned_pcd)

    if args.display:
        print("Visualizing result... (Close window to exit)")
        # Show original in red (faint) and cleaned in original colors or solid color if no colors
        # For simplicity, just show the cleaned cloud, or compare?
        # Let's show the cleaned cloud.
        window_name = f"Cleaned Point Cloud ({args.algorithm})"
        o3d.visualization.draw_geometries([cleaned_pcd], window_name=window_name)

if __name__ == "__main__":
    main()
