import open3d as o3d
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Visualize a PLY point cloud file.")
    parser.add_argument("path", help="Path to the .ply file")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"Error: File {args.path} not found.")
        return

    print(f"Loading {args.path}...")
    pcd = o3d.io.read_point_cloud(args.path)
    
    if pcd.is_empty():
        print("Warning: Point cloud is empty.")
    else:
        print(f"Loaded {len(pcd.points)} points.")

    print("Opening visualization window...")
    print("Controls:")
    print("  [Mouse] Rotate/Pan/Zoom")
    print("  [K]     Lock/Unlock keypad")
    print("  [+/-]   Increase/Decrease point size")
    print("  [R]     Reset view")
    print("  [Q]     Quit")
    
    o3d.visualization.draw_geometries([pcd], window_name=f"Open3D - {os.path.basename(args.path)}")

if __name__ == "__main__":
    main()
