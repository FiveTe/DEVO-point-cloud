import open3d as o3d
import numpy as np
import os

class PointCloudAccumulator:
    def __init__(self, output_dir, voxel_size=0.05):
        self.output_dir = output_dir
        self.voxel_size = voxel_size
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        self.global_pcd = o3d.geometry.PointCloud()

    def add_points(self, points, colors=None):
        """
        Add points to the global map and downsample.
        points: (N, 3) numpy array
        colors: (N, 3) numpy array (optional)
        """
        if points is None or len(points) == 0:
            return

        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(points)
        
        if colors is not None and len(colors) == len(points):
            new_pcd.colors = o3d.utility.Vector3dVector(colors)

        # Add to global
        self.global_pcd += new_pcd
        
        # Downsample immediately to keep memory usage bounded
        self.global_pcd = self.global_pcd.voxel_down_sample(voxel_size=self.voxel_size)

    def save_combined_map(self, filename="combined_map.ply"):
        save_path = os.path.join(self.output_dir, filename)
        o3d.io.write_point_cloud(save_path, self.global_pcd)
        print(f"Saved combined global map to {save_path}")

    def cleanup_individual_files(self):
        """
        Delete all .ply files in the output directory except the combined map.
        """
        print(f"Cleaning up individual .ply files in {self.output_dir}...")
        for filename in os.listdir(self.output_dir):
            if filename.endswith(".ply") and "combined_map" not in filename:
                file_path = os.path.join(self.output_dir, filename)
                try:
                    os.remove(file_path)
                except OSError as e:
                    print(f"Error deleting {file_path}: {e}")
        print("Cleanup complete.")

def save_point_clouds_to_ply(flowdata, output_dir, scale=1.0):
    """
    Legacy function. 
    Use PointCloudAccumulator for incremental saving.
    """
    print("Warning: save_point_clouds_to_ply is deprecated. Use PointCloudAccumulator.")
    pass

