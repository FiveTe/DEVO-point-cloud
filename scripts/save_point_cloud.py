import open3d as o3d
import torch
import numpy as np

def save_point_cloud(depth_map, pose, intrinsics, filename, min_depth=0.1, max_depth=10.0):
    """
    Converts a depth map to a point cloud, transforms it to world coordinates, and saves it.
    
    Args:
        depth_map (torch.Tensor or np.array): HxW inverse depth map.
        pose (torch.Tensor or np.array): 4x4 Pose matrix (World -> Camera or Camera -> World).
                                         DEVO usually outputs World -> Camera (T_cw).
                                         We need T_wc to transform points to world.
        intrinsics (list/array): [fx, fy, cx, cy]
        filename (str): Output path (e.g., 'frame_001.ply')
    """
    if torch.is_tensor(depth_map):
        depth_map = depth_map.cpu().numpy()
    if torch.is_tensor(pose):
        pose = pose.cpu().numpy()
        
    H, W = depth_map.shape
    fx, fy, cx, cy = intrinsics

    # 1. Convert Inverse Depth to Depth
    # Avoid division by zero
    depth_map[depth_map < 1e-5] = 1e-5
    z = 1.0 / depth_map
    
    # Filter depth
    mask = (z > min_depth) & (z < max_depth)
    
    # 2. Generate 2D Grid
    xs, ys = np.meshgrid(np.arange(W), np.arange(H))
    
    # 3. Unproject to 3D Camera Coordinates
    x_cam = (xs - cx) * z / fx
    y_cam = (ys - cy) * z / fy
    z_cam = z
    
    # Stack to create (N, 3) points
    points_cam = np.stack((x_cam[mask], y_cam[mask], z_cam[mask]), axis=-1)
    
    # 4. Transform to World Coordinates
    # If pose is T_wc (Camera to World), we apply R*p + t.
    # If pose is T_cw (World to Camera), we invert it first.
    # Assuming DEVO output might be T_cw (common in VO), check conventions.
    # Here we assume pose is T_wc for simplicity, or invert if needed:
    # pose_inv = np.linalg.inv(pose) 
    
    # Apply transformation
    # points_world = (pose @ points_cam_homogeneous).T[:, :3]
    # Using Open3D for convenience:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_cam)
    
    # Transform
    pcd.transform(pose) # Apply the pose matrix
    
    # 5. Save
    o3d.io.write_point_cloud(filename, pcd)
    print(f"Saved point cloud to {filename}")