import numpy as np
from pathlib import Path
from typing import Iterable, Tuple

from evo.core.trajectory import PoseTrajectory3D


def _build_trajectory(poses: np.ndarray, timestamps: np.ndarray) -> PoseTrajectory3D:
    if poses.size == 0:
        return PoseTrajectory3D(
            positions_xyz=np.zeros((0, 3), dtype=np.float64),
            orientations_quat_wxyz=np.zeros((0, 4), dtype=np.float64),
            timestamps=np.asarray(timestamps, dtype=np.float64),
        )

    quats_wxyz = poses[:, [6, 3, 4, 5]]
    return PoseTrajectory3D(
        positions_xyz=poses[:, :3],
        orientations_quat_wxyz=quats_wxyz,
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )


def save_sparse_map_as_colmap(
    out_dir: str,
    poses: np.ndarray,
    timestamps: np.ndarray,
    points: np.ndarray,
    colors: np.ndarray,
    intrinsics: Iterable[float],
    image_size: Tuple[int, int],
    scale: float = 10.0,
):
    """Write COLMAP text files for a DEVO sparse map."""
    colmap_dir = Path(out_dir)
    colmap_dir.mkdir(parents=True, exist_ok=True)

    traj = _build_trajectory(poses, timestamps)
    inv_poses = list(map(np.linalg.inv, traj.poses_se3))
    traj = PoseTrajectory3D(poses_se3=inv_poses, timestamps=traj.timestamps)

    fx, fy, cx, cy = intrinsics
    H, W = image_size

    images_lines = []
    for idx, (pos, quat) in enumerate(
        zip(traj.positions_xyz * scale, traj.orientations_quat_wxyz), start=1
    ):
        images_lines.append(
            f"{idx} {quat[0]} {quat[1]} {quat[2]} {quat[3]} "
            f"{pos[0]} {pos[1]} {pos[2]} 1\n\n"
        )
    (colmap_dir / "images.txt").write_text("".join(images_lines))

    if points is None or points.size == 0:
        (colmap_dir / "points3D.txt").write_text("")
    else:
        if colors is None or colors.shape[0] != points.shape[0]:
            colors = np.ones((points.shape[0], 3), dtype=np.float32)
        colors_uint = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8).tolist()
        points_lines = []
        for i, (p, c) in enumerate(zip((points * scale).tolist(), colors_uint), start=1):
            points_lines.append(
                f"{i} " + " ".join(map(str, p + c)) + " 0.0 0 0 0 0 0 0\n"
            )
        (colmap_dir / "points3D.txt").write_text("".join(points_lines))

    (colmap_dir / "cameras.txt").write_text(f"1 PINHOLE {W} {H} {fx} {fy} {cx} {cy}")
    print(f"Saved COLMAP-compatible reconstruction in {colmap_dir.resolve()}")
