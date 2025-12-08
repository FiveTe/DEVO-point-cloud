import argparse
from pathlib import Path

import numpy as np


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Render a 3D PNG of a DEVO point cloud.")
    parser.add_argument(
        "--pointcloud",
        default="results/corridor_pointcloud_0_40s.npy",
        help="Path to the .npy point cloud file.",
    )
    parser.add_argument(
        "--color-by",
        choices=["x", "y", "z", "none"],
        default="z",
        help="Coordinate used to colorize the scatter plot (or 'none').",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Stride for subsampling points before plotting.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Marker size passed to Matplotlib's scatter3D.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output PNG path (defaults to results/<file>_3d.png).",
    )
    parser.add_argument(
        "--elev",
        type=float,
        default=15.0,
        help="Elevation angle in degrees for the camera.",
    )
    parser.add_argument(
        "--azim",
        type=float,
        default=-60.0,
        help="Azimuth angle in degrees for the camera.",
    )
    parser.add_argument(
        "--tight",
        action="store_true",
        help="Apply tight axis limits based on the data range.",
    )
    return parser


def ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  Needed to register projection
        return plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib (with its 3D toolkit) is required for 3D PNG output.\n"
            "Install it via `pip install matplotlib` inside your environment and retry."
        ) from exc


def load_pointcloud(path):
    pc_path = Path(path)
    if not pc_path.is_file():
        raise FileNotFoundError(f"Unable to locate point cloud '{pc_path}'.")

    point_cloud = np.load(pc_path)
    if point_cloud.ndim != 2 or point_cloud.shape[1] < 3:
        raise ValueError(f"Expected (N, 3) array but got shape {point_cloud.shape}.")
    return point_cloud, pc_path


def downsample_cloud(point_cloud, stride):
    if stride <= 1:
        return point_cloud
    return point_cloud[::stride]


def resolve_output_path(pc_path, user_output):
    if user_output:
        out_path = Path(user_output)
        if out_path.suffix.lower() != ".png":
            out_path = out_path.with_suffix(".png")
    else:
        out_path = Path("results") / f"{pc_path.stem}_3d.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def choose_colors(point_cloud, axis_key):
    if axis_key == "none":
        return "k", None
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis_key]
    values = point_cloud[:, axis_idx]
    return values, f"viridis"


def set_limits(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    spreads = maxs - mins
    max_span = np.max(spreads)
    center = (mins + maxs) / 2.0
    half = max_span / 2.0 if max_span > 0 else 1.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)


def plot_pointcloud(args):
    plt = ensure_matplotlib()
    point_cloud, pc_path = load_pointcloud(args.pointcloud)
    point_cloud = downsample_cloud(point_cloud, args.downsample)

    if point_cloud.size == 0:
        raise ValueError("Point cloud is empty after downsampling.")

    colors, cmap = choose_colors(point_cloud, args.color_by)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        point_cloud[:, 0],
        point_cloud[:, 1],
        point_cloud[:, 2],
        c=colors,
        cmap=cmap,
        s=args.point_size,
        depthshade=False,
        linewidths=0,
    )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{pc_path.stem} - 3D point cloud")
    ax.view_init(elev=args.elev, azim=args.azim)

    if args.tight:
        set_limits(ax, point_cloud)

    if cmap:
        cbar = fig.colorbar(scatter, shrink=0.7, pad=0.05)
        cbar.set_label(f"{args.color_by.upper()} value (m)")

    output_path = resolve_output_path(pc_path, args.output)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D PNG to {output_path}")


if __name__ == "__main__":
    parser = build_arg_parser()
    plot_pointcloud(parser.parse_args())
