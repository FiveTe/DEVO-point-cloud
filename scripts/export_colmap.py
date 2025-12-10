import argparse
import numpy as np
from devo.config import cfg

from utils.eval_utils import run_rgb, run_voxel
from utils.load_utils import video_iterator, voxel_iterator
from utils.colmap_utils import save_sparse_map_as_colmap


def _parse_intrinsics(args) -> np.ndarray:
    if args.intrinsics_file:
        values = np.loadtxt(args.intrinsics_file).reshape(-1)
        if values.size < 4:
            raise ValueError(f"Expected 4 values in {args.intrinsics_file}, got {values.size}")
        return values[:4].astype(np.float32)

    required = [args.fx, args.fy, args.cx, args.cy]
    if any(v is None for v in required):
        raise ValueError("Provide either --intrinsics-file or all of --fx/--fy/--cx/--cy.")
    return np.asarray(required, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Export DEVO outputs to COLMAP text files.")
    parser.add_argument("--config", default="config/default_rgb.yaml")
    parser.add_argument("--weights", default="DEVO.pth")
    parser.add_argument("--input", required=True, help="Directory with images or voxel grids.")
    parser.add_argument("--mode", choices=["rgb", "evs"], default="evs")
    parser.add_argument("--timestamps", default=None, help="Optional timestamps file (txt).")
    parser.add_argument("--intrinsics-file", default=None, help="Text file containing fx fy cx cy.")
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0, help="Image/voxel scale factor.")
    parser.add_argument("--viz", action="store_true", help="Enable live visualization.")
    parser.add_argument("--viz-flow", action="store_true", help="Store flow debug data.")
    parser.add_argument("--colmap-dir", default="colmap_export", help="Destination directory.")
    parser.add_argument("--colmap-scale", type=float, default=10.0, help="Visualization scale applied to poses/points.")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    intrinsics = _parse_intrinsics(args).tolist()

    if args.mode == "rgb":
        iterator = video_iterator(
            args.input,
            tss_file=args.timestamps,
            intrinsics=intrinsics,
            stride=args.stride,
            scale=args.scale,
        )
        results = run_rgb(
            imagedir=args.input,
            cfg=cfg,
            network=args.weights,
            iterator=iterator,
            H=args.height,
            W=args.width,
            viz=args.viz,
            viz_flow=args.viz_flow,
            return_observables=True,
            return_colors=True,
        )
    else:
        iterator = voxel_iterator(
            args.input,
            tss_file=args.timestamps,
            intrinsics=intrinsics,
            stride=args.stride,
            scale=args.scale,
        )
        results = run_voxel(
            voxeldir=args.input,
            cfg=cfg,
            network=args.weights,
            iterator=iterator,
            H=args.height,
            W=args.width,
            viz=args.viz,
            viz_flow=args.viz_flow,
            scale=args.scale,
            return_observables=True,
            return_colors=True,
        )

    poses, tstamps, _, point_cloud, _, colors = results
    save_sparse_map_as_colmap(
        args.colmap_dir,
        poses,
        tstamps,
        point_cloud,
        colors,
        intrinsics,
        (args.height, args.width),
        scale=args.colmap_scale,
    )


if __name__ == "__main__":
    main()
