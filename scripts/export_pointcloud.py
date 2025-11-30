import argparse, os
import numpy as np
from devo.config import cfg
from utils.eval_utils import run_voxel
from utils.load_utils import fpv_evs_iterator

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="config/eval_fpv.yaml")
parser.add_argument("--datapath", required=True)
parser.add_argument("--weights", default="DEVO.pth")
parser.add_argument("--out", default="point_cloud.npy")
parser.add_argument("--viz", action="store_true", help="Enable live DPViewer visualization")
parser.add_argument("--viz-flow", action="store_true", help="Enable flow computation/output")
parser.add_argument('--save_per_frame_cloud', action="store_true", help="Save point cloud for each frame")
parser.add_argument('--save_per_frame_cloud_path', type=str, default="results/clouds", help="Path to save per-frame point clouds")
parser.add_argument('--export_frame_data', action="store_true", help="Export per-frame sparse point clouds, depths, and metadata to disk")
parser.add_argument('--frame_data_out', type=str, default=None, help="Path for the per-frame .npz bundle (defaults to <out>_frames.npz)")
args = parser.parse_args()

cfg.merge_from_file(args.config)

results = run_voxel(
    voxeldir=args.datapath,
    cfg=cfg,
    network=args.weights,
    iterator=fpv_evs_iterator(args.datapath),
    H=260,
    W=346,
    viz=args.viz,
    viz_flow=args.viz_flow,
    return_observables=True,
    return_frame_observables=args.export_frame_data,
    save_per_frame_cloud=args.save_per_frame_cloud,
    save_per_frame_cloud_path=args.save_per_frame_cloud_path,
)

if args.export_frame_data:
    poses, tstamps, flow, point_cloud, depths, frame_data = results
else:
    poses, tstamps, flow, point_cloud, depths = results

base, ext = os.path.splitext(args.out)
if ext == "":
    ext = ".npy"
    pointcloud_path = base + ext
else:
    pointcloud_path = args.out

np.save(pointcloud_path, point_cloud)
np.save(base + "_depths" + ext, depths)
np.save(base + "_poses" + ext, poses)
np.save(base + "_tstamps" + ext, tstamps)

if flow is not None:
    np.save(base + "_flow" + ext, flow, allow_pickle=True)

if args.export_frame_data:
    frame_ids = np.array([f["frame_id"] for f in frame_data], dtype=np.int64)
    timestamps = np.array([f["timestamp"] for f in frame_data], dtype=np.float64)
    poses_per_frame = np.stack([f["pose"] for f in frame_data], axis=0) if frame_data else np.zeros((0, 7), dtype=np.float32)
    intrinsics_per_frame = np.stack([f["intrinsics"] for f in frame_data], axis=0) if frame_data else np.zeros((0, 4), dtype=np.float32)
    counts = np.array([len(f["points"]) for f in frame_data], dtype=np.int32)

    if counts.size > 0 and counts.sum() > 0:
        points_concat = np.concatenate([f["points"] for f in frame_data], axis=0)
        depths_concat = np.concatenate([f["depths"] for f in frame_data], axis=0)
    else:
        points_concat = np.zeros((0, 3), dtype=np.float32)
        depths_concat = np.zeros((0,), dtype=np.float32)

    offsets = np.zeros_like(counts, dtype=np.int64)
    if counts.size > 0:
        offsets[1:] = np.cumsum(counts[:-1], dtype=np.int64)

    frame_out_path = args.frame_data_out or base + "_frames.npz"
    np.savez(
        frame_out_path,
        frame_ids=frame_ids,
        timestamps=timestamps,
        poses=poses_per_frame,
        intrinsics=intrinsics_per_frame,
        counts=counts,
        offsets=offsets,
        points=points_concat,
        depths=depths_concat,
    )
