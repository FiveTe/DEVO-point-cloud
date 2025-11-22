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
args = parser.parse_args()

cfg.merge_from_file(args.config)

poses, tstamps, flow, point_cloud, depths = run_voxel(
    voxeldir=args.datapath,
    cfg=cfg,
    network=args.weights,
    iterator=fpv_evs_iterator(args.datapath),
    H=260,
    W=346,
    viz=args.viz,
    viz_flow=args.viz_flow,
    return_observables=True,
    save_per_frame_cloud=args.save_per_frame_cloud,
    save_per_frame_cloud_path=args.save_per_frame_cloud_path,
)

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
