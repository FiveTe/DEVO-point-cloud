
import os
import torch
import torch.nn.functional as F
from devo.devo import DEVO
from devo.utils import Timer
from pathlib import Path
import datetime
import numpy as np
import yaml
import glob
from itertools import chain
from natsort import natsorted
import copy
import math
import shutil
from scipy.spatial.transform import Rotation as R
from scipy.spatial import cKDTree
from tabulate import tabulate

from devo.plot_utils import plot_trajectory, fig_trajectory
from devo.plot_utils import save_trajectory_tum_format

from utils.viz_utils import show_image, visualize_voxel
from utils.pcd_utils import PointCloudAccumulator

main_ape = None
sync = None
metrics = None
PoseTrajectory3D = None
_EVO_IMPORT_ERROR = None

try:
    import evo.main_ape as main_ape
    from evo.core import sync, metrics
    from evo.core.trajectory import PoseTrajectory3D
except Exception as exc:
    # Keep export/inference utilities usable even when evo's optional ROS deps mismatch.
    _EVO_IMPORT_ERROR = exc

# [DEBUG]
# import matplotlib.pyplot as plt
# plt.switch_backend('Qt5Agg')
# plt.figure()
# plt.grid(False)
# plt.imshow(image.detach().cpu().numpy().transpose(1,2,0))
# plt.show()

def _voxel_edges_qres_uv(
    voxel,
    downsample=4,
    topk=6000,
    border=2,
):
    """Extract edge-like pixels (quarter-res) from an event voxel grid.

    Args:
        voxel (Tensor): (bins, H, W) event voxel grid.
        downsample (int): downsample factor to match DEVO patch coords (default: 4).
        topk (int): number of edge pixels to return (highest gradient magnitude).
        border (int): zero-out a border in quarter-res pixels to avoid artifacts.

    Returns:
        np.ndarray: (N, 2) array of (u, v) pixel coords in quarter-res image space.
    """
    if voxel is None:
        return np.zeros((0, 2), dtype=np.float32)

    # Use absolute event count as an "image" and downsample to DEVO's feature resolution.
    img = voxel.abs().sum(dim=0, keepdim=False)  # (H, W)
    img4 = F.avg_pool2d(img[None, None], kernel_size=downsample, stride=downsample).squeeze(0).squeeze(0)

    # Sobel gradient magnitude on quarter-res.
    dtype = img4.dtype
    device = img4.device
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device, dtype=dtype).view(1, 1, 3, 3)
    gx = F.conv2d(img4[None, None], sobel_x, padding=1).squeeze(0).squeeze(0)
    gy = F.conv2d(img4[None, None], sobel_y, padding=1).squeeze(0).squeeze(0)
    g = torch.sqrt(gx * gx + gy * gy)

    # Suppress borders.
    if border > 0:
        g[:border, :] = 0
        g[-border:, :] = 0
        g[:, :border] = 0
        g[:, -border:] = 0

    flat = g.reshape(-1)
    if flat.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32)

    k = int(min(max(topk, 0), flat.numel()))
    if k == 0:
        return np.zeros((0, 2), dtype=np.float32)

    vals, idx = torch.topk(flat, k, largest=True, sorted=False)
    keep = torch.isfinite(vals) & (vals > 0)
    idx = idx[keep]
    if idx.numel() == 0:
        return np.zeros((0, 2), dtype=np.float32)

    h4, w4 = g.shape
    u = (idx % w4).to(torch.float32)
    v = torch.div(idx, w4, rounding_mode="floor").to(torch.float32)
    uv = torch.stack([u, v], dim=-1).detach().cpu().numpy().astype(np.float32)
    return uv


def _edge_cloud_from_frame_observable(
    frame_obs,
    edge_uv,
    knn=4,
    max_dist=6.0,
):
    """Densify inverse depth from DEVO patch centers onto edge pixels and backproject to world."""
    if frame_obs is None or edge_uv is None or len(edge_uv) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    centers = np.asarray(frame_obs.get("centers", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    inv_depth = np.asarray(frame_obs.get("depths", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    intr = np.asarray(frame_obs.get("intrinsics", np.zeros((4,), dtype=np.float32)), dtype=np.float32)
    pose = np.asarray(frame_obs.get("pose", np.zeros((7,), dtype=np.float32)), dtype=np.float32)

    if centers.ndim != 2 or centers.shape[1] != 2 or inv_depth.ndim != 1:
        return np.zeros((0, 3), dtype=np.float32)

    good = np.isfinite(centers).all(axis=1) & np.isfinite(inv_depth) & (inv_depth > 0)
    centers = centers[good]
    inv_depth = inv_depth[good]
    if centers.shape[0] < max(1, int(knn)):
        return np.zeros((0, 3), dtype=np.float32)

    tree = cKDTree(centers)
    dist, idx = tree.query(edge_uv, k=int(knn), distance_upper_bound=float(max_dist))
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)

    valid = np.isfinite(dist).all(axis=1) & (idx < centers.shape[0]).all(axis=1)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    edge_uv = np.asarray(edge_uv, dtype=np.float32)[valid]
    dist = dist[valid]
    idx = idx[valid]

    w = 1.0 / np.clip(dist, 1e-3, None)
    inv_d = (w * inv_depth[idx]).sum(axis=1) / np.clip(w.sum(axis=1), 1e-6, None)

    fx, fy, cx, cy = intr.astype(np.float32)
    u = edge_uv[:, 0]
    v = edge_uv[:, 1]
    z = 1.0 / np.clip(inv_d, 1e-6, None)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pc = np.stack([x, y, z], axis=1).astype(np.float32)  # camera coords

    t = pose[:3].astype(np.float32)
    q = pose[3:].astype(np.float32)  # xyzw
    Rcw = R.from_quat(q).as_matrix().astype(np.float32)  # world -> cam
    pw = (Rcw.T @ (pc - t).T).T.astype(np.float32)
    return pw


@torch.no_grad()
def run_rgb(
    imagedir,
    cfg,
    network,
    viz=False,
    iterator=None,
    timing=False,
    H=480,
    W=640,
    viz_flow=False,
    return_observables=False,
    return_frame_observables=False,
    return_colors=False,
    **kwargs,
):
    if return_colors and not return_observables:
        raise ValueError("return_colors=True requires return_observables=True")
    record_frame_observables = kwargs.pop("record_frame_observables", False) or return_frame_observables
    slam = DEVO(cfg, network, ht=H, wd=W, viz=viz, viz_flow=viz_flow, record_frame_observables=record_frame_observables, **kwargs)
    
    for i, (image, intrinsics, t) in enumerate(iterator):
        if timing and i == 0:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

        if viz: 
            show_image(image, 1)
        
        with Timer("DPVO", enabled=False):
            slam(t, image, intrinsics)

    for _ in range(12):
        slam.update()

    frame_data = None
    colors = None
    if return_observables and return_frame_observables:
        outputs = slam.terminate(return_observables=True, return_frame_observables=True, include_colors=return_colors)
        if return_colors:
            poses, tstamps, point_cloud, depths, colors, frame_data = outputs
        else:
            poses, tstamps, point_cloud, depths, frame_data = outputs
    elif return_observables:
        outputs = slam.terminate(return_observables=True, include_colors=return_colors)
        if return_colors:
            poses, tstamps, point_cloud, depths, colors = outputs
        else:
            poses, tstamps, point_cloud, depths = outputs
    elif return_frame_observables:
        poses, tstamps, frame_data = slam.terminate(return_frame_observables=True)
        point_cloud = depths = None
    else:
        poses, tstamps = slam.terminate()
        point_cloud = depths = None

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"{imagedir}\nDPVO Network {i+1} frames in {dt} sec, e.g. {(i+1)/dt} FPS")
    
    flowdata = slam.flow_data if viz_flow else None
    if return_observables and return_frame_observables:
        if return_colors:
            return poses, tstamps, flowdata, point_cloud, depths, colors, frame_data
        return poses, tstamps, flowdata, point_cloud, depths, frame_data
    if return_observables:
        if return_colors:
            return poses, tstamps, flowdata, point_cloud, depths, colors
        return poses, tstamps, flowdata, point_cloud, depths
    if return_frame_observables:
        return poses, tstamps, flowdata, frame_data
    return poses, tstamps, flowdata


@torch.no_grad()
def run_voxel_norm_seq(
    voxeldir,
    cfg,
    network,
    viz=False,
    iterator=None,
    timing=False,
    H=480,
    W=640,
    viz_flow=False,
    scale=1.0,
    N_norm=15,
    return_observables=False,
    return_frame_observables=False,
    return_colors=False,
    **kwargs,
):
    if return_colors and not return_observables:
        raise ValueError("return_colors=True requires return_observables=True")
    record_frame_observables = kwargs.pop("record_frame_observables", False) or return_frame_observables
    slam = DEVO(cfg, network, evs=True, ht=H, wd=W, viz=viz, viz_flow=viz_flow, record_frame_observables=record_frame_observables, **kwargs)
    
    voxels = []
    tss = []
    for i, (voxel, intrinsics, t) in enumerate(iterator):
        if i == 0 or i % N_norm != 0:
            voxels.append(voxel)
            tss.append(t)
            continue
        else:
            voxels = [v.unsqueeze(0) for v in voxels]
            voxels = torch.cat(voxels, dim=0)
            n, ch, h, w = voxels.shape

            flatten_image = torch.clone(voxels).view(n,-1)
            pos = flatten_image > 0.0
            neg = flatten_image < 0.0
            vx_max = torch.Tensor([1]).to("cuda") if pos.sum().item() == 0 else flatten_image[pos].max(dim=-1, keepdim=True)[0]
            vx_min = torch.Tensor([1]).to("cuda") if neg.sum().item() == 0 else flatten_image[neg].min(dim=-1, keepdim=True)[0]
            if vx_min.item() == 0.0 or vx_max.item() == 0.0:
                print(f"empty voxel at {t}!")
            flatten_image[pos] = flatten_image[pos] / vx_max
            flatten_image[neg] = flatten_image[neg] / -vx_min
            voxels = flatten_image.view(n,ch,h,w)
            
            for t, vox in zip(tss, voxels):
                slam(t, vox, intrinsics, scale=scale)
            voxels = []
            tss = []

    for _ in range(12):
        slam.update()

    frame_data = None
    colors = None
    if return_observables and return_frame_observables:
        outputs = slam.terminate(return_observables=True, return_frame_observables=True, include_colors=return_colors)
        if return_colors:
            poses, tstamps, point_cloud, depths, colors, frame_data = outputs
        else:
            poses, tstamps, point_cloud, depths, frame_data = outputs
    elif return_observables:
        outputs = slam.terminate(return_observables=True, include_colors=return_colors)
        if return_colors:
            poses, tstamps, point_cloud, depths, colors = outputs
        else:
            poses, tstamps, point_cloud, depths = outputs
    elif return_frame_observables:
        poses, tstamps, frame_data = slam.terminate(return_frame_observables=True)
        point_cloud = depths = None
    else:
        poses, tstamps = slam.terminate()
        point_cloud = depths = None

    flowdata = slam.flow_data if viz_flow else None
    if return_observables and return_frame_observables:
        if return_colors:
            return poses, tstamps, flowdata, point_cloud, depths, colors, frame_data
        return poses, tstamps, flowdata, point_cloud, depths, frame_data
    if return_observables:
        if return_colors:
            return poses, tstamps, flowdata, point_cloud, depths, colors
        return poses, tstamps, flowdata, point_cloud, depths
    if return_frame_observables:
        return poses, tstamps, flowdata, frame_data
    return poses, tstamps, flowdata


@torch.no_grad()
def run_voxel(
    voxeldir,
    cfg,
    network,
    viz=False,
    iterator=None,
    timing=False,
    H=480,
    W=640,
    viz_flow=False,
    scale=1.0,
    return_observables=False,
    return_frame_observables=False,
    return_colors=False,
    save_per_frame_cloud=False,
    save_per_frame_cloud_path="results/clouds",
    voxel_size=0.05,
    debug=False,
    accumulate_map=False,
    full_map_out=None,
    full_map_color_out=None,
    export_edge_cloud=False,
    edge_cloud_out=None,
    edge_downsample=4,
    edge_topk=6000,
    edge_border=2,
    edge_knn=4,
    edge_max_dist=6.0,
    **kwargs,
):
    if return_colors and not return_observables:
        raise ValueError("return_colors=True requires return_observables=True")
    record_frame_observables = (
        kwargs.pop("record_frame_observables", False)
        or return_frame_observables
        or export_edge_cloud
    )
    slam = DEVO(
        cfg,
        network,
        evs=True,
        ht=H,
        wd=W,
        viz=viz,
        viz_flow=viz_flow,
        record_frame_observables=record_frame_observables,
        save_per_frame_cloud=save_per_frame_cloud,
        save_per_frame_cloud_path=save_per_frame_cloud_path,
        debug_viewer=debug,
        accumulate_map=accumulate_map,
        **kwargs,
    )
    
    accumulator = None
    if save_per_frame_cloud:
        accumulator = PointCloudAccumulator(save_per_frame_cloud_path, voxel_size=voxel_size)

    frames_processed = 0
    edge_points_accum = []
    prev_frame_obs_len = 0

    for i, (voxel, intrinsics, t) in enumerate(iterator):
        if timing and i == 0:
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()
        frames_processed += 1

        if debug:
            nonzero = (voxel.abs() > 1e-6).sum().item()
            print(f"[DEBUG][run_voxel] Frame {i} ts={t:.3f}us, voxel shape={tuple(voxel.shape)}, nonzero={nonzero}")

        if viz: 
            # import matplotlib.pyplot as plt
            # plt.switch_backend('Qt5Agg')
            visualize_voxel(voxel.detach().cpu())
        
        with Timer("DEVO", enabled=timing):
            slam(t, voxel, intrinsics, scale=scale)

        # If DEVO recorded a new per-keyframe observable, generate an edge-aligned cloud for it.
        # This mirrors DEVO's own "accumulate_map" idea: keep points even if older keyframes are
        # later removed from the active window.
        if export_edge_cloud and record_frame_observables:
            curr_len = len(getattr(slam, "frame_observables", []))
            if curr_len > prev_frame_obs_len:
                frame_obs = slam.frame_observables[-1]
                edge_uv = _voxel_edges_qres_uv(
                    voxel,
                    downsample=edge_downsample,
                    topk=edge_topk,
                    border=edge_border,
                )
                edge_pw = _edge_cloud_from_frame_observable(
                    frame_obs,
                    edge_uv,
                    knn=edge_knn,
                    max_dist=edge_max_dist,
                )
                if edge_pw.size > 0:
                    edge_points_accum.append(edge_pw)
            prev_frame_obs_len = curr_len

        if debug:
            num_points = None
            if hasattr(slam, "pg") and hasattr(slam.pg, "points_"):
                try:
                    num_points = slam.pg.points_.shape[0]
                except Exception:
                    num_points = None
            num_kf = getattr(getattr(slam, "pg", None), "m", None)
            print(f"[DEBUG][run_voxel] After frame {i}: accepted_points={num_points}, active_keyframes={num_kf}")
        
        if save_per_frame_cloud and accumulator is not None:
             points, _, _ = slam._extract_sparse_map(include_colors=True)
             accumulator.add_points(points)

    for _ in range(12):
        slam.update()

    if save_per_frame_cloud and accumulator is not None:
        accumulator.save_combined_map()
        accumulator.cleanup_individual_files()

    frame_data = None
    colors = None
    if return_observables and return_frame_observables:
        outputs = slam.terminate(return_observables=True, return_frame_observables=True, include_colors=return_colors)
        if return_colors:
            poses, tstamps, point_cloud, depths, colors, frame_data = outputs
        else:
            poses, tstamps, point_cloud, depths, frame_data = outputs
    elif return_observables:
        outputs = slam.terminate(return_observables=True, include_colors=return_colors)
        if return_colors:
            poses, tstamps, point_cloud, depths, colors = outputs
        else:
            poses, tstamps, point_cloud, depths = outputs
    elif return_frame_observables:
        poses, tstamps, frame_data = slam.terminate(return_frame_observables=True)
        point_cloud = depths = None
    else:
        poses, tstamps = slam.terminate()
        point_cloud = depths = None

    if timing:
        t1.record()
        torch.cuda.synchronize()
        dt = t0.elapsed_time(t1)/1e3
        print(f"{voxeldir}\nDEVO Network {i+1} frames in {dt} sec, e.g. {(i+1)/dt} FPS")

    if accumulate_map and full_map_out is not None:
        include_colors = full_map_color_out is not None
        slam.save_accumulated_map(full_map_out, full_map_color_out if include_colors else None)

    if export_edge_cloud and edge_cloud_out is not None:
        if edge_points_accum:
            edge_points = np.concatenate(edge_points_accum, axis=0)
        else:
            edge_points = np.zeros((0, 3), dtype=np.float32)
        np.save(edge_cloud_out, edge_points)

    if debug:
        pc_shape = None if 'point_cloud' not in locals() or point_cloud is None else point_cloud.shape
        depth_shape = None if 'depths' not in locals() or depths is None else depths.shape
        print(f"[DEBUG][run_voxel] Completed processing {frames_processed} frames. "
              f"Point cloud shape: {pc_shape}, Depth shape: {depth_shape}, "
              f"Poses shape: {poses.shape if 'poses' in locals() else None}")
    
    flowdata = slam.flow_data if viz_flow else None
    if return_observables and return_frame_observables:
        if return_colors:
            return poses, tstamps, flowdata, point_cloud, depths, colors, frame_data
        return poses, tstamps, flowdata, point_cloud, depths, frame_data
    if return_observables:
        if return_colors:
            return poses, tstamps, flowdata, point_cloud, depths, colors
        return poses, tstamps, flowdata, point_cloud, depths
    if return_frame_observables:
        return poses, tstamps, flowdata, frame_data
    return poses, tstamps, flowdata


def assert_eval_config(args):
    assert os.path.isfile(args.weights) and (".pth" in args.weights or ".pt" in args.weights)
    assert os.path.isfile(args.val_split)
    assert args.trials > 0


def _require_evo():
    if main_ape is None or sync is None or metrics is None or PoseTrajectory3D is None:
        raise ImportError(
            "evo imports failed. Install compatible evo/rosbags versions for evaluation features."
        ) from _EVO_IMPORT_ERROR

def ate(traj_ref, traj_est, timestamps):
    _require_evo()
    import evo
    import evo.main_ape as main_ape
    from evo.core.trajectory import PoseTrajectory3D
    from evo.core.metrics import PoseRelation

    traj_est = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:], # TODO wrong format: EVO uses wxyz, we use xyzw
        timestamps=timestamps)

    traj_ref = PoseTrajectory3D(
        positions_xyz=traj_ref[:,:3],
        orientations_quat_wxyz=traj_ref[:,3:],  # TODO wrong format: EVO uses wxyz, we use xyzw
        timestamps=timestamps)
    
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    return result.stats["rmse"]

def get_alg(n):
    if n == "eds" or n == "tumvie" or n == "tartanair":
        return "rgb"
    elif n == "eds_evs" or n == "tumvie_evs" or n == "tartanair_evs":
        return "evs"
    elif n == "eds_evs_viz" or n == "tumvie_evs_viz" or n == "tartanair_evs_viz":
        return "evs_viz"

def make_outfolder(outdir, dataset_name, expname, scene_name, trial, train_step, stride, calib1_eds, camID_tumvie):
    date = datetime.datetime.today().strftime('%Y-%m-%d') # TODO improve output folder
    outfolder = os.path.join(f"{outdir}/{dataset_name}/{date}_{expname}/{scene_name}_trial_{trial}_step_{train_step}")
    if stride != 1:
        outfolder = outfolder + f"_stride_{stride}"
    if calib1_eds != None:
        outfolder = outfolder + f"_calib1" if calib1_eds else outfolder + f"_calib0"
    if camID_tumvie != None:
        outfolder = outfolder + f"_camID_{camID_tumvie}"
    outfolder = os.path.abspath(outfolder)
    os.makedirs(outfolder, exist_ok=True)
    return outfolder

def run_rpg_eval(outfolder, traj_ref, tss_ref_us, traj_est, tstamps):
    p = f"{outfolder}/"
    p = os.path.abspath(p)
    os.makedirs(p, exist_ok=True)

    fnameGT = os.path.join(p, "stamped_groundtruth.txt")
    f = open(fnameGT, "w")
    f.write("# timestamp[secs] tx ty tz qx qy qz qw\n")
    for i in range(len(traj_ref)):
        f.write(f"{tss_ref_us[i]/1e6} {traj_ref[i,0]} {traj_ref[i,1]} {traj_ref[i,2]} {traj_ref[i,3]} {traj_ref[i,4]} {traj_ref[i,5]} {traj_ref[i,6]}\n")
    f.close()

    fnameEst = os.path.join(p, "stamped_traj_estimate.txt")
    f = open(fnameEst, "w")
    f.write("# timestamp[secs] tx ty tz qx qy qz qw\n")
    for i in range(len(traj_est)):
        f.write(f"{tstamps[i]/1e6} {traj_est[i,0]} {traj_est[i,1]} {traj_est[i,2]} {traj_est[i,3]} {traj_est[i,4]} {traj_est[i,5]} {traj_est[i,6]}\n")
    f.close()
    
    # cmd = f"python thirdparty/rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py --result_dir {p} --recalculate_errors --png --plot"
    cmd = f"python thirdparty/rpg_trajectory_evaluation/scripts/analyze_trajectory_single.py {p} --recalculate_errors --png --plot"
    os.system(cmd)

    return fnameGT, fnameEst

def load_stats_rpg_results(outfolder):
    rpg_fspath = os.path.join(outfolder, "saved_results/traj_est")

    absfile = natsorted(glob.glob(os.path.join(rpg_fspath, "absolute_err_stat*.yaml")))[-1]
    with open(absfile, 'r') as file:
        abs_stats = yaml.safe_load(file)

    last_relfile = natsorted(glob.glob(os.path.join(rpg_fspath, "relative_error_statistics_*.yaml")))[-1]
    with open(last_relfile, 'r') as file:
        rel_stats = yaml.safe_load(file)

    # last_relfile_time = natsorted(glob.glob(os.path.join(rpg_fspath, "Time_relative_error_statistics_*.yaml")))[-1]
    # with open(last_relfile_time, 'r') as file:
    #     rel_stats_time = yaml.safe_load(file)
    rel_stats_time = copy.deepcopy(rel_stats) 
    
    return abs_stats, rel_stats, rel_stats_time

def remove_all_patterns_from_str(s, patterns):
    for pattern in patterns:
        if pattern in s:
            s = s.replace(pattern, "")
    return s

def remove_row_from_table(table_string, row_index):
    rows = table_string.split('\n')
    if row_index < len(rows):
        del rows[row_index]
    return '\n'.join(rows)

def dict_to_table(data, scene, header=True):
    table_data = [["Scene", *data.keys()], [f"{scene}", *data.values()]]
    table_data = [row + ["\\\\"] for row in table_data]

    table = tabulate(table_data, tablefmt="plain")

    if not header:
        table = remove_row_from_table(table, 0)

    return table

def write_res_table(outfolder, res_str, scene_name, trial):
    res = res_str.split("|")
    res_dict = {}
    for r in res:
        k = r.split(":")[0]
        patterns_to_remove = ["\n", " ", ")", "("]
        k = remove_all_patterns_from_str(k, patterns_to_remove)

        v = r.split(":")[1]
        v = remove_all_patterns_from_str(v, patterns_to_remove)
        res_dict[k] = float(v)

    summtable_fnmae = os.path.join(outfolder, "../0_res.txt")
    if not os.path.isfile(summtable_fnmae): 
        f = open(summtable_fnmae, "w")
    else:
        f = open(summtable_fnmae, "a")
    if trial == 0:
        f.write("\n")

    table = dict_to_table(res_dict, scene_name, trial==0)
    f.write(table)
    f.write("\n")
    f.close()


def ate_real(traj_ref, tss_ref_us, traj_est, tstamps):
    _require_evo()
    evoGT = PoseTrajectory3D(
        positions_xyz=traj_ref[:,:3],
        orientations_quat_wxyz=traj_ref[:,3:], # TODO wrong format: EVO uses wxyz, we use xyzw
        timestamps=tss_ref_us/1e6)

    evoEst = PoseTrajectory3D(
        positions_xyz=traj_est[:,:3],
        orientations_quat_wxyz=traj_est[:,3:], # TODO wrong format: EVO uses wxyz, we use xyzw
        timestamps=tstamps/1e6)

    if traj_ref.shape == traj_est.shape:
        assert np.all(tss_ref_us == tstamps)
        return ate(traj_ref, traj_est, tstamps)*100, evoGT, evoEst
    
    evoGT, evoEst = sync.associate_trajectories(evoGT, evoEst, max_diff=1)
    ape_trans = main_ape.ape(evoGT, evoEst, pose_relation=metrics.PoseRelation.translation_part, align=True, correct_scale=True)
    evoATE = ape_trans.stats["rmse"]*100
    return evoATE, evoGT, evoEst


def make_evo_traj(poses_N_x_7, tss_us):
    _require_evo()
    assert poses_N_x_7.shape[1] == 7
    assert poses_N_x_7.shape[0] > 10
    assert tss_us.shape[0] == poses_N_x_7.shape[0]

    traj_evo = PoseTrajectory3D(
        positions_xyz=poses_N_x_7[:,:3],
        orientations_quat_wxyz=poses_N_x_7[:,3:],
        timestamps=tss_us/1e6)
    return traj_evo


@torch.no_grad()            
def log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                plot=False, save=True, return_figure=False, rpg_eval=True, stride=1, 
                calib1_eds=None, camID_tumvie=None, outdir=None, expname="", max_diff_sec=0.01):
    # results: dict of (scene, list of results)
    # all_results: list of all raw_results

    # unpack data
    traj_GT, tss_GT_us, traj_est, tss_est_us = data
    train_step, net, dataset_name, scene, trial, cfg, args = hyperparam

    # create folders
    if train_step is None:
        if isinstance(net, str) and ".pth" in net:
            train_step = os.path.basename(net.split(".")[0])
        else:
            train_step = -1
    scene_name = '_'.join(scene.split('/')[1:]).title() if "/P0" in scene else scene.title()
    if outdir is None:
        outdir = "results"
    outfolder = make_outfolder(outdir, dataset_name, expname, scene_name, trial, train_step, stride, calib1_eds, camID_tumvie)

    # save cfg & args to outfolder
    if cfg is not None:
        with open(f"{outfolder}/cfg.yaml", 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
    if args is not None:
        if args is not None:
            with open(f"{outfolder}/args.yaml", 'w') as f:
                yaml.dump(vars(args), f, default_flow_style=False)

    # compute ATE
    ate_score, evoGT, evoEst = ate_real(traj_GT, tss_GT_us, traj_est, tss_est_us)
    all_results.append(ate_score)
    results_dict_scene[scene].append(ate_score)
    
    # following https://github.com/arclab-hku/Event_based_VO-VIO-SLAM/issues/5
    evoGT = make_evo_traj(traj_GT, tss_GT_us)
    evoEst = make_evo_traj(traj_est, tss_est_us)
    gtlentraj = evoGT.get_infos()["path length (m)"]
    evoGT, evoEst = sync.associate_trajectories(evoGT, evoEst, max_diff=1)
    ape_trans = main_ape.ape(copy.deepcopy(evoGT), copy.deepcopy(evoEst), pose_relation=metrics.PoseRelation.translation_part, align=True, correct_scale=True)
    MPE = ape_trans.stats["mean"] / gtlentraj * 100
    evoATE = ape_trans.stats["rmse"]*100
    assert abs(evoATE-ate_score) < 1e-5
    R_rmse_deg = -1.0

    if save:
        Path(f"{outfolder}").mkdir(exist_ok=True)
        save_trajectory_tum_format((traj_est, tss_est_us), f"{outfolder}/{scene_name}_Trial{trial+1:02d}.txt")

    if rpg_eval:

        fnamegt, fnameest = run_rpg_eval(outfolder, traj_GT, tss_GT_us, traj_est, tss_est_us)
        abs_stats, rel_stats, _ = load_stats_rpg_results(outfolder)

        # abs errs
        ate_rpg = abs_stats["trans"]["rmse"]*100
        print(f"ate_rpg: {ate_rpg:.04f}, ate_real (EVO): {ate_score:.04f}")
        # assert abs(ate_rpg-ate_score)/ate_rpg < 0.1 # 10%
        R_rmse_deg = abs_stats["rot"]["rmse"]
        MTE_m = abs_stats["trans"]["mean"]

        # traj_GT_inter = interpolate_traj_at_tss(traj_GT, tss_GT_us, tss_est_us)
        # ate_inter, _, _ = ate_real(traj_GT_inter, tss_est_us, traj_est, tss_est_us)
        
        res_str = f"\nATE[cm]: {ate_score:.03f} | R_rmse[deg]: {R_rmse_deg:.03f} | MPE[%/m]: {MPE:.03f} \n"
        # res_str += f"MTE[m]: {MTE_m:.03f} | (ATE_int[cm]: {ate_inter:.02f} | ATE_rpg[cm]: {ate_rpg:.02f}) \n"

        write_res_table(outfolder, res_str, scene_name, trial)
    else:
        res_str = f"\nATE[cm]: {ate_score:.03f} | MPE[%/m]: {MPE:.03f}"

    if plot:
        Path(f"{outfolder}/").mkdir(exist_ok=True)
        pdfname = f"{outfolder}/../{scene_name}_Trial{trial+1:02d}_exp_{expname}_step_{train_step}_stride_{stride}.pdf"
        plot_trajectory((traj_est, tss_est_us/1e6), (traj_GT, tss_GT_us/1e6), 
                        f"{dataset_name} {expname} {scene_name.replace('_', ' ')} Trial #{trial+1} {res_str}",
                        pdfname, align=True, correct_scale=True, max_diff_sec=max_diff_sec)
        shutil.copy(pdfname, f"{outfolder}/{scene_name}_Trial{trial+1:02d}_step_{train_step}_stride_{stride}.pdf")

        # [DEBUG]
        pdfname = f"{outfolder}/GT_{scene_name}_Trial{trial+1:02d}_exp_{expname}_step_{train_step}_stride_{stride}.pdf"
        plot_trajectory((traj_GT, tss_GT_us/1e6), (traj_GT, tss_GT_us/1e6), 
                        f"{dataset_name} {expname} {scene_name.replace('_', ' ')} Trial #{trial+1} {res_str}",
                        pdfname, align=True, correct_scale=True, max_diff_sec=max_diff_sec)

    if return_figure:
        fig = fig_trajectory((traj_est, tss_est_us/1e6), (traj_GT, tss_GT_us/1e6), f"{dataset_name} {scene_name.replace('_', ' ')} {res_str})",
                            return_figure=True, max_diff_sec=max_diff_sec)
        figures[f"{dataset_name}_{scene_name}"] = fig

    return all_results, results_dict_scene, figures, outfolder



@torch.no_grad()
def write_raw_results(all_results, outfolder):
    # all_results: list of all raw_results
    os.makedirs(os.path.join(f"{outfolder}/../raw_results"), exist_ok=True)
    with open(os.path.join(f"{outfolder}/../raw_results", datetime.datetime.now().strftime('%m-%d-%I%p.txt')), "w") as f:
        f.write(','.join([str(x) for x in all_results]))

@torch.no_grad()
def compute_median_results(results, all_results, dataset_name, outfolder=None):
    # results: dict of (scene, list of results)
    # all_results: list of all raw_results
        
    results_dict = dict([(f"{dataset_name}/{k}", np.median(v)) for (k, v) in results.items()])
    results_dict["AUC"] = np.maximum(1 - np.array(all_results), 0).mean()

    xs = []
    for scene in results:
        x = np.median(results[scene])
        xs.append(x)
    results_dict["AVG"] = np.mean(xs) / 100.0 # cm -> m

    if outfolder is not None:
        with open(os.path.join(f"{outfolder}/../results_dict_latex_{datetime.datetime.now().strftime('%m-%d-%I%p.txt')}"), 'w') as f:
            k0 = list(results.keys())[0]
            num_runs = len(results[k0])
            f.write(' & '.join([str(k) for k in results.keys()]))
            f.write('\n')

 
            for i in range(num_runs):
                print(f"{[str(v[i]) for v in results.values()]}")
                f.write(' & '.join([str(v[i]) for v in results.values()]))
                f.write('\n')

            f.write(f"Medians\n")
            for i in range(num_runs):
                print(f"{[str(v[i]) for v in results.values()]}")
                f.write(' & '.join([str(np.median(v)) for v in results.values()]))
                f.write('\n')

            f.write('\n\n')

    return results_dict
