import json
import os
import numpy as np
import torch
from devo.config import cfg

from utils.load_utils import load_gt_us, vector_evs_iterator
from utils.eval_utils import assert_eval_config, run_voxel
from utils.eval_utils import log_results, write_raw_results, compute_median_results
from utils.viz_utils import viz_flow_inference
from utils.pcd_utils import save_point_clouds_to_ply

H, W = 480, 640


def load_vector_intrinsics(scene_path, side):
    intr_path = os.path.join(scene_path, f"calib_undist_evs_{side}.txt")
    if not os.path.exists(intr_path):
        print(f"Warning: could not find intrinsics at {intr_path}")
        return None, intr_path
    try:
        intrinsics = np.loadtxt(intr_path)
    except Exception as exc:
        print(f"Warning: failed to read intrinsics from {intr_path}: {exc}")
        intrinsics = None
    return intrinsics, intr_path


def log_camera_metadata(save_dir, dataset_name, scene, trial_idx, side, poses, timestamps, intrinsics, intr_path, start_us=None, stop_us=None):
    os.makedirs(save_dir, exist_ok=True)
    safe_scene = scene.replace("/", "_")
    base_name = f"{dataset_name}_{safe_scene}_trial{trial_idx+1:02d}"
    json_path = os.path.join(save_dir, f"{base_name}_camera_info.json")
    npz_path = os.path.join(save_dir, f"{base_name}_camera_info.npz")

    start_pose = None
    start_timestamp = None
    if poses is not None and len(poses) > 0:
        start_pose = np.asarray(poses)[0].tolist()
    if timestamps is not None and len(timestamps) > 0:
        start_timestamp = float(np.asarray(timestamps)[0])

    metadata = {
        "dataset": dataset_name,
        "scene": scene,
        "trial": trial_idx + 1,
        "side": side,
        "start_timestamp_us": start_timestamp,
        "start_pose_xyz_quat_xyzw": start_pose,
        "intrinsics_fx_fy_cx_cy": intrinsics.tolist() if intrinsics is not None else None,
        "intrinsics_path": intr_path,
        "t_start_filter_us": start_us,
        "t_stop_filter_us": stop_us,
        "poses_file": os.path.basename(npz_path),
    }

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    np.savez(
        npz_path,
        poses=np.asarray(poses) if poses is not None else np.zeros((0, 7), dtype=np.float32),
        timestamps=np.asarray(timestamps) if timestamps is not None else np.zeros((0,), dtype=np.float64),
        intrinsics=np.asarray(intrinsics) if intrinsics is not None else np.zeros((0,), dtype=np.float32),
    )
    print(f"Saved camera metadata to {json_path}")


@torch.no_grad()
def evaluate(config, args, net, train_step=None, datapath="", split_file=None, 
             trials=1, stride=1, plot=False, save=False, return_figure=False, viz=False, timing=False, side='left', viz_flow=False, dT_ms=None,
             save_per_frame_cloud=False, save_per_frame_cloud_path="results/clouds", start_us=None, stop_us=None, voxel_size=0.05):
    dataset_name = "vector_evs"
    assert side == "left" or side == "right"

    if config is None:
        config = cfg
        config.merge_from_file("config/default.yaml")
        
    scenes = open(split_file).read().split()

    results_dict_scene, figures = {}, {}
    all_results = []
    for i, scene in enumerate(scenes):
        print(f"Eval on {scene}")
        results_dict_scene[scene] = []

        max_diff_sec = 0.1 if "units_scooter" in scene else 0.01
        if dT_ms is not None:
            max_diff_sec *= (dT_ms / 33)
        datapath_val = os.path.join(datapath, scene)
        for trial in range(trials):
            intrinsics = None
            intr_path = None
            if save_per_frame_cloud:
                intrinsics, intr_path = load_vector_intrinsics(datapath_val, side)
            # run the slam system
            traj_est, tstamps, flowdata = run_voxel(datapath_val, config, net, viz=viz, 
                                          iterator=vector_evs_iterator(datapath_val, side, stride=stride, dT_ms=dT_ms, timing=timing, H=H, W=W,
                                                                       t_start_us=start_us, t_stop_us=stop_us),
                                          timing=timing, H=H, W=W, viz_flow=viz_flow,
                                          save_per_frame_cloud=save_per_frame_cloud, save_per_frame_cloud_path=save_per_frame_cloud_path, voxel_size=voxel_size)

            if save_per_frame_cloud:
                log_camera_metadata(
                    save_per_frame_cloud_path,
                    dataset_name,
                    scene,
                    trial,
                    side,
                    traj_est,
                    tstamps,
                    intrinsics,
                    intr_path,
                    start_us=start_us,
                    stop_us=stop_us,
                )

            # load  traj
            tss_traj_us, traj_hf = load_gt_us(os.path.join(datapath_val, f"poses_evs_{side}.txt"))

            # do evaluation 
            data = (traj_hf, tss_traj_us, traj_est, tstamps)
            hyperparam = (train_step, net, dataset_name, scene, trial, cfg, args)
            all_results, results_dict_scene, figures, outfolder = log_results(data, hyperparam, all_results, results_dict_scene, figures, 
                                                                   plot=plot, save=save, return_figure=return_figure, stride=stride,
                                                                   expname=args.expname, max_diff_sec=max_diff_sec)
            
            if viz_flow:
                viz_flow_inference(outfolder, flowdata)
            
        print(scene, sorted(results_dict_scene[scene]))

    # write output to file with timestamp
    write_raw_results(all_results, outfolder)
    results_dict = compute_median_results(results_dict_scene, all_results, dataset_name)
        
    if return_figure:
        return results_dict, figures
    return results_dict, None


if __name__ == '__main__': 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/eval_vector.yaml")
    parser.add_argument('--datapath', default='', help='path to dataset directory')
    parser.add_argument('--weights', default="DEVO.pth")
    parser.add_argument('--val_split', type=str, default="splits/vector/vector_val.txt")
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--save_trajectory', action="store_true")
    parser.add_argument('--return_figs', action="store_true")
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--timing', action="store_true")
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--side', type=str, default="left")
    parser.add_argument('--viz_flow', action="store_true")
    parser.add_argument('--expname', type=str, default="")
    parser.add_argument('--save_per_frame_cloud', action="store_true", help="Save point cloud for each frame")
    parser.add_argument('--save_per_frame_cloud_path', type=str, default="results/clouds", help="Path to save per-frame point clouds")
    parser.add_argument('--voxel_size', type=float, default=0.05, help="Voxel size for point cloud downsampling")
    parser.add_argument('--start_us', type=float, default=None, help="Restrict evaluation to timestamps >= start_us (microseconds)")
    parser.add_argument('--stop_us', type=float, default=None, help="Restrict evaluation to timestamps <= stop_us (microseconds)")

    args = parser.parse_args()
    assert_eval_config(args)

    cfg.merge_from_file(args.config)
    print("Running eval_vector_evs.py with config...")
    print(cfg) 

    torch.manual_seed(1234)

    args.save_trajectory = True
    args.plot = True
    val_results, val_figures = evaluate(cfg, args, args.weights, datapath=args.datapath, split_file=args.val_split, trials=args.trials, \
                        plot=args.plot, save=args.save_trajectory, return_figure=args.return_figs, viz=args.viz, timing=args.timing, \
                        stride=args.stride, side=args.side, viz_flow=args.viz_flow,
                        save_per_frame_cloud=args.save_per_frame_cloud, save_per_frame_cloud_path=args.save_per_frame_cloud_path,
                        start_us=args.start_us, stop_us=args.stop_us, voxel_size=args.voxel_size)
    
    print("val_results= \n")
    for k in val_results:
        print(k, val_results[k])
