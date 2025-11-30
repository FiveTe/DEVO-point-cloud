# Frame-by-Frame Point Cloud Export

This feature allows you to export the sparse point cloud reconstruction for every frame processed by DEVO. The point clouds are saved as `.ply` files and, optionally, as a compact `.npz` bundle that also stores timestamps, poses, depths, and intrinsics for downstream TSDF fusion.

## Usage

You can enable this feature by passing the `--save_per_frame_cloud` flag to the evaluation scripts. You can also specify the output directory using `--save_per_frame_cloud_path`.

### Example: Evaluating on RPG Dataset

To run evaluation on the RPG dataset and export point clouds:

```bash
python evals/eval_evs/eval_rpg_evs.py \
    --datapath <PATH_TO_RPG_DATASET> \
    --weights DEVO.pth \
    --save_per_frame_cloud \
    --save_per_frame_cloud_path results/rpg_clouds
```

This will generate files like `results/rpg_clouds/cloud_00001.ply`, `results/rpg_clouds/cloud_00002.ply`, etc.

### Example: Evaluating on TartanAir Dataset

Similarly for TartanAir:

```bash
python evals/eval_evs/eval_tartan_evs.py \
    --datapath <PATH_TO_TARTANAIR_DATASET> \
    --weights DEVO.pth \
    --save_per_frame_cloud \
    --save_per_frame_cloud_path results/tartan_clouds
```

### Example: Using export_pointcloud.py (FPV Dataset)

You can also use the `scripts/export_pointcloud.py` script to export both the final reconstruction and per-frame point clouds:

```bash
python scripts/export_pointcloud.py \
    --datapath <PATH_TO_FPV_DATASET> \
    --weights DEVO.pth \
    --out results/fpv_final.npy \
    --save_per_frame_cloud \
    --save_per_frame_cloud_path results/fpv_clouds \
    --export_frame_data \
    --frame_data_out results/fpv_frames.npz
```

## Output Format

The exported `.ply` files contain vertex positions `(x, y, z)` per frame. `export_pointcloud.py` can also drop a `results/..._frames.npz` bundle with the following arrays:

- `frame_ids` – zero-based frame indices recorded right after initialization.
- `timestamps` – the sequence timestamp per frame (float, whatever iterator emits).
- `poses` – rigid-body poses in `(tx, ty, tz, qx, qy, qz, qw)` format.
- `intrinsics` – `(fx, fy, cx, cy)` per frame (already scaled for DEVO’s resolution).
- `counts` – number of valid sparse points contributed by each frame.
- `offsets` – starting index of each frame in the concatenated point/depth arrays.
- `points` – concatenated `(x, y, z)` samples in world coordinates.
- `depths` – per-point depth value that was used during bundle adjustment.

You can reconstruct the point cloud for frame `i` via

```python
data = np.load("results/fpv_frames.npz")
start = data["offsets"][i]
count = data["counts"][i]
pts_i = data["points"][start:start+count]
deps_i = data["depths"][start:start+count]
pose_i = data["poses"][i]
```

`counts` can be zero for frames where no reliable patches were promoted; skip those gracefully.

## Supported Scripts

Currently, the following evaluation scripts support these arguments:

- `evals/eval_evs/eval_rpg_evs.py`
- `evals/eval_evs/eval_tartan_evs.py`
- `scripts/export_pointcloud.py`

If you need to use this feature with other scripts, you can easily add the arguments by following the pattern in the supported scripts.
