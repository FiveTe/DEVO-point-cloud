# Frame-by-Frame Point Cloud Export

This feature allows you to export the sparse point cloud reconstruction for every frame processed by DEVO. The point clouds are saved as `.ply` files.

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
    --save_per_frame_cloud_path results/fpv_clouds
```

## Output Format

The exported files are standard binary PLY files containing vertex positions (x, y, z). You can visualize them using tools like MeshLab, CloudCompare, or Open3D.

## Supported Scripts

Currently, the following evaluation scripts support these arguments:

- `evals/eval_evs/eval_rpg_evs.py`
- `evals/eval_evs/eval_tartan_evs.py`
- `scripts/export_pointcloud.py`

If you need to use this feature with other scripts, you can easily add the arguments by following the pattern in the supported scripts.
