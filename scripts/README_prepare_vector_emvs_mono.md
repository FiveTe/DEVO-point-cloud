# prepare_vector_emvs_mono.py

Convert Vector events + poses into a single ROS1 bag and `.conf` for `rpg_emvs/mapper_emvs`.

## What it writes

- One bag with:
  - `/dvs/events` (`dvs_msgs/EventArray`)
  - `/dvs/camera_info` (`sensor_msgs/CameraInfo`)
  - `/pose` (`geometry_msgs/PoseStamped`)
- One EMVS config file (`.conf`) for `rosrun mapper_emvs run_emvs --flagfile=...`

## EventArray schema (important)

`--eventarray_schema` controls `/dvs/events` message layout:

- `5e8` (default): MD5 `5e8beee5a6c107e504c2e78903c224b8`
- `732`: MD5 `732c9cc0676a970b9aa2873b0d444ac2`

If your ROS environment expects `5e8...`, you can run the script with defaults.

## Example 1: Use DEVO poses

```bash
cd /shared/DEVO-point-cloud

/opt/conda/envs/devo/bin/python scripts/prepare_vector_emvs_mono.py \
  --seq_dir datasets/board_slow/board_slow \
  --devo_poses_npy results/board_slow/pointcloud_poses.npy \
  --devo_tstamps_npy results/board_slow/pointcloud_tstamps.npy \
  --frames_npz results/board_slow/frames.npz \
  --out_bag results/board_slow/emvs_input.bag \
  --out_conf results/board_slow/emvs_input.conf \
  --side left \
  --eventarray_schema 5e8 \
  --overwrite
```

## Example 2: Use ground-truth poses (`poses_evs_left.txt`)

`poses_evs_left.txt` format:

`timestamp_us tx ty tz qx qy qz qw`

Create npy files first:

```bash
cd /shared/DEVO-point-cloud

/opt/conda/envs/devo/bin/python - <<'PY'
import numpy as np
a = np.loadtxt("datasets/board_slow/board_slow/poses_evs_left.txt")
np.save("results/board_slow/gt_poses.npy", a[:, 1:].astype(np.float32))
np.save("results/board_slow/gt_tstamps.npy", a[:, 0].astype(np.float64))
print("saved gt_poses.npy and gt_tstamps.npy")
PY
```

Then generate EMVS bag/conf:

```bash
/opt/conda/envs/devo/bin/python scripts/prepare_vector_emvs_mono.py \
  --seq_dir datasets/board_slow/board_slow \
  --devo_poses_npy results/board_slow/gt_poses.npy \
  --devo_tstamps_npy results/board_slow/gt_tstamps.npy \
  --frames_npz results/board_slow/frames.npz \
  --out_bag results/board_slow/emvs_input_gt.bag \
  --out_conf results/board_slow/emvs_input_gt.conf \
  --side left \
  --eventarray_schema 5e8 \
  --overwrite
```

## Verify `/dvs/events` MD5 in output bag

```bash
/opt/conda/envs/devo/bin/python - <<'PY'
from rosbags.rosbag1 import Reader
bag = "/shared/DEVO-point-cloud/results/board_slow/emvs_input_gt.bag"
with Reader(bag) as r:
    for c in r.connections:
        if c.topic == "/dvs/events":
            print(c.digest)
            break
PY
```

Expected for `--eventarray_schema 5e8`:

`5e8beee5a6c107e504c2e78903c224b8`

## Run EMVS

```bash
rosrun mapper_emvs run_emvs --flagfile=/shared/DEVO-point-cloud/results/board_slow/emvs_input_gt.conf
```

Use the `.conf` file for `--flagfile` (not the `.bag` path).
