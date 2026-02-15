# Seeded MC-EMVS (mono/stereo) from DEVO outputs

This folder glues together:

- **DEVO outputs** (poses + sparse inverse-depth seeds)
- **raw event streams** (Vector-format `.hdf5`)
- **MC-EMVS / mapper_emvs_stereo** from `dvs_mcemvs`

to run **MC-EMVS** with **seeded depth range** (from DEVO) and produce depth/pointcloud outputs.

## What “seeded” means here

MC-EMVS internally sweeps a fixed depth range (`min_depth..max_depth`) with `dimZ` planes.

This pipeline sets that depth range automatically from DEVO’s sparse inverse-depths (e.g., using percentiles),
so MC-EMVS searches a *much tighter* and more relevant depth interval for your sequence.

Per-pixel depth priors are not injected into the MC-EMVS core here (that would require modifying the C++ code).

## Requirements

1) A built `mapper_emvs_stereo` binary (ROS Noetic / catkin build) from the MC-EMVS repo.

   The MC-EMVS repo is present on this machine at:
   - `/shared/dvs_mcemvs`

   Follow `/shared/dvs_mcemvs/docs/installation.md` (ROS Noetic + `catkin build mapper_emvs_stereo`).

2) Python deps (run in the `devo` conda env):

```bash
/opt/conda/envs/devo/bin/pip install rosbags
```

## Inputs (from your DEVO run)

From `scripts/export_pointcloud.py` you already have:

- `results/<seq>/pointcloud_poses.npy`  (DEVO poses, world->cam, shape `(N,7)` as `xyz + quat_xyzw`)
- `results/<seq>/pointcloud_tstamps.npy` (timestamps in microseconds, shape `(N,)`)
- `results/<seq>/frames.npz` (sparse centers + inverse depths for seeding)

## Prepare ROS bags + config (Vector dataset)

Example for your sequence:

```bash
/opt/conda/envs/devo/bin/python pipelines/seeded_mcemvs/prepare_vector_seeded_mcemvs.py \
  --seq_dir datasets/corridors_dolly/corridors_dolly \
  --out_dir results/corridors_dolly/mcemvs_seeded \
  --devo_poses_npy results/corridors_dolly/pointcloud_poses.npy \
  --devo_tstamps_npy results/corridors_dolly/pointcloud_tstamps.npy \
  --frames_npz results/corridors_dolly/frames.npz \
  --side left \
  --mode mono
```

This writes:
- `results/.../mcemvs_seeded/events_left.bag`
- `results/.../mcemvs_seeded/events_right.bag` (duplicated from left when `--mode mono`)
- `results/.../mcemvs_seeded/poses.bag`
- `results/.../mcemvs_seeded/calib_seeded.yaml`
- `results/.../mcemvs_seeded/seeded.conf`

## Run MC-EMVS

In your ROS environment (where `rosrun mapper_emvs_stereo run_emvs` works):

```bash
rosrun mapper_emvs_stereo run_emvs --flagfile=results/corridors_dolly/mcemvs_seeded/seeded.conf
```

Outputs go to the `--out_path` folder defined inside `seeded.conf` (the same `out_dir` by default).

