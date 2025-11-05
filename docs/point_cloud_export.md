# DEVO Point-Cloud & Depth Export

This note explains the changes that enable DEVO to expose its sparse 3D map,
how the export works internally, and how to use the new API hook in practice.

## What Changed
- `devo/devo.py`
  - `DEVO.terminate()` now accepts `return_observables=True`.  
    When enabled, the method gathers the latest sparse point cloud and per-patch
    depths before returning.
- `utils/eval_utils.py`
  - The helper entrypoints (`run_rgb`, `run_voxel_norm_seq`, `run_voxel`) mirror
    the new flag so you can request the additional outputs without touching the
    core SLAM loop.

None of the existing call sites break: if you omit the flag, the functions
behave exactly as before.

## Background: DPVO’s Sparse Point Cloud
DPVO builds a point cloud from small image patches. Each tracked patch stores a
3×3 grid containing

- the pixel coordinates of the patch center,
- the offsets of the surrounding pixels, and
- the per-sample depth (actually inverse depth during optimization).

Whenever bundle adjustment updates the camera poses and patch depths, DPVO
converts the patch center back into a 3D point:

1. **Back-project** the center pixel using the camera intrinsics to obtain a
   homogeneous ray (`dpvo/projective_ops.py::iproj`).
2. **Transform** the ray with the estimated pose so that it lies in world space.
3. **De-homogenize** (divide by the last coordinate) to recover Euclidean XYZ.

This produces one sparse 3D point per active patch. The points live in
`PatchGraph.points_` and feed the Pangolin viewer or any downstream consumer.

## How This Carries Over to DEVO
DEVO reuses the same projective operators. Every event patch is stored in
`self.patches_` with the same `(x, y, depth)` layout as DPVO, so the DPVO
pipeline drops in unchanged:

- `devo/projective_ops.py` is a near mirror of `dpvo/projective_ops.py`.
- During each update step, DEVO already called `pops.point_cloud` to refresh the
  viewer buffers (`devo/devo.py:342-344`).
- The new `return_observables` flag simply lifts those tensors to CPU memory and
  returns them to the caller. No additional geometry code was required.

The depth values we export are the same numbers the optimizer uses internally:
the center entry of each patch (`self.patches[..., 2, P//2, P//2]`). For a
beginner, you can think of each patch as a little “rangefinder” that knows the
distance to the scene at that pixel; stitching all of those readings together
gives the sparse reconstruction.

## How the Point Cloud Is Produced in DEVO
DEVO tracks a 3×3 patch around every selected keypoint. Each patch carries the
pixel coordinates and an inverse-depth estimate at each grid location. During
bundle adjustment, the center sample of every active patch is lifted to 3D with
the following steps (implemented in `devo/projective_ops.py`):

1. **Back-project** the patch center using the calibrated intrinsics to obtain a
   homogeneous 3D ray (`iproj`).
2. **Transform** that ray into the world frame with the current pose estimate.
3. **Normalize** by the homogeneous coordinate to recover Euclidean XYZ.

The resulting tensor is flattened and cached in `self.points_` for visualization
and export. At the same time, the scalar depth value used for back-projection is
read from `self.patches[..., 2, P//2, P//2]`.

## Using the New Outputs
You can request the sparse map from either the high-level helpers or directly
from the SLAM object.

```python
from utils.eval_utils import run_rgb

poses, tstamps, flowdata, point_cloud, depths = run_rgb(
    imagedir=data_root,
    cfg=config,
    network=weights,
    iterator=your_iterator,
    return_observables=True,
)

print(point_cloud.shape)  # (num_patches, 3)
print(depths.shape)       # (num_patches,)
```

Or, if you manage the SLAM instance yourself:

```python
slam = DEVO(cfg, network, ht=H, wd=W)
...
poses, tstamps, point_cloud, depths = slam.terminate(return_observables=True)
```

Both functions return NumPy arrays ready for saving to `.ply`, `.npy`, or any
other downstream format.

## Why It Works
The export reuses the same mapping utilities DEVO already relies on for
visualization (`pops.point_cloud`). No new geometry code is introduced; we simply
expose the intermediate tensors that were already being computed at each update
step. Because the data lives on the GPU during tracking, the export step moves
it to CPU memory only when explicitly requested, keeping the default path
lightweight.

## Notes & Limitations
- The point cloud is **sparse** and reflects the currently active patches. It
  is not a dense depth map.
- Depth values correspond to the inverse-depth normalization DEVO uses
  internally; they are per-patch, not per-pixel.
- Requesting observables incurs a small sync and copy overhead when
  `return_observables=True`.

That’s all—call the helper with the new flag whenever you need the sparse map. 
If additional export formats are useful (e.g. direct `.ply` serialization), feel
free to extend the helper with a simple writer.
