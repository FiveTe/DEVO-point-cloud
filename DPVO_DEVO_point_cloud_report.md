# DPVO & DEVO Sparse Map Pipeline

This document explains how Deep Patch Visual Odometry (DPVO) produces its
sparse 3D reconstruction, how that map propagates through the system, and how
the same machinery is reused in Deep Event Visual Odometry (DEVO). It is aimed
at readers who are comfortable with SLAM fundamentals but new to the specific
implementation choices in these projects.

---

## 1. System-Level Overview

Both DPVO and DEVO maintain a sliding window of frames (or event voxel grids in
DEVO) and track a fixed number of patches per keyframe. Each patch encodes the
image coordinates of a small 3×3 neighborhood and a depth estimate at every
sample. After every bundle-adjustment (BA) update, the systems back-project the
center of each patch into world coordinates, yielding a sparse point cloud that
represents the currently tracked scene structure.

Key architectural components shared across the two codebases:

- **Patch extraction (`dpvo/net.py`, `devo/enet.py`)**  
  Neural encoders produce per-frame features, select patch centroids, and
  extract 3×3 neighborhoods around those centroids.

- **Correlation & update (`dpvo/altcorr`, `dpvo/net.py:Update`, `devo/enet.py`)**  
  Custom CUDA layers compute matching cost volumes. A recurrent update module
  refines optical flow, depth, and pose residuals.

- **Factor graph bookkeeping (`dpvo/patchgraph.py`, inlined in `devo/devo.py`)**  
  Maintains the list of active frames, patches, and pairwise edges that feed
  the optimizer.

- **Differentiable BA (`dpvo/fastba`, `devo/fastba`)**  
  GPU-accelerated Gauss-Newton steps optimize camera poses and patch depths.

- **Projective operators (`dpvo/projective_ops.py`, `devo/projective_ops.py`)**  
  Provide the math for back-projecting patches and generating the sparse point
  cloud.

---

## 2. DPVO Mapping Pipeline

### 2.1 Patch Geometry

When a new frame enters DPVO (`dpvo/dpvo.py::__call__`), the network selects
`PATCHES_PER_FRAME` locations. For each patch, it stores a 3×3×3 tensor in
`PatchGraph.patches_` (`dpvo/patchgraph.py:21-56`). The three channels hold:

1. **x** – pixel abscissa of each sample in the patch (in the scaled pyramid),
2. **y** – pixel ordinate,
3. **d** – (inverse) depth per sample.

These tensors live on the GPU for fast access during optimization.

### 2.2 Tracking and Optimization

1. **Reprojection** (`dpvo/dpvo.py:323-331`): For each edge in the factor graph,
   `pops.transform` back-projects the patch from its host frame and predicts its
   location in the target frame.

2. **Neural Update** (`dpvo/net.py:101-187`, `dpvo/dpvo.py:327-339`): A learned
   GRU-like block fuses the correlation response, context features, and
   historical state to produce optical-flow corrections (`delta`) and per-edge
   weights.

3. **Bundle Adjustment** (`dpvo/dpvo.py:340-353`): The fast BA module solves for
   pose and depth increments. This step overwrites `patches_` and `poses_` with
   the refined estimates.

### 2.3 Sparse Point Cloud Generation

After each BA iteration, DPVO updates the 3D point cloud:

```python
points = pops.point_cloud(
    SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m]
)
points = (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(-1, 3)
self.pg.points_[:len(points)] = points[:]
```

— `dpvo/dpvo.py:352-360`

Breaking down `pops.point_cloud` (`dpvo/projective_ops.py:107-113`):

1. **Back-projection** (`iproj`): Converts the stored `(x, y, depth)` grid into
   homogeneous coordinates in the host camera frame.
2. **Pose transform**: Applies the inverse world pose of the host frame to place
   the points in global coordinates.
3. **Homogeneous normalization**: Divides xyz by the last component to obtain
   Euclidean positions.

Only the center sample `[1,1]` of each 3×3 patch is used for visualization and
export, providing one 3D point per tracked patch. The point cloud is stored in
`PatchGraph.points_`, while per-patch RGB averages are stored in
`PatchGraph.colors_` for viewer overlays.

---

## 3. DEVO Mapping Pipeline

DEVO introduces event-voxel inputs and a different neural backbone, but the
mapping logic closely follows DPVO:

- **Patch representation** (`devo/devo.py:96-118`) mirrors the DPVO layout.
- **Update loop** (`devo/devo.py:308-344`) performs the same steps: reprojection,
  neural residual prediction, and `fastba` optimization.
- **Point cloud update** uses the same helper:

```python
points = pops.point_cloud(
    SE3(self.poses), self.patches[:, :self.m], self.intrinsics, self.ix[:self.m]
)
points = (points[..., 1, 1, :3] / points[..., 1, 1, 3:]).reshape(-1, 3)
self.points_[:len(points)] = points[:]
```

— `devo/devo.py:342-344`

Because the `projective_ops` module is copied almost verbatim into DEVO, the
exact same math maps event-derived depths into world-space coordinates.

---

## 4. Exporting the Sparse Map

Originally, neither project exposed the sparse map directly; the points were
used for real-time visualization only. The recent change adds an opt-in path to
retrieve them:

- `DEVO.devo.DEVO.terminate(return_observables=True)` (see
  `devo/devo.py:186-233`) now returns:
  - `point_cloud` – an `(N, 3)` NumPy array of XYZ positions,
  - `depths` – a `(N,)` array containing the per-patch depth values.

- Helper wrappers (`utils/eval_utils.py:38-154`) gained a matching
  `return_observables` flag, so evaluation scripts can pull the data without
  touching the core event-processing loop.

Export is a purely read-only operation: it runs the existing `pops.point_cloud`
routine, moves the tensor to CPU, and converts it to NumPy. No new geometry code
is required.

---

## 5. Summary

- Both DPVO and DEVO maintain a sparse set of 3D landmarks derived from 3×3
  patches centered on neural keypoints.
- The point cloud is produced by back-projecting the stored `(x, y, depth)` of
  each patch’s center using the camera intrinsics and current pose.
- DEVO inherits DPVO’s projective operators wholesale, so the sparse-map logic
  is identical across the two systems. The only difference lies in how the
  patches are selected (images vs. event voxels) and the encoder network that
  provides the features.
- The new `return_observables` flag exposes the map to downstream code, enabling
  visualizations, mesh reconstruction, or evaluation without modifying the SLAM
  core.

For further reading, consult:

- `dpvo/dpvo.py`, `dpvo/projective_ops.py`, and `dpvo/patchgraph.py`
- `devo/devo.py` and `devo/projective_ops.py`
- `DEVO/README_point_cloud.md` for quick usage instructions
