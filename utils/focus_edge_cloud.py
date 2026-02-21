import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


def _sobel_grad_mag(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude for a 2D array (float32)."""
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    img = img.astype(np.float32, copy=False)
    # replicate padding
    p = np.pad(img, ((1, 1), (1, 1)), mode="edge")

    # Sobel kernels:
    # gx = [[1,0,-1],[2,0,-2],[1,0,-1]]
    # gy = [[1,2,1],[0,0,0],[-1,-2,-1]]
    gx = (
        (p[0:-2, 0:-2] + 2.0 * p[1:-1, 0:-2] + p[2:, 0:-2])
        - (p[0:-2, 2:] + 2.0 * p[1:-1, 2:] + p[2:, 2:])
    )
    gy = (
        (p[0:-2, 0:-2] + 2.0 * p[0:-2, 1:-1] + p[0:-2, 2:])
        - (p[2:, 0:-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:])
    )
    return np.sqrt(gx * gx + gy * gy, dtype=np.float32)


def extract_edge_pixels_qres(
    event_img_qres: np.ndarray,
    topk: int = 6000,
    border: int = 2,
) -> np.ndarray:
    """Return (N,2) uv edge pixels (quarter-res) from an event image."""
    if event_img_qres is None:
        return np.zeros((0, 2), dtype=np.float32)

    g = _sobel_grad_mag(event_img_qres)

    if border > 0:
        g[:border, :] = 0
        g[-border:, :] = 0
        g[:, :border] = 0
        g[:, -border:] = 0

    flat = g.reshape(-1)
    if flat.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    k = int(min(max(int(topk), 0), flat.size))
    if k == 0:
        return np.zeros((0, 2), dtype=np.float32)

    # argpartition for speed; then filter zero/NaN
    idx = np.argpartition(-flat, k - 1)[:k]
    vals = flat[idx]
    keep = np.isfinite(vals) & (vals > 0)
    idx = idx[keep]
    if idx.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    h, w = g.shape
    u = (idx % w).astype(np.float32)
    v = (idx // w).astype(np.float32)
    return np.stack([u, v], axis=1).astype(np.float32)


def inv_depth_prior_knn(
    query_uv: np.ndarray,
    seed_uv: np.ndarray,
    seed_inv_depth: np.ndarray,
    knn: int = 4,
    max_dist: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted-mean inverse-depth prior for each query pixel.

    Returns:
        (inv_depth0, valid_mask)
    """
    if query_uv is None or len(query_uv) == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)

    seed_uv = np.asarray(seed_uv, dtype=np.float32)
    seed_inv_depth = np.asarray(seed_inv_depth, dtype=np.float32)
    if seed_uv.ndim != 2 or seed_uv.shape[1] != 2:
        return np.zeros((len(query_uv),), dtype=np.float32), np.zeros((len(query_uv),), dtype=bool)
    if seed_inv_depth.ndim != 1 or seed_inv_depth.shape[0] != seed_uv.shape[0]:
        return np.zeros((len(query_uv),), dtype=np.float32), np.zeros((len(query_uv),), dtype=bool)

    good = np.isfinite(seed_uv).all(axis=1) & np.isfinite(seed_inv_depth) & (seed_inv_depth > 0)
    seed_uv = seed_uv[good]
    seed_inv_depth = seed_inv_depth[good]
    if seed_uv.shape[0] < max(1, int(knn)):
        return np.zeros((len(query_uv),), dtype=np.float32), np.zeros((len(query_uv),), dtype=bool)

    tree = cKDTree(seed_uv)
    dist, idx = tree.query(np.asarray(query_uv, dtype=np.float32), k=int(knn), distance_upper_bound=float(max_dist))
    dist = np.atleast_2d(dist)
    idx = np.atleast_2d(idx)

    valid = np.isfinite(dist).all(axis=1) & (idx < seed_uv.shape[0]).all(axis=1)
    inv0 = np.zeros((len(query_uv),), dtype=np.float32)
    if not np.any(valid):
        return inv0, valid

    w = 1.0 / np.clip(dist[valid], 1e-3, None)
    inv = (w * seed_inv_depth[idx[valid]]).sum(axis=1) / np.clip(w.sum(axis=1), 1e-6, None)
    inv0[valid] = inv.astype(np.float32)
    return inv0, valid


def _bilinear(img: np.ndarray, u: float, v: float) -> float:
    h, w = img.shape
    if u < 0 or v < 0 or u > (w - 1) or v > (h - 1):
        return 0.0
    x0 = int(np.floor(u))
    y0 = int(np.floor(v))
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    a = float(u - x0)
    b = float(v - y0)
    v00 = float(img[y0, x0])
    v10 = float(img[y0, x1])
    v01 = float(img[y1, x0])
    v11 = float(img[y1, x1])
    return (1.0 - a) * (1.0 - b) * v00 + a * (1.0 - b) * v10 + (1.0 - a) * b * v01 + a * b * v11


def _patch_score(img: np.ndarray, u: float, v: float, radius: int) -> float:
    if radius <= 0:
        return _bilinear(img, u, v)
    best = 0.0
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            best = max(best, _bilinear(img, u + du, v + dv))
    return best


def _pose_to_r_cw_t(pose_xyz_quat_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose_xyz_quat_xyzw, dtype=np.float32).reshape(-1)
    if pose.size != 7:
        raise ValueError(f"Expected pose size 7 (xyz + quat xyzw), got {pose.size}")
    t = pose[:3].astype(np.float32)
    q = pose[3:].astype(np.float32)  # xyzw
    rcw = R.from_quat(q).as_matrix().astype(np.float32)  # world -> cam
    return rcw, t


def _cam_center_world(rcw: np.ndarray, t: np.ndarray) -> np.ndarray:
    # p_c = Rcw p_w + t => camera center: p_w = -Rcw^T t
    return (-rcw.T @ t).astype(np.float32)


def refine_edge_inv_depth_focus(
    edge_uv: np.ndarray,
    inv0: np.ndarray,
    ref_pose: np.ndarray,
    ref_intr: np.ndarray,
    support_frames: list[dict],
    rho_samples: int = 25,
    rho_rel: float = 0.2,
    rho_abs: float | None = None,
    patch_radius: int = 1,
    min_peak_ratio: float = 1.2,
    min_score: float = 0.0,
    min_baseline: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-view focus refinement of inverse depth for edge pixels.

    Args:
        edge_uv: (N,2) edge pixels in ref frame qres coords.
        inv0: (N,) prior inverse depth for each edge pixel (0 if invalid).
        ref_pose: (7,) xyz + quat xyzw, world->cam.
        ref_intr: (4,) fx,fy,cx,cy in qres coords.
        support_frames: list of dicts with keys:
            - "pose": (7,) world->cam
            - "intr": (4,) qres intrinsics
            - "event_img": (Hq,Wq) float

    Returns:
        inv_star: (N,) refined inverse depths (0 if rejected)
        conf: (N,) confidence score (peak ratio; 0 if rejected)
    """
    edge_uv = np.asarray(edge_uv, dtype=np.float32)
    inv0 = np.asarray(inv0, dtype=np.float32).reshape(-1)
    if edge_uv.ndim != 2 or edge_uv.shape[1] != 2:
        raise ValueError(f"edge_uv must be (N,2); got {edge_uv.shape}")
    if inv0.shape[0] != edge_uv.shape[0]:
        raise ValueError(f"inv0 must be (N,); got {inv0.shape} for N={edge_uv.shape[0]}")

    if len(support_frames) == 0:
        return np.zeros_like(inv0), np.zeros_like(inv0)

    fx0, fy0, cx0, cy0 = np.asarray(ref_intr, dtype=np.float32).reshape(4)
    rcw0, t0 = _pose_to_r_cw_t(ref_pose)
    rwc0 = rcw0.T
    ow = (-rwc0 @ t0).astype(np.float32)
    c0 = _cam_center_world(rcw0, t0)

    # Precompute per-support transforms (p_ck = A_k * p_c0 + b_k)
    Ak = []
    bk = []
    intr_k = []
    img_k = []
    for fr in support_frames:
        rcwk, tk = _pose_to_r_cw_t(fr["pose"])
        ck = _cam_center_world(rcwk, tk)
        if min_baseline > 0.0:
            if float(np.linalg.norm(ck - c0)) < float(min_baseline):
                continue
        Ak.append((rcwk @ rwc0).astype(np.float32))
        bk.append((rcwk @ ow + tk).astype(np.float32))
        intr_k.append(np.asarray(fr["intr"], dtype=np.float32).reshape(4))
        img_k.append(np.asarray(fr["event_img"], dtype=np.float32))

    if len(Ak) == 0:
        return np.zeros_like(inv0), np.zeros_like(inv0)

    Ak = np.stack(Ak, axis=0)  # (K,3,3)
    bk = np.stack(bk, axis=0)  # (K,3)
    intr_k = np.stack(intr_k, axis=0)  # (K,4)

    inv_star = np.zeros_like(inv0, dtype=np.float32)
    conf = np.zeros_like(inv0, dtype=np.float32)

    for i in range(edge_uv.shape[0]):
        rho0 = float(inv0[i])
        if not np.isfinite(rho0) or rho0 <= 0:
            continue

        if rho_abs is None:
            dr = float(abs(rho_rel)) * rho0
        else:
            dr = float(abs(rho_abs))
        if dr <= 0:
            continue

        rhos = rho0 + np.linspace(-dr, dr, int(max(rho_samples, 3)), dtype=np.float32)
        rhos = np.clip(rhos, 1e-6, None)
        zs = 1.0 / rhos  # depth

        u, v = float(edge_uv[i, 0]), float(edge_uv[i, 1])
        s = np.array([(u - cx0) / fx0, (v - cy0) / fy0, 1.0], dtype=np.float32)

        # a_k = A_k @ s, p_ck(z) = z * a_k + b_k
        a = (Ak @ s.reshape(3, 1)).squeeze(-1)  # (K,3)

        scores = np.zeros((zs.shape[0],), dtype=np.float32)
        for j, z in enumerate(zs):
            p = z * a + bk  # (K,3)
            zc = p[:, 2]
            good = zc > 1e-6
            if not np.any(good):
                continue

            x = p[good, 0]
            y = p[good, 1]
            zgg = zc[good]
            fxk = intr_k[good, 0]
            fyk = intr_k[good, 1]
            cxk = intr_k[good, 2]
            cyk = intr_k[good, 3]
            uu = fxk * (x / zgg) + cxk
            vv = fyk * (y / zgg) + cyk

            ssum = 0.0
            gi = np.flatnonzero(good)
            for idx_local, kk in enumerate(gi):
                ssum += _patch_score(img_k[kk], float(uu[idx_local]), float(vv[idx_local]), int(patch_radius))
            scores[j] = float(ssum)

        if not np.any(np.isfinite(scores)):
            continue

        best = int(np.argmax(scores))
        s1 = float(scores[best])
        if s1 <= float(min_score):
            continue

        # second-best for peak ratio
        tmp = scores.copy()
        tmp[best] = -np.inf
        s2 = float(np.max(tmp))
        ratio = (s1 / max(s2, 1e-6)) if np.isfinite(s2) else 1e6
        if ratio < float(min_peak_ratio):
            continue

        inv_star[i] = float(rhos[best])
        conf[i] = float(ratio)

    return inv_star, conf


def backproject_edge_points_world(
    edge_uv: np.ndarray,
    inv_depth: np.ndarray,
    pose_xyz_quat_xyzw: np.ndarray,
    intr_fx_fy_cx_cy: np.ndarray,
) -> np.ndarray:
    """Backproject edge pixels with inverse depth into world coords."""
    edge_uv = np.asarray(edge_uv, dtype=np.float32)
    inv_depth = np.asarray(inv_depth, dtype=np.float32).reshape(-1)
    if edge_uv.ndim != 2 or edge_uv.shape[1] != 2:
        return np.zeros((0, 3), dtype=np.float32)
    if inv_depth.shape[0] != edge_uv.shape[0]:
        return np.zeros((0, 3), dtype=np.float32)

    good = np.isfinite(edge_uv).all(axis=1) & np.isfinite(inv_depth) & (inv_depth > 0)
    if not np.any(good):
        return np.zeros((0, 3), dtype=np.float32)

    fx, fy, cx, cy = np.asarray(intr_fx_fy_cx_cy, dtype=np.float32).reshape(4)
    u = edge_uv[good, 0]
    v = edge_uv[good, 1]
    z = 1.0 / np.clip(inv_depth[good], 1e-6, None)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pc = np.stack([x, y, z], axis=1).astype(np.float32)

    rcw, t = _pose_to_r_cw_t(pose_xyz_quat_xyzw)
    pw = (rcw.T @ (pc - t).T).T.astype(np.float32)
    return pw

