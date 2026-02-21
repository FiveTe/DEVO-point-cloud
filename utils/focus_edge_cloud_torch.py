import torch
import torch.nn.functional as F


def sobel_edges_topk_qres_torch(
    event_img_qres: torch.Tensor,
    topk: int = 6000,
    border: int = 2,
) -> torch.Tensor:
    """Extract top-k edge pixels (u,v) from a 2D event image (qres) on GPU.

    Args:
        event_img_qres: (H,W) tensor on CUDA.
    Returns:
        uv: (N,2) float32 tensor with columns (u,v) in qres pixel coordinates.
    """
    if event_img_qres is None:
        return torch.zeros((0, 2), device="cuda", dtype=torch.float32)
    if event_img_qres.ndim != 2:
        raise ValueError(f"event_img_qres must be 2D (H,W), got {tuple(event_img_qres.shape)}")

    img = event_img_qres.to(torch.float32)[None, None]  # (1,1,H,W)
    device = img.device
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device, dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(img, sobel_x, padding=1)[0, 0]
    gy = F.conv2d(img, sobel_y, padding=1)[0, 0]
    g = torch.sqrt(gx * gx + gy * gy)

    if border > 0:
        g[:border, :] = 0
        g[-border:, :] = 0
        g[:, :border] = 0
        g[:, -border:] = 0

    flat = g.reshape(-1)
    if flat.numel() == 0:
        return torch.zeros((0, 2), device=device, dtype=torch.float32)

    k = int(min(max(int(topk), 0), flat.numel()))
    if k == 0:
        return torch.zeros((0, 2), device=device, dtype=torch.float32)

    vals, idx = torch.topk(flat, k, largest=True, sorted=False)
    keep = torch.isfinite(vals) & (vals > 0)
    idx = idx[keep]
    if idx.numel() == 0:
        return torch.zeros((0, 2), device=device, dtype=torch.float32)

    h, w = g.shape
    u = (idx % w).to(torch.float32)
    v = torch.div(idx, w, rounding_mode="floor").to(torch.float32)
    return torch.stack([u, v], dim=-1)


def _bilinear_sample(img_hw: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Bilinear sample a single-channel image at float coords.

    Args:
        img_hw: (H,W) float tensor
        u,v: same shape float tensors (pixel coords, 0..W-1 / 0..H-1)
    Returns:
        samples: same shape tensor
    """
    h, w = img_hw.shape
    u0 = torch.floor(u).to(torch.int64)
    v0 = torch.floor(v).to(torch.int64)
    u1 = u0 + 1
    v1 = v0 + 1

    u0c = u0.clamp(0, w - 1)
    u1c = u1.clamp(0, w - 1)
    v0c = v0.clamp(0, h - 1)
    v1c = v1.clamp(0, h - 1)

    a = (u - u0.to(u.dtype)).clamp(0, 1)
    b = (v - v0.to(v.dtype)).clamp(0, 1)

    v00 = img_hw[v0c, u0c]
    v10 = img_hw[v0c, u1c]
    v01 = img_hw[v1c, u0c]
    v11 = img_hw[v1c, u1c]
    out = (1 - a) * (1 - b) * v00 + a * (1 - b) * v10 + (1 - a) * b * v01 + a * b * v11

    inside = (u >= 0) & (v >= 0) & (u <= (w - 1)) & (v <= (h - 1))
    return torch.where(inside, out, torch.zeros_like(out))


def patch_max_score(img_hw: torch.Tensor, u: torch.Tensor, v: torch.Tensor, radius: int) -> torch.Tensor:
    """Max event evidence in a (2r+1)x(2r+1) patch around each (u,v)."""
    if radius <= 0:
        return _bilinear_sample(img_hw, u, v)
    best = None
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            s = _bilinear_sample(img_hw, u + float(du), v + float(dv))
            best = s if best is None else torch.maximum(best, s)
    return best


def knn_inv_depth_prior_torch(
    query_uv: torch.Tensor,
    seed_uv: torch.Tensor,
    seed_inv_depth: torch.Tensor,
    knn: int = 4,
    max_dist: float = 6.0,
    chunk: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute inverse-depth prior for each query pixel using kNN in 2D (torch, GPU).

    Returns:
        inv0: (N,) float32
        valid: (N,) bool
    """
    device = query_uv.device
    query_uv = query_uv.to(torch.float32)
    seed_uv = seed_uv.to(torch.float32)
    seed_inv_depth = seed_inv_depth.to(torch.float32)

    n = query_uv.shape[0]
    inv0 = torch.zeros((n,), device=device, dtype=torch.float32)
    valid = torch.zeros((n,), device=device, dtype=torch.bool)

    if n == 0 or seed_uv.numel() == 0:
        return inv0, valid

    good = torch.isfinite(seed_uv).all(dim=1) & torch.isfinite(seed_inv_depth) & (seed_inv_depth > 0)
    seed_uv = seed_uv[good]
    seed_inv_depth = seed_inv_depth[good]
    if seed_uv.shape[0] < max(1, int(knn)):
        return inv0, valid

    k = int(max(1, knn))
    max_d2 = float(max_dist) * float(max_dist)

    for s in range(0, n, int(chunk)):
        e = min(s + int(chunk), n)
        q = query_uv[s:e]  # (B,2)
        diff = q[:, None, :] - seed_uv[None, :, :]  # (B,M,2)
        d2 = (diff * diff).sum(dim=-1)  # (B,M)

        # Take k smallest; torch.topk on negative for smallest
        neg = -d2
        vals, idx = torch.topk(neg, k=min(k, d2.shape[1]), dim=1, largest=True, sorted=False)
        d2k = -vals
        inr = d2k <= max_d2

        # weights: 1/sqrt(d2)
        dk = torch.sqrt(torch.clamp(d2k, min=1e-6))
        w = 1.0 / dk
        w = torch.where(inr, w, torch.zeros_like(w))

        invk = seed_inv_depth[idx]  # (B,k)
        num = (w * invk).sum(dim=1)
        den = w.sum(dim=1).clamp_min(1e-6)
        inv = num / den
        ok = inr.any(dim=1) & torch.isfinite(inv) & (inv > 0)

        inv0[s:e] = torch.where(ok, inv, torch.zeros_like(inv))
        valid[s:e] = ok

    return inv0, valid


def quat_xyzw_to_R_cw(q_xyzw: torch.Tensor) -> torch.Tensor:
    """Quaternion (x,y,z,w) -> rotation matrix Rcw (world->cam)."""
    q = q_xyzw.to(torch.float32)
    x, y, z, w = q.unbind(-1)
    n = torch.sqrt(x * x + y * y + z * z + w * w).clamp_min(1e-8)
    x, y, z, w = x / n, y / n, z / n, w / n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    r00 = 1 - 2 * (yy + zz)
    r01 = 2 * (xy - wz)
    r02 = 2 * (xz + wy)
    r10 = 2 * (xy + wz)
    r11 = 1 - 2 * (xx + zz)
    r12 = 2 * (yz - wx)
    r20 = 2 * (xz - wy)
    r21 = 2 * (yz + wx)
    r22 = 1 - 2 * (xx + yy)

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def focus_refine_inv_depth_torch(
    edge_uv: torch.Tensor,  # (N,2)
    inv0: torch.Tensor,  # (N,)
    ref_pose_xyz_quat_xyzw: torch.Tensor,  # (7,)
    ref_intr_fx_fy_cx_cy: torch.Tensor,  # (4,)
    support_frames: list[dict],  # each dict: pose(7,), intr(4,), event_img(H,W)
    rho_samples: int = 25,
    rho_rel: float = 0.2,
    rho_abs: float | None = None,
    patch_radius: int = 1,
    min_peak_ratio: float = 1.2,
    min_score: float = 0.0,
    min_baseline: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized focus refinement over all edge pixels and rho samples (torch, GPU)."""
    device = edge_uv.device
    n = edge_uv.shape[0]
    inv_star = torch.zeros((n,), device=device, dtype=torch.float32)
    conf = torch.zeros((n,), device=device, dtype=torch.float32)
    if n == 0 or len(support_frames) == 0:
        return inv_star, conf

    edge_uv = edge_uv.to(torch.float32)
    inv0 = inv0.to(torch.float32)
    fx0, fy0, cx0, cy0 = ref_intr_fx_fy_cx_cy.to(torch.float32).unbind(0)

    t0 = ref_pose_xyz_quat_xyzw[:3].to(torch.float32)
    q0 = ref_pose_xyz_quat_xyzw[3:].to(torch.float32)
    rcw0 = quat_xyzw_to_R_cw(q0)
    rwc0 = rcw0.transpose(0, 1)
    ow = (-rwc0 @ t0)  # world origin in cam0
    c0 = (-rwc0 @ t0)  # camera center in world (same expression with this pose convention)

    # rays in cam0
    u = edge_uv[:, 0]
    v = edge_uv[:, 1]
    s = torch.stack([(u - cx0) / fx0, (v - cy0) / fy0, torch.ones_like(u)], dim=-1)  # (N,3)

    # Build candidate rho grid per pixel: (N,S)
    rho0 = inv0.clamp_min(1e-6)
    if rho_abs is None:
        dr = (abs(float(rho_rel)) * rho0).clamp_min(1e-6)
    else:
        dr = torch.full_like(rho0, abs(float(rho_abs))).clamp_min(1e-6)
    lin = torch.linspace(-1.0, 1.0, int(max(rho_samples, 3)), device=device, dtype=torch.float32)[None, :]  # (1,S)
    rhos = (rho0[:, None] + dr[:, None] * lin).clamp_min(1e-6)  # (N,S)
    zs = 1.0 / rhos  # (N,S)

    scores = torch.zeros((n, rhos.shape[1]), device=device, dtype=torch.float32)

    for fr in support_frames:
        posek = fr["pose"].to(device=device, dtype=torch.float32).reshape(7)
        intrk = fr["intr"].to(device=device, dtype=torch.float32).reshape(4)
        imgk = fr["event_img"].to(device=device)

        tk = posek[:3]
        qk = posek[3:]
        rcwk = quat_xyzw_to_R_cw(qk)
        rwck = rcwk.transpose(0, 1)
        ck = (-rwck @ tk)
        if min_baseline > 0.0:
            if float(torch.linalg.norm(ck - c0).item()) < float(min_baseline):
                continue

        A = rcwk @ rwc0  # (3,3)
        b = rcwk @ ow + tk  # (3,)

        a = s @ A.transpose(0, 1)  # (N,3)
        p = zs[..., None] * a[:, None, :] + b[None, None, :]  # (N,S,3)

        zc = p[..., 2].clamp_min(1e-6)
        good = p[..., 2] > 1e-6
        x = p[..., 0]
        y = p[..., 1]

        fx, fy, cx, cy = intrk.unbind(0)
        uu = fx * (x / zc) + cx
        vv = fy * (y / zc) + cy

        # sample evidence
        ev = patch_max_score(imgk.to(torch.float32), uu, vv, int(patch_radius))
        ev = torch.where(good, ev, torch.zeros_like(ev))
        scores += ev

    best = torch.argmax(scores, dim=1)  # (N,)
    s1 = scores.gather(1, best[:, None]).squeeze(1)
    tmp = scores.clone()
    tmp.scatter_(1, best[:, None], float("-inf"))
    s2 = tmp.max(dim=1).values
    ratio = s1 / torch.clamp(s2, min=1e-6)

    ok = torch.isfinite(ratio) & (ratio >= float(min_peak_ratio)) & (s1 >= float(min_score)) & torch.isfinite(inv0) & (inv0 > 0)
    inv_best = rhos.gather(1, best[:, None]).squeeze(1)
    inv_star = torch.where(ok, inv_best, torch.zeros_like(inv_best))
    conf = torch.where(ok, ratio, torch.zeros_like(ratio))
    return inv_star, conf


def backproject_world_torch(
    edge_uv: torch.Tensor,
    inv_depth: torch.Tensor,
    pose_xyz_quat_xyzw: torch.Tensor,
    intr_fx_fy_cx_cy: torch.Tensor,
) -> torch.Tensor:
    """Backproject edge pixels to world coords (torch, GPU)."""
    device = edge_uv.device
    edge_uv = edge_uv.to(torch.float32)
    inv_depth = inv_depth.to(torch.float32)
    good = torch.isfinite(edge_uv).all(dim=1) & torch.isfinite(inv_depth) & (inv_depth > 0)
    if not good.any():
        return torch.zeros((0, 3), device=device, dtype=torch.float32)

    fx, fy, cx, cy = intr_fx_fy_cx_cy.to(torch.float32).unbind(0)
    u = edge_uv[good, 0]
    v = edge_uv[good, 1]
    z = 1.0 / inv_depth[good].clamp_min(1e-6)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    pc = torch.stack([x, y, z], dim=1)  # cam coords

    t = pose_xyz_quat_xyzw[:3].to(torch.float32)
    q = pose_xyz_quat_xyzw[3:].to(torch.float32)
    rcw = quat_xyzw_to_R_cw(q)
    pw = (rcw.transpose(0, 1) @ (pc - t[None, :]).T).T
    return pw

