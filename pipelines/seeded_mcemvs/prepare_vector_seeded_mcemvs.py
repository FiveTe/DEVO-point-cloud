import argparse
import os
from dataclasses import dataclass

import numpy as np


def _load_vector_events_hdf5(seq_dir: str, side: str) -> tuple[str, str]:
    """Return (hdf5_path, prefix_name) for Vector-format sequences."""
    seq = os.path.basename(os.path.normpath(seq_dir))
    fname = os.path.join(seq_dir, f"{seq}1.synced.{side}_event.hdf5")
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Vector events file not found: {fname}")
    return fname, seq


def _load_intrinsics_fx_fy_cx_cy(seq_dir: str, side: str) -> np.ndarray:
    path = os.path.join(seq_dir, f"calib_undist_evs_{side}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Intrinsics not found: {path}")
    intr = np.loadtxt(path).astype(np.float32).reshape(-1)
    if intr.size != 4:
        raise ValueError(f"Expected 4 intrinsics (fx,fy,cx,cy), got {intr.size} from {path}")
    return intr


def _estimate_depth_range_from_frames_npz(frames_npz: str) -> tuple[float, float]:
    data = np.load(frames_npz, allow_pickle=False)
    if "depths" not in data:
        raise ValueError(f"{frames_npz} missing 'depths' (inverse depths).")
    inv = np.asarray(data["depths"], dtype=np.float32).reshape(-1)
    inv = inv[np.isfinite(inv) & (inv > 0)]
    if inv.size == 0:
        raise ValueError(f"No valid inverse depths in {frames_npz}.")
    z = 1.0 / inv
    z = z[np.isfinite(z) & (z > 0)]
    lo = float(np.percentile(z, 5))
    hi = float(np.percentile(z, 95))
    # Expand slightly.
    lo = max(1e-3, 0.8 * lo)
    hi = 1.2 * hi
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _devo_world_to_cam_to_cam_to_world(poses_w2c_xyzw: np.ndarray) -> np.ndarray:
    """Convert DEVO pose convention (world->cam) to (cam->world) for MC-EMVS.

    DEVO pose format here: [t_x, t_y, t_z, qx, qy, qz, qw] where the transform is:
        p_c = R_cw * p_w + t

    MC-EMVS expects trajectory.getPoseAt() to return T_w_c (cam->world).

    Returns:
        poses_c2w: (N,7) [c_x, c_y, c_z, qx, qy, qz, qw] for cam->world.
    """
    poses = np.asarray(poses_w2c_xyzw, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"Expected poses (N,7), got {poses.shape}")

    t = poses[:, :3]
    q = poses[:, 3:]  # xyzw, world->cam
    # inverse quaternion for rotation transpose:
    q_inv = q.copy()
    q_inv[:, 0:3] *= -1.0  # (-x,-y,-z,w)

    # rotation matrix from q_inv (cam->world)
    x, y, z, w = q_inv[:, 0], q_inv[:, 1], q_inv[:, 2], q_inv[:, 3]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    n[n < 1e-12] = 1.0
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

    # Camera center in world: c = -R_cw^T t = -R_wc t
    # Here R_wc is exactly the rotation from q_inv.
    cx = -(r00 * t[:, 0] + r01 * t[:, 1] + r02 * t[:, 2])
    cy = -(r10 * t[:, 0] + r11 * t[:, 1] + r12 * t[:, 2])
    cz = -(r20 * t[:, 0] + r21 * t[:, 1] + r22 * t[:, 2])

    out = np.stack([cx, cy, cz, q_inv[:, 0], q_inv[:, 1], q_inv[:, 2], q_inv[:, 3]], axis=1).astype(np.float32)
    return out


@dataclass
class BagPaths:
    left_events: str
    right_events: str
    poses: str


def _write_calib_yaml(out_path: str, intr_left: np.ndarray, intr_right: np.ndarray, h: int, w: int, mode: str) -> None:
    """Write a minimal calib YAML compatible with get_camera_calib_yaml() in MC-EMVS."""
    fx0, fy0, cx0, cy0 = [float(x) for x in intr_left.reshape(4)]
    fx1, fy1, cx1, cy1 = [float(x) for x in intr_right.reshape(4)]

    # Base-to-camera transforms. For mono we keep both identity (camera0 and camera1 collocated).
    T0 = np.eye(4, dtype=np.float64)
    T1 = np.eye(4, dtype=np.float64)
    if mode == "stereo":
        # If you know the stereo baseline/extrinsics, replace T1 here.
        # This placeholder keeps identity (won't behave like real stereo).
        pass

    def mat_to_list(m: np.ndarray) -> list[float]:
        return [float(x) for x in m.reshape(-1).tolist()]

    txt = []
    txt.append("cameras:\n")
    txt.append("  - camera:\n")
    txt.append(f"      image_height: {int(h)}\n")
    txt.append(f"      image_width: {int(w)}\n")
    txt.append("      intrinsics:\n")
    txt.append(f"        data: [{fx0}, {fy0}, {cx0}, {cy0}]\n")
    txt.append("      distortion:\n")
    txt.append("        type: none\n")
    txt.append("    T_B_C:\n")
    txt.append(f"      data: {mat_to_list(T0)}\n")

    txt.append("  - camera:\n")
    txt.append(f"      image_height: {int(h)}\n")
    txt.append(f"      image_width: {int(w)}\n")
    txt.append("      intrinsics:\n")
    txt.append(f"        data: [{fx1}, {fy1}, {cx1}, {cy1}]\n")
    txt.append("      distortion:\n")
    txt.append("        type: none\n")
    txt.append("    T_B_C:\n")
    txt.append(f"      data: {mat_to_list(T1)}\n")

    with open(out_path, "w") as f:
        f.writelines(txt)


def _write_seeded_conf(
    out_path: str,
    bag_left: str,
    bag_right: str,
    bag_pose: str,
    out_dir: str,
    calib_yaml: str,
    start_s: float,
    stop_s: float,
    min_depth: float,
    max_depth: float,
    dimZ: int,
    process_method: int,
    save_mono: bool,
) -> None:
    lines = []
    lines.append(f"--bag_filename_left={bag_left}\n")
    lines.append(f"--bag_filename_right={bag_right}\n")
    lines.append(f"--bag_filename_pose={bag_pose}\n")
    lines.append(f"--out_path={out_dir.rstrip('/') + '/'}\n")
    lines.append("--calib_type=yaml\n")
    lines.append(f"--calib_path={calib_yaml}\n")
    lines.append("--mocap_calib_path=\n")

    # Topics (we write these in the converter).
    lines.append("--event_topic0=/dvs/left/events\n")
    lines.append("--event_topic1=/dvs/right/events\n")
    lines.append("--camera_info_topic0=/dvs/left/camera_info\n")
    lines.append("--camera_info_topic1=/dvs/right/camera_info\n")
    lines.append("--pose_topic=/pose\n")

    lines.append(f"--start_time_s={float(start_s)}\n")
    lines.append(f"--stop_time_s={float(stop_s)}\n")
    lines.append(f"--min_depth={float(min_depth)}\n")
    lines.append(f"--max_depth={float(max_depth)}\n")
    lines.append(f"--dimZ={int(dimZ)}\n")
    lines.append(f"--process_method={int(process_method)}\n")
    lines.append(f"--save_mono={'true' if save_mono else 'false'}\n")
    lines.append("--save_dsi=false\n")
    lines.append("--full_seq=false\n")

    with open(out_path, "w") as f:
        f.writelines(lines)


def _write_rosbags_vector(
    seq_dir: str,
    out_dir: str,
    side: str,
    mode: str,
    devo_poses_c2w: np.ndarray,
    devo_tstamps_us: np.ndarray,
    intr_left: np.ndarray,
    intr_right: np.ndarray,
    h: int,
    w: int,
    max_events_per_msg: int = 20000,
    overwrite: bool = False,
    t_start_us_abs: float | None = None,
    t_stop_us_abs: float | None = None,
    progress_every_events: int = 1_000_000,
) -> BagPaths:
    """Write ROS1 bags using rosbags (no ROS install required)."""
    try:
        # Some Vector HDF5 files may use compressed datasets; ensure HDF5 plugins are available.
        # Also guard against broken HDF5_PLUGIN_PATH pointing to a non-existent directory.
        import os as _os
        plugin_path = _os.environ.get("HDF5_PLUGIN_PATH", "")
        if plugin_path and not _os.path.exists(plugin_path):
            _os.environ.pop("HDF5_PLUGIN_PATH", None)
        import hdf5plugin  # noqa: F401
        import h5py
        from rosbags.rosbag1 import Writer
        from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    except Exception as exc:
        raise RuntimeError(
            "Missing python deps. Run: /opt/conda/envs/devo/bin/pip install rosbags h5py"
        ) from exc

    os.makedirs(out_dir, exist_ok=True)
    bag_left = os.path.join(out_dir, "events_left.bag")
    bag_right = os.path.join(out_dir, "events_right.bag")
    bag_pose = os.path.join(out_dir, "poses.bag")
    if overwrite:
        for p in (bag_left, bag_right, bag_pose):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    # Register dvs_msgs types in a ROS1 typestore.
    ts = get_typestore(Stores.ROS1_NOETIC)
    event_msg = "time ts\nuint16 x\nuint16 y\nbool polarity\n"
    # Match ROS Noetic `dvs_msgs/EventArray` (height/width are uint32).
    event_array_msg = "std_msgs/Header header\ndvs_msgs/Event[] events\nuint32 height\nuint32 width\n"
    ts.register(get_types_from_msg(event_msg, "dvs_msgs/msg/Event"))
    ts.register(get_types_from_msg(event_array_msg, "dvs_msgs/msg/EventArray"))

    Event = ts.get_msgdef("dvs_msgs/msg/Event").cls
    EventArray = ts.get_msgdef("dvs_msgs/msg/EventArray").cls
    Header = ts.get_msgdef("std_msgs/msg/Header").cls
    Time = ts.get_msgdef("builtin_interfaces/msg/Time").cls
    PoseStamped = ts.get_msgdef("geometry_msgs/msg/PoseStamped").cls
    Pose = ts.get_msgdef("geometry_msgs/msg/Pose").cls
    Point = ts.get_msgdef("geometry_msgs/msg/Point").cls
    Quaternion = ts.get_msgdef("geometry_msgs/msg/Quaternion").cls
    CameraInfo = ts.get_msgdef("sensor_msgs/msg/CameraInfo").cls

    def time_from_ns(ns: int) -> Time:
        sec = int(ns // 1_000_000_000)
        nsec = int(ns % 1_000_000_000)
        return Time(sec=sec, nanosec=nsec)

    def stamp_tuple(ns: int) -> tuple[int, int]:
        return int(ns // 1_000_000_000), int(ns % 1_000_000_000)

    def make_camerainfo(intr: np.ndarray) -> CameraInfo:
        fx, fy, cx, cy = [float(x) for x in intr.reshape(4)]
        # rosbags' ROS1 serializer expects numpy arrays for fixed-size arrays.
        K = np.asarray([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0], dtype=np.float64)
        P = np.asarray([fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        R = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        D = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return CameraInfo(
            # Set a non-zero stamp (ROS tools sometimes treat Time(0) as invalid/unset).
            header=Header(seq=0, stamp=time_from_ns(1_000_000_000), frame_id="camera"),
            height=int(h),
            width=int(w),
            distortion_model="plumb_bob",
            D=D,
            K=K,
            R=R,
            P=P,
            binning_x=0,
            binning_y=0,
            roi=ts.get_msgdef("sensor_msgs/msg/RegionOfInterest").cls(
                x_offset=0, y_offset=0, height=0, width=0, do_rectify=False
            ),
        )

    # Event time base from Vector HDF5 uses t_offset. We output relative times starting at 0.
    def write_events_bag(
        h5_path: str,
        bag_path: str,
        topic_events: str,
        topic_info: str,
        intr: np.ndarray,
        base_us: int,
        t_start_us_abs_local: float | None,
        t_stop_us_abs_local: float | None,
    ) -> tuple[float, float]:
        with h5py.File(h5_path, "r") as f:
            t_offset = int(np.asarray(f["t_offset"])[0])
            ev = f["events"]
            ms_to_idx = f["ms_to_idx"]

            # Determine conservative slice bounds using the ms_to_idx table.
            # Absolute time = ev['t'] (us) + t_offset.
            if t_start_us_abs_local is None:
                t_start_us_abs_local = float(t_offset) + float(ev["t"][0])
            if t_stop_us_abs_local is None:
                t_stop_us_abs_local = float(t_offset) + float(ev["t"][-1])

            t0_us_rel = int(max(0, int(t_start_us_abs_local) - t_offset))
            t1_us_rel = int(max(0, int(t_stop_us_abs_local) - t_offset))
            t0_ms = int(t0_us_rel // 1000)
            t1_ms = int(t1_us_rel // 1000)
            t0_ms = max(0, min(t0_ms, int(ms_to_idx.shape[0] - 1)))
            t1_ms = max(0, min(t1_ms, int(ms_to_idx.shape[0] - 1)))

            i0 = int(ms_to_idx[t0_ms])
            i1 = int(ms_to_idx[t1_ms + 1]) if (t1_ms + 1) < ms_to_idx.shape[0] else int(ev["t"].shape[0])
            i0 = max(0, min(i0, int(ev["t"].shape[0])))
            i1 = max(i0, min(i1, int(ev["t"].shape[0])))

            # Refine bounds with a small local search using timestamps (only on the sliced window).
            t_slice_rel = np.asarray(ev["t"][i0:i1], dtype=np.uint64)  # us, relative to t_offset
            t_slice_abs = t_slice_rel.astype(np.int64) + int(t_offset)
            j0 = int(np.searchsorted(t_slice_abs, int(t_start_us_abs_local), side="left"))
            j1 = int(np.searchsorted(t_slice_abs, int(t_stop_us_abs_local), side="right"))
            i0 = i0 + j0
            i1 = i0 + max(0, j1 - j0)

            # Load the filtered event fields (still potentially large, but much smaller than full file).
            t_us = np.asarray(ev["t"][i0:i1], dtype=np.uint64) + np.uint64(t_offset)
            x = np.asarray(ev["x"][i0:i1], dtype=np.uint16)
            y = np.asarray(ev["y"][i0:i1], dtype=np.uint16)
            p = np.asarray(ev["p"][i0:i1], dtype=np.uint8)

        if t_us.size == 0:
            raise RuntimeError(f"No events selected for {h5_path} in [{t_start_us_abs_local}, {t_stop_us_abs_local}]")

        t_rel_ns = (t_us.astype(np.int64) - int(base_us)) * 1000
        # Avoid Time(0) which ROS tooling may treat as invalid.
        t_rel_ns = t_rel_ns - int(t_rel_ns[0]) + 1_000_000_000
        t_start_s = float(t_rel_ns[0]) / 1e9
        t_stop_s = float(t_rel_ns[-1]) / 1e9

        cam_info = make_camerainfo(intr)
        with Writer(bag_path) as writer:
            c_ev = writer.add_connection(topic_events, "dvs_msgs/msg/EventArray", typestore=ts)
            c_ci = writer.add_connection(topic_info, "sensor_msgs/msg/CameraInfo", typestore=ts)

            # Write one CameraInfo at the first event time.
            writer.write(c_ci, int(t_rel_ns[0]), ts.serialize_ros1(cam_info, "sensor_msgs/msg/CameraInfo"))

            n = t_rel_ns.shape[0]
            i = 0
            last_report = 0
            while i < n:
                j = min(i + int(max_events_per_msg), n)
                # Set header stamp to first event time in packet.
                t_ns = int(t_rel_ns[i])
                header = Header(seq=0, stamp=time_from_ns(t_ns), frame_id="camera")

                evs = []
                for k in range(i, j):
                    sec, nsec = stamp_tuple(int(t_rel_ns[k]))
                    evs.append(Event(ts=Time(sec=sec, nanosec=nsec), x=int(x[k]), y=int(y[k]), polarity=bool(p[k])))

                msg = EventArray(header=header, events=evs, height=int(h), width=int(w))
                writer.write(c_ev, t_ns, ts.serialize_ros1(msg, "dvs_msgs/msg/EventArray"))
                i = j
                if progress_every_events > 0 and (i - last_report) >= int(progress_every_events):
                    print(f"[seeded-mcemvs] wrote {i}/{n} events to {os.path.basename(bag_path)}")
                    last_report = i

        return t_start_s, t_stop_s

    # Pose bag: use DEVO pose timestamps (us) relative to first pose time.
    def write_pose_bag(bag_path: str, base_us: int) -> tuple[float, float]:
        # Keep timestamps in integer microseconds to avoid float precision issues
        # (which can create occasional negative/huge nanosecond stamps when base_us is large).
        t_us = np.asarray(devo_tstamps_us).reshape(-1)
        if np.issubdtype(t_us.dtype, np.floating):
            t_us_i = np.rint(t_us).astype(np.int64)
        else:
            t_us_i = t_us.astype(np.int64, copy=False)
        poses = np.asarray(devo_poses_c2w, dtype=np.float32)
        if t_us_i.size != poses.shape[0]:
            raise ValueError(f"tstamps {t_us_i.shape} and poses {poses.shape} mismatch")

        t_rel_ns = (t_us_i - int(base_us)) * 1000
        # Ensure non-negative, non-zero start (robust to any rounding/offset oddities).
        t_rel_ns = t_rel_ns - int(np.min(t_rel_ns)) + 1_000_000_000
        t_start_s = float(int(t_rel_ns[0])) / 1e9
        t_stop_s = float(int(t_rel_ns[-1])) / 1e9

        with Writer(bag_path) as writer:
            c_pose = writer.add_connection("/pose", "geometry_msgs/msg/PoseStamped", typestore=ts)
            for i in range(poses.shape[0]):
                t_ns = int(t_rel_ns[i])
                sec, nsec = stamp_tuple(t_ns)
                header = Header(seq=0, stamp=Time(sec=sec, nanosec=nsec), frame_id="world")
                px, py, pz, qx, qy, qz, qw = [float(x) for x in poses[i].tolist()]
                msg = PoseStamped(
                    header=header,
                    pose=Pose(
                        position=Point(x=px, y=py, z=pz),
                        orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
                    ),
                )
                writer.write(c_pose, t_ns, ts.serialize_ros1(msg, "geometry_msgs/msg/PoseStamped"))

        return t_start_s, t_stop_s

    left_h5, _ = _load_vector_events_hdf5(seq_dir, side=side)
    # Mono mode: duplicate left events into right bag (MC-EMVS code expects both).
    right_h5 = left_h5
    if mode == "stereo":
        # If you have right events, point to them here by using --side right in a second pass.
        right_h5, _ = _load_vector_events_hdf5(seq_dir, side="right")

    # Choose a common time base so data_loading.cpp's static initial timestamp works across bags.
    pose_min_us = int(np.min(np.asarray(devo_tstamps_us, dtype=np.float64)))
    pose_max_us = int(np.max(np.asarray(devo_tstamps_us, dtype=np.float64)))
    ev_start_us = int(t_start_us_abs) if t_start_us_abs is not None else pose_min_us
    ev_stop_us = int(t_stop_us_abs) if t_stop_us_abs is not None else pose_max_us
    base_us = int(min(ev_start_us, pose_min_us))

    t0s, t1s = write_events_bag(left_h5, bag_left, "/dvs/left/events", "/dvs/left/camera_info", intr_left, base_us, t_start_us_abs, t_stop_us_abs)
    _t0r, _t1r = write_events_bag(right_h5, bag_right, "/dvs/right/events", "/dvs/right/camera_info", intr_right, base_us, t_start_us_abs, t_stop_us_abs)
    tp0, tp1 = write_pose_bag(bag_pose, base_us)

    # Use overlap of pose and events for config time range.
    start_s = max(t0s, tp0)
    stop_s = min(t1s, tp1)
    if stop_s <= start_s:
        start_s, stop_s = t0s, t1s

    return BagPaths(left_events=bag_left, right_events=bag_right, poses=bag_pose), start_s, stop_s


def main():
    ap = argparse.ArgumentParser(description="Prepare seeded MC-EMVS inputs for Vector-format data using DEVO outputs.")
    ap.add_argument("--seq_dir", required=True, help="Vector sequence folder containing calib_undist_evs_*.txt and *.hdf5")
    ap.add_argument("--out_dir", required=True, help="Output folder for bags/config/calib")
    ap.add_argument("--devo_poses_npy", required=True, help="DEVO poses .npy (world->cam, xyz+quat_xyzw)")
    ap.add_argument("--devo_tstamps_npy", required=True, help="DEVO timestamps .npy (microseconds)")
    ap.add_argument("--frames_npz", default=None, help="DEVO frames.npz for seeding min/max depth (inverse depths)")
    ap.add_argument("--side", default="left", choices=["left", "right"], help="Event camera side")
    ap.add_argument("--mode", default="mono", choices=["mono", "stereo"], help="mono duplicates left->right; stereo expects both sides.")

    ap.add_argument("--H", type=int, default=480)
    ap.add_argument("--W", type=int, default=640)
    ap.add_argument("--dimZ", type=int, default=100)
    ap.add_argument("--process_method", type=int, default=2, help="1=stereo only, 2=stereo+temporal, 5=shuffled")
    ap.add_argument("--save_mono", action="store_true", help="Ask MC-EMVS to also save monocular EMVS outputs.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output bags in --out_dir.")
    ap.add_argument("--t_margin_s", type=float, default=0.5, help="Expand event time window around DEVO poses by this margin (seconds).")
    ap.add_argument("--progress_every_events", type=int, default=1_000_000, help="Print progress every N events written (0=disable).")
    ap.add_argument(
        "--absolute_paths",
        action="store_true",
        help="Write absolute file paths into seeded.conf (recommended when running from another working directory).",
    )

    ap.add_argument("--min_depth", type=float, default=None)
    ap.add_argument("--max_depth", type=float, default=None)
    args = ap.parse_args()

    intr_left = _load_intrinsics_fx_fy_cx_cy(args.seq_dir, side="left")
    intr_right = _load_intrinsics_fx_fy_cx_cy(args.seq_dir, side="right") if os.path.exists(os.path.join(args.seq_dir, "calib_undist_evs_right.txt")) else intr_left.copy()

    poses_w2c = np.load(args.devo_poses_npy).astype(np.float32)
    tstamps_us = np.load(args.devo_tstamps_npy).astype(np.float64)
    poses_c2w = _devo_world_to_cam_to_cam_to_world(poses_w2c)

    if args.frames_npz is not None and (args.min_depth is None or args.max_depth is None):
        mn, mx = _estimate_depth_range_from_frames_npz(args.frames_npz)
    else:
        mn = float(args.min_depth) if args.min_depth is not None else 0.3
        mx = float(args.max_depth) if args.max_depth is not None else 5.0

    os.makedirs(args.out_dir, exist_ok=True)
    t_pose_min = float(np.min(tstamps_us))
    t_pose_max = float(np.max(tstamps_us))
    t_start_us_abs = t_pose_min - float(args.t_margin_s) * 1e6
    t_stop_us_abs = t_pose_max + float(args.t_margin_s) * 1e6

    bags, start_s, stop_s = _write_rosbags_vector(
        seq_dir=args.seq_dir,
        out_dir=args.out_dir,
        side=args.side,
        mode=args.mode,
        devo_poses_c2w=poses_c2w,
        devo_tstamps_us=tstamps_us,
        intr_left=intr_left,
        intr_right=intr_right,
        h=int(args.H),
        w=int(args.W),
        overwrite=bool(args.overwrite),
        t_start_us_abs=float(t_start_us_abs),
        t_stop_us_abs=float(t_stop_us_abs),
        progress_every_events=int(args.progress_every_events),
    )

    calib_yaml = os.path.join(args.out_dir, "calib_seeded.yaml")
    _write_calib_yaml(calib_yaml, intr_left, intr_right, h=int(args.H), w=int(args.W), mode=args.mode)

    conf_path = os.path.join(args.out_dir, "seeded.conf")
    if args.absolute_paths:
        bag_left = os.path.abspath(bags.left_events)
        bag_right = os.path.abspath(bags.right_events)
        bag_pose = os.path.abspath(bags.poses)
        out_dir = os.path.abspath(args.out_dir)
        calib_path = os.path.abspath(calib_yaml)
    else:
        bag_left = bags.left_events
        bag_right = bags.right_events
        bag_pose = bags.poses
        out_dir = args.out_dir
        calib_path = calib_yaml
    _write_seeded_conf(
        conf_path,
        bag_left=bag_left,
        bag_right=bag_right,
        bag_pose=bag_pose,
        out_dir=out_dir,
        calib_yaml=calib_path,
        start_s=float(start_s),
        stop_s=float(stop_s),
        min_depth=float(mn),
        max_depth=float(mx),
        dimZ=int(args.dimZ),
        process_method=int(args.process_method),
        save_mono=bool(args.save_mono),
    )

    print("[seeded-mcemvs] wrote:")
    print("  ", bags.left_events)
    print("  ", bags.right_events)
    print("  ", bags.poses)
    print("  ", calib_yaml)
    print("  ", conf_path)
    print(f"[seeded-mcemvs] suggested time window: start_time_s={start_s:.3f} stop_time_s={stop_s:.3f}")
    print(f"[seeded-mcemvs] seeded depth range: min_depth={mn:.3f} max_depth={mx:.3f} (dimZ={int(args.dimZ)})")


if __name__ == "__main__":
    main()
