from __future__ import annotations

import argparse
import os

import numpy as np


def _load_vector_events_hdf5(seq_dir: str, side: str) -> str:
    seq = os.path.basename(os.path.normpath(seq_dir))
    h5_path = os.path.join(seq_dir, f"{seq}1.synced.{side}_event.hdf5")
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"Vector events file not found: {h5_path}")
    return h5_path


def _load_intrinsics(seq_dir: str, side: str) -> np.ndarray:
    path = os.path.join(seq_dir, f"calib_undist_evs_{side}.txt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Intrinsics file not found: {path}")
    intr = np.loadtxt(path).astype(np.float32).reshape(-1)
    if intr.size != 4:
        raise ValueError(f"Expected intrinsics [fx, fy, cx, cy], got {intr.size} values from {path}")
    return intr


def _load_camera_calibration(seq_dir: str, side: str) -> dict[str, np.ndarray | str]:
    yaml_path = os.path.join(seq_dir, "..", "0_calib", f"{side}_event_camera_intrinsic_results.yaml")
    if os.path.isfile(yaml_path):
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return {
            "distortion_model": str(data.get("camera_model", "plumb_bob")),
            "D": np.asarray(data["distortion_coefficients"]["data"], dtype=np.float64).reshape(-1),
            "K": np.asarray(data["camera_matrix"]["data"], dtype=np.float64).reshape(9),
            "R": np.asarray(data["rectification_matrix"]["data"], dtype=np.float64).reshape(9),
            "P": np.asarray(data["projection_matrix"]["data"], dtype=np.float64).reshape(12),
        }

    fx, fy, cx, cy = [float(x) for x in _load_intrinsics(seq_dir, side).reshape(4)]
    return {
        "distortion_model": "plumb_bob",
        "D": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "K": np.asarray([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0], dtype=np.float64),
        "R": np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "P": np.asarray([fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
    }


def _estimate_depth_range_from_frames_npz(frames_npz: str) -> tuple[float, float]:
    data = np.load(frames_npz, allow_pickle=False)
    if "depths" not in data:
        raise ValueError(f"{frames_npz} does not contain 'depths' (inverse depth seeds).")
    inv = np.asarray(data["depths"], dtype=np.float32).reshape(-1)
    inv = inv[np.isfinite(inv) & (inv > 0)]
    if inv.size == 0:
        raise ValueError(f"No valid inverse depths found in {frames_npz}")

    z = 1.0 / inv
    z = z[np.isfinite(z) & (z > 0)]
    lo = float(np.percentile(z, 5))
    hi = float(np.percentile(z, 95))

    lo = max(1e-3, 0.8 * lo)
    hi = 1.2 * hi
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _devo_world_to_cam_to_cam_to_world(poses_w2c_xyzw: np.ndarray) -> np.ndarray:
    """Convert DEVO world->cam poses into cam->world poses expected by EMVS.

    Input and output pose layout is [tx, ty, tz, qx, qy, qz, qw].
    """
    poses = np.asarray(poses_w2c_xyzw, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError(f"Expected poses shape (N,7), got {poses.shape}")

    t = poses[:, :3]
    q = poses[:, 3:]  # xyzw for world->cam

    q_inv = q.copy()
    q_inv[:, :3] *= -1.0

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

    cx = -(r00 * t[:, 0] + r01 * t[:, 1] + r02 * t[:, 2])
    cy = -(r10 * t[:, 0] + r11 * t[:, 1] + r12 * t[:, 2])
    cz = -(r20 * t[:, 0] + r21 * t[:, 1] + r22 * t[:, 2])

    return np.stack([cx, cy, cz, q_inv[:, 0], q_inv[:, 1], q_inv[:, 2], q_inv[:, 3]], axis=1).astype(np.float32)


def _write_emvs_conf(
    out_path: str,
    bag_filename: str,
    event_topic: str,
    camera_info_topic: str,
    pose_topic: str,
    start_time_s: float,
    stop_time_s: float,
    min_depth: float,
    max_depth: float,
    dim_z: int,
) -> None:
    lines = [
        f"--bag_filename={bag_filename}\n",
        f"--event_topic={event_topic}\n",
        f"--camera_info_topic={camera_info_topic}\n",
        f"--pose_topic={pose_topic}\n",
        f"--start_time_s={float(start_time_s)}\n",
        f"--stop_time_s={float(stop_time_s)}\n",
        f"--min_depth={float(min_depth)}\n",
        f"--max_depth={float(max_depth)}\n",
        f"--dimZ={int(dim_z)}\n",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare a single monocular EMVS bag from Vector events + DEVO poses.")
    ap.add_argument("--seq_dir", required=True, help="Vector sequence folder.")
    ap.add_argument("--devo_poses_npy", required=True, help="DEVO poses .npy in world->cam format (N,7).")
    ap.add_argument("--devo_tstamps_npy", required=True, help="DEVO timestamps .npy in microseconds (N,).")
    ap.add_argument(
        "--no_pose_conversion",
        action="store_true",
        help="Use input poses as-is (assume they are already cam->world) and skip world->cam to cam->world conversion.",
    )
    ap.add_argument("--out_bag", required=True, help="Output ROS1 bag file path (single bag for mapper_emvs).")
    ap.add_argument("--out_conf", default=None, help="Optional output .conf path for mapper_emvs.")
    ap.add_argument("--frames_npz", default=None, help="Optional DEVO frames.npz to auto-estimate min/max depth.")

    ap.add_argument("--side", default="left", choices=["left", "right"], help="Vector event camera side.")
    ap.add_argument("--H", type=int, default=480)
    ap.add_argument("--W", type=int, default=640)

    ap.add_argument("--event_topic", default="/dvs/events")
    ap.add_argument("--camera_info_topic", default="/dvs/camera_info")
    ap.add_argument("--pose_topic", default="/pose")

    ap.add_argument("--min_depth", type=float, default=None)
    ap.add_argument("--max_depth", type=float, default=None)
    ap.add_argument("--dimZ", type=int, default=100)

    ap.add_argument("--t_margin_s", type=float, default=0.5, help="Expand event window around pose time range.")
    ap.add_argument("--max_events_per_msg", type=int, default=20000)
    ap.add_argument("--progress_every_events", type=int, default=1_000_000)
    ap.add_argument(
        "--eventarray_schema",
        choices=["5e8", "732"],
        default="5e8",
        help=(
            "dvs_msgs/EventArray field layout for /dvs/events. "
            "'5e8' => md5 5e8beee5a6c107e504c2e78903c224b8 "
            "(Header, height, width, events). "
            "'732' => md5 732c9cc0676a970b9aa2873b0d444ac2 "
            "(Header, events, height, width)."
        ),
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    try:
        import os as _os

        plugin_path = _os.environ.get("HDF5_PLUGIN_PATH", "")
        if plugin_path and not _os.path.exists(plugin_path):
            _os.environ.pop("HDF5_PLUGIN_PATH", None)

        import hdf5plugin  # noqa: F401
        import h5py
        from rosbags.rosbag1 import Writer
        from rosbags.typesys import Stores, get_typestore, get_types_from_msg
    except Exception as exc:
        raise RuntimeError("Missing deps. Install with: /opt/conda/envs/devo/bin/pip install rosbags h5py") from exc

    out_bag = os.path.abspath(args.out_bag)
    os.makedirs(os.path.dirname(out_bag) or ".", exist_ok=True)
    if os.path.exists(out_bag) and not args.overwrite:
        raise FileExistsError(f"Output bag exists: {out_bag}. Use --overwrite to replace it.")
    if os.path.exists(out_bag) and args.overwrite:
        os.remove(out_bag)

    h5_path = _load_vector_events_hdf5(args.seq_dir, args.side)
    camera_calib = _load_camera_calibration(args.seq_dir, args.side)

    poses_w2c = np.load(args.devo_poses_npy).astype(np.float32)
    tstamps_us = np.load(args.devo_tstamps_npy).astype(np.float64).reshape(-1)
    if tstamps_us.ndim != 1:
        raise ValueError(f"Expected 1D timestamps, got {tstamps_us.shape}")
    if poses_w2c.shape[0] != tstamps_us.shape[0]:
        raise ValueError(f"Mismatch: poses {poses_w2c.shape} vs timestamps {tstamps_us.shape}")
    if args.no_pose_conversion:
        poses_c2w = poses_w2c
    else:
        poses_c2w = _devo_world_to_cam_to_cam_to_world(poses_w2c)

    if args.frames_npz and (args.min_depth is None or args.max_depth is None):
        min_depth, max_depth = _estimate_depth_range_from_frames_npz(args.frames_npz)
    else:
        min_depth = float(args.min_depth) if args.min_depth is not None else 0.3
        max_depth = float(args.max_depth) if args.max_depth is not None else 5.0

    pose_min_us = int(np.min(tstamps_us))
    pose_max_us = int(np.max(tstamps_us))
    t_start_us_abs = int(pose_min_us - float(args.t_margin_s) * 1e6)
    t_stop_us_abs = int(pose_max_us + float(args.t_margin_s) * 1e6)

    ts = get_typestore(Stores.ROS1_NOETIC)
    if args.eventarray_schema == "5e8":
        event_msg = "uint16 x\nuint16 y\ntime ts\nbool polarity\n"
        event_array_msg = "std_msgs/Header header\nuint32 height\nuint32 width\ndvs_msgs/Event[] events\n"
    else:
        event_msg = "time ts\nuint16 x\nuint16 y\nbool polarity\n"
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
    RegionOfInterest = ts.get_msgdef("sensor_msgs/msg/RegionOfInterest").cls

    def time_from_ns(ns: int) -> Time:
        return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))

    def make_camera_info(first_stamp_ns: int) -> CameraInfo:
        return CameraInfo(
            header=Header(seq=0, stamp=time_from_ns(first_stamp_ns), frame_id=f"{args.side}_event_camera"),
            height=int(args.H),
            width=int(args.W),
            distortion_model=str(camera_calib["distortion_model"]),
            D=np.asarray(camera_calib["D"], dtype=np.float64),
            K=np.asarray(camera_calib["K"], dtype=np.float64),
            R=np.asarray(camera_calib["R"], dtype=np.float64),
            P=np.asarray(camera_calib["P"], dtype=np.float64),
            binning_x=0,
            binning_y=0,
            roi=RegionOfInterest(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False),
        )

    with h5py.File(h5_path, "r") as f:
        t_offset = int(np.asarray(f["t_offset"])[0])
        ev = f["events"]
        ms_to_idx = f["ms_to_idx"]

        t0_us_rel = int(max(0, t_start_us_abs - t_offset))
        t1_us_rel = int(max(0, t_stop_us_abs - t_offset))

        t0_ms = int(max(0, min(t0_us_rel // 1000, int(ms_to_idx.shape[0] - 1))))
        t1_ms = int(max(0, min(t1_us_rel // 1000, int(ms_to_idx.shape[0] - 1))))

        i0 = int(ms_to_idx[t0_ms])
        i1 = int(ms_to_idx[t1_ms + 1]) if (t1_ms + 1) < ms_to_idx.shape[0] else int(ev["t"].shape[0])
        i0 = max(0, min(i0, int(ev["t"].shape[0])))
        i1 = max(i0, min(i1, int(ev["t"].shape[0])))

        t_slice_rel = np.asarray(ev["t"][i0:i1], dtype=np.uint64)
        t_slice_abs = t_slice_rel.astype(np.int64) + int(t_offset)

        j0 = int(np.searchsorted(t_slice_abs, t_start_us_abs, side="left"))
        j1 = int(np.searchsorted(t_slice_abs, t_stop_us_abs, side="right"))
        i0 = i0 + j0
        i1 = i0 + max(0, j1 - j0)

        t_us_abs = np.asarray(ev["t"][i0:i1], dtype=np.uint64) + np.uint64(t_offset)
        x = np.asarray(ev["x"][i0:i1], dtype=np.uint16)
        y = np.asarray(ev["y"][i0:i1], dtype=np.uint16)
        p = np.asarray(ev["p"][i0:i1], dtype=np.uint8)

    if t_us_abs.size == 0:
        raise RuntimeError("No events selected in requested time window.")

    base_us = min(int(np.min(t_us_abs)), pose_min_us)
    event_t_ns = (t_us_abs.astype(np.int64) - base_us) * 1000 + 1_000_000_000
    pose_t_ns = (np.rint(tstamps_us).astype(np.int64) - base_us) * 1000 + 1_000_000_000

    start_ns = int(min(event_t_ns[0], np.min(pose_t_ns)))
    stop_ns = int(max(event_t_ns[-1], np.max(pose_t_ns)))

    with Writer(out_bag) as writer:
        c_ev = writer.add_connection(args.event_topic, "dvs_msgs/msg/EventArray", typestore=ts)
        c_ci = writer.add_connection(args.camera_info_topic, "sensor_msgs/msg/CameraInfo", typestore=ts)
        c_pose = writer.add_connection(args.pose_topic, "geometry_msgs/msg/PoseStamped", typestore=ts)

        cam_info = make_camera_info(start_ns)
        writer.write(c_ci, start_ns, ts.serialize_ros1(cam_info, "sensor_msgs/msg/CameraInfo"))

        n_events = event_t_ns.shape[0]
        i = 0
        last_report = 0
        while i < n_events:
            j = min(i + int(args.max_events_per_msg), n_events)
            pkt_t_ns = int(event_t_ns[i])
            header = Header(seq=0, stamp=time_from_ns(pkt_t_ns), frame_id=f"{args.side}_event_camera")
            evs = []
            for k in range(i, j):
                evs.append(
                    Event(
                        x=int(x[k]),
                        y=int(y[k]),
                        ts=time_from_ns(int(event_t_ns[k])),
                        polarity=bool(p[k]),
                    )
                )

            if args.eventarray_schema == "5e8":
                msg = EventArray(header=header, height=int(args.H), width=int(args.W), events=evs)
            else:
                msg = EventArray(header=header, events=evs, height=int(args.H), width=int(args.W))
            writer.write(c_ev, pkt_t_ns, ts.serialize_ros1(msg, "dvs_msgs/msg/EventArray"))

            i = j
            if args.progress_every_events > 0 and (i - last_report) >= int(args.progress_every_events):
                print(f"[emvs-mono] wrote {i}/{n_events} events")
                last_report = i

        for i in range(poses_c2w.shape[0]):
            t_ns = int(pose_t_ns[i])
            px, py, pz, qx, qy, qz, qw = [float(v) for v in poses_c2w[i].tolist()]
            header = Header(seq=0, stamp=time_from_ns(t_ns), frame_id="world")
            msg = PoseStamped(
                header=header,
                pose=Pose(
                    position=Point(x=px, y=py, z=pz),
                    orientation=Quaternion(x=qx, y=qy, z=qz, w=qw),
                ),
            )
            writer.write(c_pose, t_ns, ts.serialize_ros1(msg, "geometry_msgs/msg/PoseStamped"))

    conf_path = os.path.abspath(args.out_conf) if args.out_conf else os.path.splitext(out_bag)[0] + ".conf"
    os.makedirs(os.path.dirname(conf_path) or ".", exist_ok=True)
    _write_emvs_conf(
        out_path=conf_path,
        bag_filename=out_bag,
        event_topic=args.event_topic,
        camera_info_topic=args.camera_info_topic,
        pose_topic=args.pose_topic,
        start_time_s=0.0,
        stop_time_s=float(stop_ns - start_ns) / 1e9,
        min_depth=min_depth,
        max_depth=max_depth,
        dim_z=int(args.dimZ),
    )

    print("[emvs-mono] wrote bag:", out_bag)
    print("[emvs-mono] wrote conf:", conf_path)
    if args.no_pose_conversion:
        print("[emvs-mono] pose convention: input used as-is (no conversion)")
    else:
        print("[emvs-mono] pose convention: input world->cam converted to cam->world")
    if args.eventarray_schema == "5e8":
        print("[emvs-mono] /dvs/events schema: EventArray md5 5e8beee5a6c107e504c2e78903c224b8")
    else:
        print("[emvs-mono] /dvs/events schema: EventArray md5 732c9cc0676a970b9aa2873b0d444ac2")
    print(f"[emvs-mono] depth range: min_depth={min_depth:.3f}, max_depth={max_depth:.3f}, dimZ={int(args.dimZ)}")
    print("[emvs-mono] run:")
    print(f"  rosrun mapper_emvs run_emvs --flagfile={conf_path}")


if __name__ == "__main__":
    main()
