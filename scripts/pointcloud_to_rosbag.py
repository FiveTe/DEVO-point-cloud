from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _as_float32_xyz(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected point array shape (N,3), got {pts.shape}")
    pts = pts.astype(np.float32, copy=False)
    valid = np.isfinite(pts).all(axis=1)
    return pts[valid]


def _load_frames_from_npz(npz_path: str, include_empty: bool) -> tuple[list[np.ndarray], np.ndarray]:
    data = np.load(npz_path, allow_pickle=False)

    if {"counts", "offsets", "points"}.issubset(set(data.files)):
        counts = np.asarray(data["counts"], dtype=np.int64).reshape(-1)
        offsets = np.asarray(data["offsets"], dtype=np.int64).reshape(-1)
        all_points = np.asarray(data["points"], dtype=np.float32)
        if all_points.ndim != 2 or all_points.shape[1] != 3:
            raise ValueError(f"'points' in {npz_path} must be (N,3), got {all_points.shape}")

        if "timestamps" in data.files:
            ts = np.asarray(data["timestamps"], dtype=np.float64).reshape(-1)
        elif "tstamps" in data.files:
            ts = np.asarray(data["tstamps"], dtype=np.float64).reshape(-1)
        else:
            ts = np.arange(len(counts), dtype=np.float64)

        if not (len(counts) == len(offsets) == len(ts)):
            raise ValueError(
                f"Mismatch in frame arrays: len(counts)={len(counts)}, len(offsets)={len(offsets)}, len(timestamps)={len(ts)}"
            )

        frames: list[np.ndarray] = []
        times: list[float] = []
        for i in range(len(counts)):
            c = int(counts[i])
            o = int(offsets[i])
            if c < 0 or o < 0 or (o + c) > all_points.shape[0]:
                raise ValueError(
                    f"Invalid slice for frame {i}: offset={o}, count={c}, total_points={all_points.shape[0]}"
                )
            pts = _as_float32_xyz(all_points[o : o + c])
            if pts.shape[0] == 0 and not include_empty:
                continue
            frames.append(pts)
            times.append(float(ts[i]))

        if len(frames) == 0:
            return [], np.zeros((0,), dtype=np.float64)
        return frames, np.asarray(times, dtype=np.float64)

    if "points" in data.files:
        pts = _as_float32_xyz(np.asarray(data["points"], dtype=np.float32))
        return [pts], np.asarray([0.0], dtype=np.float64)

    raise ValueError(
        f"Unsupported npz format in {npz_path}. Expected keys (counts, offsets, points[, timestamps]) or at least 'points'."
    )


def _load_frames_from_npy(npy_path: str, include_empty: bool) -> tuple[list[np.ndarray], np.ndarray]:
    arr = np.load(npy_path)

    if arr.ndim == 2 and arr.shape[1] == 3:
        pts = _as_float32_xyz(arr)
        if pts.shape[0] == 0 and not include_empty:
            return [], np.zeros((0,), dtype=np.float64)
        return [pts], np.asarray([0.0], dtype=np.float64)

    if arr.ndim == 3 and arr.shape[2] == 3:
        frames = []
        times = []
        for i in range(arr.shape[0]):
            pts = _as_float32_xyz(arr[i])
            if pts.shape[0] == 0 and not include_empty:
                continue
            frames.append(pts)
            times.append(float(i))
        if len(frames) == 0:
            return [], np.zeros((0,), dtype=np.float64)
        return frames, np.asarray(times, dtype=np.float64)

    raise ValueError(f"Unsupported npy format in {npy_path}. Expected (N,3) or (F,N,3), got {arr.shape}")


def _load_frames_from_ply(ply_path: str, include_empty: bool) -> tuple[list[np.ndarray], np.ndarray]:
    try:
        from plyfile import PlyData
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'plyfile'. Install with: /opt/conda/envs/devo/bin/pip install plyfile"
        ) from exc

    ply_data = PlyData.read(ply_path)
    vertex = ply_data["vertex"]

    if not all(k in vertex.data.dtype.names for k in ["x", "y", "z"]):
        raise ValueError(
            f"PLY file {ply_path} missing required 'x', 'y', 'z' fields. Found: {list(vertex.data.dtype.names)}"
        )

    pts = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    pts = _as_float32_xyz(pts)
    if pts.shape[0] == 0 and not include_empty:
        return [], np.zeros((0,), dtype=np.float64)
    return [pts], np.asarray([0.0], dtype=np.float64)


def _load_frames_from_pcd(pcd_path: str, include_empty: bool) -> tuple[list[np.ndarray], np.ndarray]:
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'open3d'. Install with: /opt/conda/envs/devo/bin/pip install open3d"
        ) from exc

    pcd = o3d.io.read_point_cloud(pcd_path)
    if pcd.has_points() == 0:
        if not include_empty:
            return [], np.zeros((0,), dtype=np.float64)
        return [np.zeros((0, 3), dtype=np.float32)], np.asarray([0.0], dtype=np.float64)

    pts = np.asarray(pcd.points, dtype=np.float32)
    pts = _as_float32_xyz(pts)
    if pts.shape[0] == 0 and not include_empty:
        return [], np.zeros((0,), dtype=np.float64)
    return [pts], np.asarray([0.0], dtype=np.float64)


def _load_trajectory_from_txt(txt_path: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(txt_path, comments="#", ndmin=2)
    if data.size == 0:
        return np.zeros((0, 7), dtype=np.float32), np.zeros((0,), dtype=np.float64)

    if data.shape[1] != 8:
        raise ValueError(
            f"Expected trajectory txt with 8 columns [timestamp tx ty tz qx qy qz qw], got {data.shape[1]} in {txt_path}"
        )

    timestamps = data[:, 0].astype(np.float64, copy=False)
    poses = data[:, 1:].astype(np.float32, copy=False)
    if poses.shape[1] != 7:
        raise ValueError(f"Expected poses shape (N,7), got {poses.shape} in {txt_path}")
    return poses, timestamps


def _load_frames(input_path: str, timestamps_npy: str | None, include_empty: bool) -> tuple[list[np.ndarray], np.ndarray]:
    suffix = Path(input_path).suffix.lower()
    if suffix == ".npz":
        frames, ts = _load_frames_from_npz(input_path, include_empty=include_empty)
    elif suffix == ".npy":
        frames, ts = _load_frames_from_npy(input_path, include_empty=include_empty)
    elif suffix == ".ply":
        frames, ts = _load_frames_from_ply(input_path, include_empty=include_empty)
    elif suffix == ".pcd":
        frames, ts = _load_frames_from_pcd(input_path, include_empty=include_empty)
    else:
        raise ValueError(f"Unsupported input extension '{suffix}'. Use .npy, .npz, .ply, or .pcd")

    if timestamps_npy is not None:
        ts_override = np.load(timestamps_npy).reshape(-1).astype(np.float64)
        if ts_override.shape[0] != len(frames):
            raise ValueError(
                f"timestamps override has {ts_override.shape[0]} values but there are {len(frames)} frames"
            )
        ts = ts_override

    return frames, ts


def _load_any_input(input_path: str, timestamps_npy: str | None, include_empty: bool):
    suffix = Path(input_path).suffix.lower()
    if suffix == ".txt":
        poses, timestamps = _load_trajectory_from_txt(input_path)
        if timestamps_npy is not None:
            ts_override = np.load(timestamps_npy).reshape(-1).astype(np.float64)
            if ts_override.shape[0] != poses.shape[0]:
                raise ValueError(
                    f"timestamps override has {ts_override.shape[0]} values but there are {poses.shape[0]} poses"
                )
            timestamps = ts_override
        return "trajectory", poses, timestamps, "s"

    frames, timestamps = _load_frames(input_path, timestamps_npy, include_empty=include_empty)
    return "cloud", frames, timestamps, "auto"


def _looks_like_microseconds(timestamps: np.ndarray) -> bool:
    if timestamps.size == 0:
        return True
    mx = float(np.max(np.abs(timestamps)))
    return mx > 1.0e5


def _normalize_to_ros_ns(timestamps: np.ndarray, unit: str) -> np.ndarray:
    if timestamps.size == 0:
        return np.zeros((0,), dtype=np.int64)

    t = timestamps.astype(np.float64, copy=False)
    t = t - float(np.min(t))

    if unit == "auto":
        unit = "us" if _looks_like_microseconds(t) else "s"

    if unit == "us":
        ns = np.rint(t * 1_000.0).astype(np.int64)
    elif unit == "s":
        ns = np.rint(t * 1_000_000_000.0).astype(np.int64)
    else:
        raise ValueError(f"Unsupported --timestamp_unit '{unit}'. Use auto|us|s")

    # Avoid zero stamps: some ROS tools dislike Time(0).
    return ns - int(np.min(ns)) + 1_000_000_000


def _time_msg_from_ns(ts, ns: int):
    Time = ts.get_msgdef("builtin_interfaces/msg/Time").cls
    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def _make_cloud_msg(ts, topic_frame_id: str, points_xyz: np.ndarray, stamp_ns: int):
    Header = ts.get_msgdef("std_msgs/msg/Header").cls
    PointCloud2 = ts.get_msgdef("sensor_msgs/msg/PointCloud2").cls
    PointField = ts.get_msgdef("sensor_msgs/msg/PointField").cls

    pts = _as_float32_xyz(points_xyz)
    n = int(pts.shape[0])

    fields = [
        PointField(name="x", offset=0, datatype=7, count=1),  # FLOAT32
        PointField(name="y", offset=4, datatype=7, count=1),
        PointField(name="z", offset=8, datatype=7, count=1),
    ]

    point_step = 12
    row_step = n * point_step
    raw = np.frombuffer(pts.tobytes(order="C"), dtype=np.uint8).copy()

    msg = PointCloud2(
        header=Header(seq=0, stamp=_time_msg_from_ns(ts, stamp_ns), frame_id=topic_frame_id),
        height=1,
        width=n,
        fields=fields,
        is_bigendian=False,
        point_step=point_step,
        row_step=row_step,
        data=raw,
        is_dense=True,
    )
    return msg


def _make_pose_msg(ts, topic_frame_id: str, pose_xyz_qxyzw: np.ndarray, stamp_ns: int):
    Header = ts.get_msgdef("std_msgs/msg/Header").cls
    PoseStamped = ts.get_msgdef("geometry_msgs/msg/PoseStamped").cls
    Pose = ts.get_msgdef("geometry_msgs/msg/Pose").cls
    Point = ts.get_msgdef("geometry_msgs/msg/Point").cls
    Quaternion = ts.get_msgdef("geometry_msgs/msg/Quaternion").cls

    pose = np.asarray(pose_xyz_qxyzw, dtype=np.float32).reshape(-1)
    if pose.size != 7:
        raise ValueError(f"Expected pose shape (7,), got {pose.shape}")

    return PoseStamped(
        header=Header(seq=0, stamp=_time_msg_from_ns(ts, stamp_ns), frame_id=topic_frame_id),
        pose=Pose(
            position=Point(x=float(pose[0]), y=float(pose[1]), z=float(pose[2])),
            orientation=Quaternion(x=float(pose[3]), y=float(pose[4]), z=float(pose[5]), w=float(pose[6])),
        ),
    )


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Convert point cloud outputs or ground-truth trajectories into a ROS1 bag.\n"
            "Supported formats:\n"
            "  .npy    - NumPy array (N,3) or (F,N,3)\n"
            "  .npz    - DEVO per-frame export with counts/offsets/points\n"
            "  .ply    - PLY mesh/point cloud (reads x,y,z fields)\n"
            "  .pcd    - PCD point cloud (via Open3D)\n"
            "  .txt    - Ground-truth trajectory with [timestamp tx ty tz qx qy qz qw]"
        )
    )
    ap.add_argument("--input", required=True, help="Input file (.npy, .npz, .ply, .pcd, or .txt)")
    ap.add_argument("--output_bag", required=True, help="Output ROS1 bag path (.bag)")
    ap.add_argument("--topic", default="/devo/pointcloud", help="PointCloud2 topic name")
    ap.add_argument("--frame_id", default="world", help="header.frame_id for PointCloud2")
    ap.add_argument(
        "--timestamp_unit",
        choices=["auto", "us", "s"],
        default="auto",
        help="Unit for timestamps from input data or --timestamps_npy",
    )
    ap.add_argument(
        "--timestamps_npy",
        default=None,
        help="Optional .npy overrides timestamps (one value per frame).",
    )
    ap.add_argument(
        "--include_empty",
        action="store_true",
        help="Include empty frames as width=0 PointCloud2 messages (default: skip empty frames).",
    )
    ap.add_argument(
        "--emvs_style",
        action="store_true",
        help=(
            "For .txt trajectory input, write an EMVS-style bag layout matching prepare_vector_emvs_mono.py: "
            "/dvs/events, /dvs/camera_info, and /pose. A placeholder empty EventArray is written so the bag "
            "contains the same topics, even though real event data is not available."
        ),
    )
    ap.add_argument(
        "--H",
        type=int,
        default=480,
        help="Image height for EMVS-style placeholder /dvs/events and /dvs/camera_info (default: 480)",
    )
    ap.add_argument(
        "--W",
        type=int,
        default=640,
        help="Image width for EMVS-style placeholder /dvs/events and /dvs/camera_info (default: 640)",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output bag if it exists")
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    input_path = os.path.abspath(args.input)
    output_bag = os.path.abspath(args.output_bag)

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    out_dir = os.path.dirname(output_bag) or "."
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(output_bag):
        if args.overwrite:
            os.remove(output_bag)
        else:
            raise FileExistsError(f"Output bag already exists: {output_bag}. Use --overwrite to replace it.")

    try:
        from rosbags.rosbag1 import Writer
        from rosbags.typesys import Stores, get_typestore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency 'rosbags'. Install with: /opt/conda/envs/devo/bin/pip install rosbags"
        ) from exc

    input_kind, payload, timestamps, default_unit = _load_any_input(
        input_path, args.timestamps_npy, include_empty=args.include_empty
    )

    if input_kind == "cloud" and len(payload) == 0:
        raise RuntimeError("No frames to write (all empty or invalid).")
    if input_kind == "trajectory" and payload.shape[0] == 0:
        raise RuntimeError("No poses to write (empty trajectory).")

    timestamp_unit = args.timestamp_unit
    if timestamp_unit == "auto":
        timestamp_unit = default_unit

    stamps_ns = _normalize_to_ros_ns(timestamps, unit=timestamp_unit)

    ts = get_typestore(Stores.ROS1_NOETIC)

    with Writer(output_bag) as writer:
        if input_kind == "cloud":
            conn = writer.add_connection(args.topic, "sensor_msgs/msg/PointCloud2", typestore=ts)
            for i, (pts, stamp_ns) in enumerate(zip(payload, stamps_ns)):
                msg = _make_cloud_msg(ts, args.frame_id, pts, int(stamp_ns))
                writer.write(conn, int(stamp_ns), ts.serialize_ros1(msg, "sensor_msgs/msg/PointCloud2"))
                if (i + 1) % 100 == 0:
                    print(f"[pointcloud_to_rosbag] wrote {i + 1}/{len(payload)} messages")
        else:
            if args.emvs_style:
                try:
                    from rosbags.typesys import get_types_from_msg
                except Exception as exc:
                    raise RuntimeError("Missing rosbags type helpers for EMVS-style output.") from exc

                event_msg = "uint16 x\nuint16 y\ntime ts\nbool polarity\n"
                event_array_msg = "std_msgs/Header header\nuint32 height\nuint32 width\ndvs_msgs/Event[] events\n"
                ts.register(get_types_from_msg(event_msg, "dvs_msgs/msg/Event"))
                ts.register(get_types_from_msg(event_array_msg, "dvs_msgs/msg/EventArray"))

                EventArray = ts.get_msgdef("dvs_msgs/msg/EventArray").cls
                Header = ts.get_msgdef("std_msgs/msg/Header").cls
                Time = ts.get_msgdef("builtin_interfaces/msg/Time").cls

                def _time_from_ns(ns: int):
                    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))

                c_ev = writer.add_connection("/dvs/events", "dvs_msgs/msg/EventArray", typestore=ts)
                c_ci = writer.add_connection("/dvs/camera_info", "sensor_msgs/msg/CameraInfo", typestore=ts)
                c_pose = writer.add_connection("/pose", "geometry_msgs/msg/PoseStamped", typestore=ts)

                # CameraInfo matches the EMVS writer shape: a single message at the first timestamp.
                first_stamp_ns = int(stamps_ns[0])
                cam_info = ts.get_msgdef("sensor_msgs/msg/CameraInfo").cls(
                    header=Header(seq=0, stamp=_time_from_ns(first_stamp_ns), frame_id="camera"),
                    height=1,
                    width=1,
                    distortion_model="plumb_bob",
                    D=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
                    K=np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                    R=np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                    P=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64),
                    binning_x=0,
                    binning_y=0,
                    roi=ts.get_msgdef("sensor_msgs/msg/RegionOfInterest").cls(
                        x_offset=0, y_offset=0, height=0, width=0, do_rectify=False
                    ),
                )
                writer.write(c_ci, first_stamp_ns, ts.serialize_ros1(cam_info, "sensor_msgs/msg/CameraInfo"))

                # Placeholder empty /dvs/events message so the topic exists like in prepare_vector_emvs_mono.py.
                empty_event_header = Header(seq=0, stamp=_time_from_ns(first_stamp_ns), frame_id="camera")
                empty_events = EventArray(header=empty_event_header, height=int(args.H), width=int(args.W), events=[])
                writer.write(c_ev, first_stamp_ns, ts.serialize_ros1(empty_events, "dvs_msgs/msg/EventArray"))

                for i, (pose, stamp_ns) in enumerate(zip(payload, stamps_ns)):
                    msg = _make_pose_msg(ts, "world", pose, int(stamp_ns))
                    writer.write(c_pose, int(stamp_ns), ts.serialize_ros1(msg, "geometry_msgs/msg/PoseStamped"))
                    if (i + 1) % 100 == 0:
                        print(f"[pointcloud_to_rosbag] wrote {i + 1}/{len(payload)} poses")
            else:
                conn = writer.add_connection(args.topic, "geometry_msgs/msg/PoseStamped", typestore=ts)
                for i, (pose, stamp_ns) in enumerate(zip(payload, stamps_ns)):
                    msg = _make_pose_msg(ts, args.frame_id, pose, int(stamp_ns))
                    writer.write(conn, int(stamp_ns), ts.serialize_ros1(msg, "geometry_msgs/msg/PoseStamped"))
                    if (i + 1) % 100 == 0:
                        print(f"[pointcloud_to_rosbag] wrote {i + 1}/{len(payload)} poses")

    print("[pointcloud_to_rosbag] wrote bag:", output_bag)
    print("[pointcloud_to_rosbag] topic:", args.topic)
    print("[pointcloud_to_rosbag] frame_id:", args.frame_id)
    print("[pointcloud_to_rosbag] messages:", len(payload))


if __name__ == "__main__":
    main()
