#!/usr/bin/env python3
"""
Rewrite a ROS1 bag with sanitized timestamps (record + embedded message time fields).

This is intended to fix bags that `rosbag info` can open but cannot compute start/end
because at least one timestamp is out-of-range.

Strategy:
  1) Scan message record times and find a "sane" minimum timestamp (uint32 sec).
  2) Rewrite all messages with new_time = old_time - min_sane_time.
  3) Clamp/repair any remaining outliers and enforce monotonic non-decreasing times.
  4) For known message types used by seeded_mcemvs, also shift embedded Time fields.

Works without ROS; uses the `rosbags` Python package.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


UINT32_MAX = 0xFFFFFFFF


def _make_typestore():
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    ts = get_typestore(Stores.ROS1_NOETIC)

    event_msg = "time ts\nuint16 x\nuint16 y\nbool polarity\n"
    event_array_msg = "std_msgs/Header header\ndvs_msgs/Event[] events\nuint32 height\nuint32 width\n"
    ts.register(get_types_from_msg(event_msg, "dvs_msgs/msg/Event"))
    ts.register(get_types_from_msg(event_array_msg, "dvs_msgs/msg/EventArray"))

    return ts


def _ns_to_sec_nsec(t_ns: int) -> tuple[int, int]:
    sec = int(t_ns // 1_000_000_000)
    nsec = int(t_ns % 1_000_000_000)
    return sec, nsec


def _is_sane_ns(t_ns: int) -> bool:
    # ROS tools (rosbag_storage) can reject Time(0) as "invalid".
    if t_ns <= 0:
        return False
    sec, nsec = _ns_to_sec_nsec(t_ns)
    return 0 <= nsec < 1_000_000_000 and 0 <= sec <= UINT32_MAX


def _time_obj_to_ns(t: Any) -> int | None:
    sec = getattr(t, "sec", None)
    nsec = getattr(t, "nanosec", None)
    if sec is None or nsec is None:
        return None
    try:
        return int(sec) * 1_000_000_000 + int(nsec)
    except Exception:
        return None


def _set_time_obj_from_ns(t: Any, t_ns: int) -> None:
    sec, nsec = _ns_to_sec_nsec(int(t_ns))
    setattr(t, "sec", int(sec))
    setattr(t, "nanosec", int(nsec))


@dataclass
class RewriteStats:
    messages_total: int = 0
    messages_time_clamped: int = 0
    messages_time_nonmonotonic: int = 0
    embedded_time_shifted: int = 0
    embedded_time_skipped: int = 0


def _shift_embedded_times(ts, msg: Any, msgtype: str, delta_ns: int) -> bool:
    """
    Shift embedded Time fields by subtracting delta_ns.
    Returns True if updated, False if skipped/unknown.
    """
    if delta_ns == 0:
        return True

    # Common: header.stamp
    header = getattr(msg, "header", None)
    if header is not None:
        stamp = getattr(header, "stamp", None)
        stamp_ns = _time_obj_to_ns(stamp)
        if stamp_ns is not None:
            _set_time_obj_from_ns(stamp, max(0, stamp_ns - delta_ns))

    if msgtype == "dvs_msgs/msg/EventArray":
        events = getattr(msg, "events", None)
        if isinstance(events, list) and events:
            for ev in events:
                ts_obj = getattr(ev, "ts", None)
                ev_ns = _time_obj_to_ns(ts_obj)
                if ev_ns is None:
                    continue
                _set_time_obj_from_ns(ts_obj, max(0, ev_ns - delta_ns))
        return True

    if msgtype in {"geometry_msgs/msg/PoseStamped", "sensor_msgs/msg/CameraInfo"}:
        return True

    # Unknown msg: we may have shifted header if present, but avoid guessing deeper.
    return header is not None


def _make_typestores_eventarray_compat():
    """
    Returns (ts_in, ts_out) where:
      - ts_in can deserialize the *old* dvs_msgs/EventArray schema (uint16 height/width)
      - ts_out serializes the *new* ROS Noetic schema (uint32 height/width)
    """
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    # Input typestore: old schema (uint16 height/width).
    ts_in = get_typestore(Stores.ROS1_NOETIC)
    event_msg = "time ts\nuint16 x\nuint16 y\nbool polarity\n"
    event_array_old = "std_msgs/Header header\ndvs_msgs/Event[] events\nuint16 height\nuint16 width\n"
    ts_in.register(get_types_from_msg(event_msg, "dvs_msgs/msg/Event"))
    ts_in.register(get_types_from_msg(event_array_old, "dvs_msgs/msg/EventArray"))

    # Output typestore: new schema (uint32 height/width).
    ts_out = get_typestore(Stores.ROS1_NOETIC)
    event_array_new = "std_msgs/Header header\ndvs_msgs/Event[] events\nuint32 height\nuint32 width\n"
    ts_out.register(get_types_from_msg(event_msg, "dvs_msgs/msg/Event"))
    ts_out.register(get_types_from_msg(event_array_new, "dvs_msgs/msg/EventArray"))

    return ts_in, ts_out


def _convert_eventarray_to_new_schema(ts_out, msg_in: Any) -> Any:
    """
    Convert an old-schema EventArray object (uint16 height/width) into a
    new-schema EventArray object (uint32 height/width).
    """
    EventOut = ts_out.get_msgdef("dvs_msgs/msg/Event").cls
    EventArrayOut = ts_out.get_msgdef("dvs_msgs/msg/EventArray").cls

    # Copy header (std_msgs/Header) and nested builtin Time objects as-is.
    header = getattr(msg_in, "header", None)

    # Convert events list by recreating Event objects (safe across typestores).
    events_in = getattr(msg_in, "events", []) or []
    events_out = []
    events_out_extend = events_out.extend
    for ev in events_in:
        events_out_extend(
            [
                EventOut(
                    ts=getattr(ev, "ts"),
                    x=int(getattr(ev, "x")),
                    y=int(getattr(ev, "y")),
                    polarity=bool(getattr(ev, "polarity")),
                )
            ]
        )

    h = int(getattr(msg_in, "height"))
    w = int(getattr(msg_in, "width"))
    return EventArrayOut(header=header, events=events_out, height=h, width=w)


def rewrite_bag(
    in_path: Path,
    out_path: Path,
    update_embedded: bool,
    enforce_monotonic: bool,
    start_at_ns: int,
    progress_every: int,
) -> RewriteStats:
    from rosbags.rosbag1 import Reader, Writer

    # General typestore (new/noetic schema for everything we write).
    ts_out = _make_typestore()
    # Compatibility typestore for reading old EventArray schema (uint16 h/w).
    ts_in_eventarray, ts_out_eventarray = _make_typestores_eventarray_compat()

    # Pass 1: find the earliest timestamp and a "sane" positive timestamp.
    raw_min: int | None = None
    sane_min: int | None = None
    any_msgs = False
    with Reader(in_path) as reader:
        for conn, t_ns, _raw in reader.messages():
            any_msgs = True
            t_ns_i = int(t_ns)
            raw_min = t_ns_i if raw_min is None else min(raw_min, t_ns_i)
            if not _is_sane_ns(t_ns_i):
                continue
            sane_min = t_ns_i if sane_min is None else min(sane_min, t_ns_i)

    if not any_msgs:
        raise RuntimeError(f"No messages in bag: {in_path}")
    if raw_min is None:
        raw_min = 0

    # Prefer shifting by the true earliest timestamp (including 0) when it is non-negative,
    # since rosbag_storage rejects Time(0) and we'd like the bag to start at start_at_ns.
    if raw_min >= 0:
        base_min = int(raw_min)
    else:
        base_min = int(sane_min) if sane_min is not None else int(raw_min)

    delta_ns = int(base_min) - int(start_at_ns)
    print(f"[rewrite] in:  {in_path}")
    print(f"[rewrite] out: {out_path}")
    print(f"[rewrite] start_at_ns: {int(start_at_ns)}")
    print(f"[rewrite] base_min_ns: {int(base_min)}")
    print(f"[rewrite] shift delta_ns: {delta_ns}  (new_t = old_t - delta_ns)")

    stats = RewriteStats()

    # Pass 2: rewrite messages.
    last_written: int = int(start_at_ns) if enforce_monotonic else 0
    with Reader(in_path) as reader, Writer(out_path) as writer:
        # Mirror connections.
        out_conn_by_in_id: dict[int, Any] = {}
        for c in reader.connections:
            out_conn_by_in_id[c.id] = writer.add_connection(c.topic, c.msgtype, typestore=ts_out)

        for i, (conn, t_ns, raw) in enumerate(reader.messages(), start=1):
            stats.messages_total += 1
            old_t = int(t_ns)
            new_t = old_t - delta_ns

            if new_t < int(start_at_ns):
                stats.messages_time_clamped += 1
                new_t = max(int(start_at_ns), last_written) if enforce_monotonic else int(start_at_ns)

            # Clamp obviously broken timestamps.
            if not _is_sane_ns(new_t):
                stats.messages_time_clamped += 1
                if enforce_monotonic:
                    new_t = max(last_written, int(start_at_ns))
                else:
                    new_t = max(0, min(new_t, UINT32_MAX * 1_000_000_000 + 999_999_999))

            if enforce_monotonic and new_t < last_written:
                stats.messages_time_nonmonotonic += 1
                new_t = last_written

            # We must reserialize dvs_msgs/EventArray to fix schema/MD5 if the input bag
            # was created with uint16 height/width (common mismatch with ROS Noetic).
            must_reencode = conn.msgtype == "dvs_msgs/msg/EventArray"

            out_raw = raw
            if update_embedded or must_reencode:
                try:
                    if conn.msgtype == "dvs_msgs/msg/EventArray":
                        msg_in = ts_in_eventarray.deserialize_ros1(raw, conn.msgtype)
                        msg = _convert_eventarray_to_new_schema(ts_out_eventarray, msg_in)
                        updated = _shift_embedded_times(ts_out_eventarray, msg, conn.msgtype, delta_ns)
                        out_raw = ts_out_eventarray.serialize_ros1(msg, conn.msgtype)
                    else:
                        msg = ts_out.deserialize_ros1(raw, conn.msgtype)
                        updated = _shift_embedded_times(ts_out, msg, conn.msgtype, delta_ns)
                        out_raw = ts_out.serialize_ros1(msg, conn.msgtype)

                    if updated:
                        stats.embedded_time_shifted += 1
                    else:
                        stats.embedded_time_skipped += 1
                except Exception:
                    stats.embedded_time_skipped += 1

            writer.write(out_conn_by_in_id[conn.id], int(new_t), out_raw)
            last_written = int(new_t)

            if progress_every > 0 and (i % progress_every) == 0:
                print(f"[rewrite] {i} msgs...")

    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_bag", type=str)
    ap.add_argument("out_bag", type=str)
    ap.add_argument("--no-update-embedded", action="store_true", help="Only fix record times; do not touch message contents.")
    ap.add_argument("--no-monotonic", action="store_true", help="Do not enforce non-decreasing record timestamps.")
    ap.add_argument(
        "--start-at-s",
        type=float,
        default=1.0,
        help="Shift bag so the earliest sane time becomes this many seconds (default: 1.0 to avoid Time(0)).",
    )
    ap.add_argument("--progress-every", type=int, default=500, help="Print progress every N messages (0=disable).")
    args = ap.parse_args()

    in_path = Path(args.in_bag)
    out_path = Path(args.out_bag)
    if not in_path.exists():
        raise SystemExit(f"bag not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = rewrite_bag(
        in_path=in_path,
        out_path=out_path,
        update_embedded=not bool(args.no_update_embedded),
        enforce_monotonic=not bool(args.no_monotonic),
        start_at_ns=int(float(args.start_at_s) * 1e9),
        progress_every=int(args.progress_every),
    )

    print("[rewrite] done")
    print(f"[rewrite] messages_total: {stats.messages_total}")
    print(f"[rewrite] messages_time_clamped: {stats.messages_time_clamped}")
    print(f"[rewrite] messages_time_nonmonotonic_fixed: {stats.messages_time_nonmonotonic}")
    print(f"[rewrite] embedded_time_shifted: {stats.embedded_time_shifted}")
    print(f"[rewrite] embedded_time_skipped: {stats.embedded_time_skipped}")


if __name__ == "__main__":
    main()
