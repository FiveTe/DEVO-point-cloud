#!/usr/bin/env python3
"""
Quickly probe ROS1 .bag timestamp health.

Focus:
  - outer record timestamps (what `rosbag info` uses for start/end/duration)
  - embedded message time fields for seeded_mcemvs topics (optional)

Works without ROS; uses the `rosbags` Python package.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _make_typestore():
    from rosbags.typesys import Stores, get_typestore, get_types_from_msg

    ts = get_typestore(Stores.ROS1_NOETIC)

    # Seeded MC-EMVS pipeline writes these custom messages.
    event_msg = "time ts\nuint16 x\nuint16 y\nbool polarity\n"
    event_array_msg = "std_msgs/Header header\ndvs_msgs/Event[] events\nuint32 height\nuint32 width\n"
    ts.register(get_types_from_msg(event_msg, "dvs_msgs/msg/Event"))
    ts.register(get_types_from_msg(event_array_msg, "dvs_msgs/msg/EventArray"))

    return ts


UINT32_MAX = 0xFFFFFFFF


def _ns_to_sec_nsec(t_ns: int) -> tuple[int, int]:
    sec = int(t_ns // 1_000_000_000)
    nsec = int(t_ns % 1_000_000_000)
    return sec, nsec


def _is_record_time_sane(t_ns: int) -> tuple[bool, str]:
    if t_ns is None:
        return False, "none"
    if not isinstance(t_ns, int):
        return False, f"non-int({type(t_ns).__name__})"
    sec, nsec = _ns_to_sec_nsec(t_ns)
    if t_ns == 0:
        # ROS tools (rosbag_storage) can reject Time(0) as "invalid".
        return False, "zero"
    if t_ns < 0:
        return False, "negative"
    if not (0 <= nsec < 1_000_000_000):
        return False, "nsec_out_of_range"
    if not (0 <= sec <= UINT32_MAX):
        return False, "sec_out_of_range_uint32"
    return True, "ok"


def _time_obj_to_ns(t: Any) -> int | None:
    # rosbags Time has fields: sec, nanosec
    sec = getattr(t, "sec", None)
    nsec = getattr(t, "nanosec", None)
    if sec is None or nsec is None:
        return None
    try:
        sec_i = int(sec)
        nsec_i = int(nsec)
    except Exception:
        return None
    return sec_i * 1_000_000_000 + nsec_i


@dataclass
class Finding:
    kind: str
    msg_index: int
    topic: str
    msgtype: str
    detail: str


def probe_bag(path: Path, check_embedded: bool, max_messages: int | None) -> int:
    from rosbags.rosbag1 import Reader

    ts = _make_typestore()

    findings: list[Finding] = []
    min_ts: int | None = None
    max_ts: int | None = None
    first_bad_record: Finding | None = None

    with Reader(path) as reader:
        conns = list(reader.connections)
        conn_by_id = {c.id: c for c in conns}

        # Record timestamp scan + (optional) embedded time checks.
        for i, (conn, t_ns, raw) in enumerate(reader.messages(), start=1):
            if max_messages is not None and i > max_messages:
                break

            if min_ts is None or t_ns < min_ts:
                min_ts = t_ns
            if max_ts is None or t_ns > max_ts:
                max_ts = t_ns

            ok, reason = _is_record_time_sane(int(t_ns))
            if not ok and first_bad_record is None:
                first_bad_record = Finding(
                    kind="record_time",
                    msg_index=i,
                    topic=conn.topic,
                    msgtype=conn.msgtype,
                    detail=f"{reason}: t_ns={t_ns} sec={_ns_to_sec_nsec(int(t_ns))[0]}",
                )

            if not check_embedded:
                continue

            # Only attempt to deserialize types we know how to handle.
            # If a message can't be deserialized, skip quietly.
            try:
                msg = ts.deserialize_ros1(raw, conn.msgtype)
            except Exception:
                continue

            # Common: header.stamp
            header = getattr(msg, "header", None)
            if header is not None:
                stamp = getattr(header, "stamp", None)
                stamp_ns = _time_obj_to_ns(stamp)
                if stamp_ns is not None:
                    ok2, reason2 = _is_record_time_sane(int(stamp_ns))
                    if not ok2:
                        findings.append(
                            Finding(
                                kind="header_stamp",
                                msg_index=i,
                                topic=conn.topic,
                                msgtype=conn.msgtype,
                                detail=f"{reason2}: stamp_ns={stamp_ns}",
                            )
                        )

            # dvs_msgs/EventArray: check a few event timestamps to avoid O(N) scans.
            if conn.msgtype == "dvs_msgs/msg/EventArray":
                events = getattr(msg, "events", None)
                if isinstance(events, list) and events:
                    for label, ev in (("first", events[0]), ("last", events[-1])):
                        ts_obj = getattr(ev, "ts", None)
                        ev_ns = _time_obj_to_ns(ts_obj)
                        if ev_ns is None:
                            continue
                        ok3, reason3 = _is_record_time_sane(int(ev_ns))
                        if not ok3:
                            findings.append(
                                Finding(
                                    kind=f"event_ts_{label}",
                                    msg_index=i,
                                    topic=conn.topic,
                                    msgtype=conn.msgtype,
                                    detail=f"{reason3}: event_{label}_ns={ev_ns}",
                                )
                            )

        # Summary
        print(f"[probe] bag: {path}")
        if min_ts is None or max_ts is None:
            print("[probe] no messages found")
            return 2

        min_sec, min_nsec = _ns_to_sec_nsec(int(min_ts))
        max_sec, max_nsec = _ns_to_sec_nsec(int(max_ts))
        print(f"[probe] connections: {len(conns)}")
        print(f"[probe] record_time min: {min_ts} (sec={min_sec} nsec={min_nsec})")
        print(f"[probe] record_time max: {max_ts} (sec={max_sec} nsec={max_nsec})")
        print(f"[probe] record_time duration_s: {(max_ts - min_ts) / 1e9:.6f}")

        if first_bad_record is not None:
            print(
                "[probe] FIRST BAD RECORD TIME:",
                f"msg#{first_bad_record.msg_index}",
                first_bad_record.topic,
                first_bad_record.msgtype,
                first_bad_record.detail,
            )
        else:
            print("[probe] record_time: OK (uint32 sec, nsec in range, non-negative)")

        if check_embedded:
            if findings:
                print(f"[probe] embedded_time findings: {len(findings)} (showing up to 20)")
                for fnd in findings[:20]:
                    print(
                        "  -",
                        fnd.kind,
                        f"msg#{fnd.msg_index}",
                        fnd.topic,
                        fnd.msgtype,
                        fnd.detail,
                    )
            else:
                print("[probe] embedded_time: no issues found (sampled)")

        # Print a connection list for convenience.
        by_topic = sorted({(c.topic, c.msgtype, getattr(c, "id", None)) for c in conns})
        print("[probe] topics:")
        for topic, msgtype, cid in by_topic:
            print("  -", f"{topic} ({msgtype})", f"id={cid}")

        # Exit code: 0 ok, 1 bad record time, 2 unreadable/no msgs
        return 1 if first_bad_record is not None else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bag", type=str, help="Path to ROS1 .bag")
    ap.add_argument(
        "--check-embedded",
        action="store_true",
        help="Deserialize messages (if possible) and sanity-check header/event stamps.",
    )
    ap.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Limit scan to first N messages (record_time always scanned up to N).",
    )
    args = ap.parse_args()

    path = Path(args.bag)
    if not path.exists():
        raise SystemExit(f"bag not found: {path}")

    raise SystemExit(probe_bag(path, check_embedded=bool(args.check_embedded), max_messages=args.max_messages))


if __name__ == "__main__":
    main()
