from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

try:  # pragma: no cover - prefers native rosbag when available
    import rosbag as _rosbag  # type: ignore
except ImportError:  # pragma: no cover
    _rosbag = None  # type: ignore

__all__ = ["Bag"]


if _rosbag is not None:  # pragma: no cover - ROS environment provided
    Bag = _rosbag.Bag  # type: ignore[misc]
else:
    from rosbags.highlevel import AnyReader

    _TopicInfo = Tuple[str, int, Tuple[int, ...]]
    _TopicsDict = Dict[str, _TopicInfo]

    class Bag:
        """Lightweight rosbag reader built on top of rosbags."""

        def __init__(self, filename: Union[str, Path], mode: str = "r") -> None:
            if mode not in ("r", "rb"):
                raise ValueError("rosbags-based reader only supports read mode.")

            bag_path = Path(filename)
            if not bag_path.exists():
                raise FileNotFoundError(bag_path)

            self._reader = AnyReader([bag_path])
            self._reader.__enter__()
            self._topic_connections = self._build_topic_map()

        def __enter__(self) -> "Bag":
            return self

        def __exit__(self, exc_type, exc, exc_tb) -> None:
            self.close()

        def close(self) -> None:
            if self._reader is not None:
                self._reader.__exit__(None, None, None)
                self._reader = None

        def __del__(self) -> None:  # pragma: no cover - best effort cleanup
            self.close()

        def get_type_and_topic_info(self) -> Tuple[Dict[str, None], _TopicsDict]:
            """Replicate rosbag.Bag.get_type_and_topic_info()."""
            topic_info: _TopicsDict = {}
            for topic, connections in self._topic_connections.items():
                msgtype = connections[0].msgtype
                count = sum(conn.msgcount or 0 for conn in connections)
                topic_info[topic] = (msgtype, count, tuple(conn.id for conn in connections))
            return {}, topic_info

        def get_message_count(self, topic: str) -> int:
            connections = self._topic_connections.get(topic)
            if not connections:
                return 0
            return sum(conn.msgcount or 0 for conn in connections)

        def read_messages(
            self, topics: Optional[Union[str, Iterable[str]]] = None
        ) -> Iterator[Tuple[str, object, int]]:
            connections = self._select_connections(topics)
            for connection, timestamp, rawdata in self._reader.messages(connections=connections):
                msg = self._reader.deserialize(rawdata, connection.msgtype)
                yield connection.topic, msg, timestamp

        def _build_topic_map(self):
            topic_map: Dict[str, List] = {}
            for connection in self._reader.connections:
                topic_map.setdefault(connection.topic, []).append(connection)
            return topic_map

        def _select_connections(
            self, topics: Optional[Union[str, Iterable[str]]]
        ) -> Sequence:
            if topics is None:
                return list(self._reader.connections)

            if isinstance(topics, (str, bytes)):
                topic_list: List[str] = [topics]
            else:
                topic_list = list(topics)

            selected = []
            seen = set()
            for topic in topic_list:
                if topic in seen:
                    continue
                seen.add(topic)
                selected.extend(self._topic_connections.get(topic, []))
            return selected
