from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import rclpy
from rclpy.node import Node

from ..bundle.schema import RobotInterface, SourceSpec

@dataclass
class CachedMsg:
    msg: Any | None = None
    stamp_sec: float | None = None

class SourceManager:
    def __init__(self, node: Node, robot_if: RobotInterface):
        self._node = node
        self._robot_if = robot_if
        self._cache: Dict[str, CachedMsg] = {}

        # subscriptions created lazily on demand
        self._subs: Dict[str, Any] = {}

    def ensure_source(self, source_name: str) -> None:
        if source_name in self._subs:
            return
        if source_name not in self._robot_if.sources:
            raise KeyError(f"Source '{source_name}' not defined in robot_interface.yaml -> sources")

        spec: SourceSpec = self._robot_if.sources[source_name]

        # Resolve message type string to class
        msg_cls = self._resolve_msg_type(spec.msg_type)
        self._cache[source_name] = CachedMsg()

        self._node.get_logger().info(f"Subscribing source '{source_name}' -> {spec.topic} ({spec.msg_type})")
        self._subs[source_name] = self._node.create_subscription(
            msg_cls,
            spec.topic,
            lambda m, s=source_name: self._on_msg(s, m),
            10,
        )

    def get(self, source_name: str) -> Any | None:
        return self._cache.get(source_name, CachedMsg()).msg

    def _on_msg(self, source_name: str, msg: Any) -> None:
        t = self._node.get_clock().now().nanoseconds / 1e9
        self._cache[source_name] = CachedMsg(msg=msg, stamp_sec=t)

    def _resolve_msg_type(self, type_str: str):
        # Minimal resolver for common types used in v1.
        # Extend as needed.
        if type_str == "sensor_msgs/Imu":
            from sensor_msgs.msg import Imu
            return Imu
        if type_str == "sensor_msgs/JointState":
            from sensor_msgs.msg import JointState
            return JointState
        if type_str == "geometry_msgs/Twist":
            from geometry_msgs.msg import Twist
            return Twist
        raise ValueError(f"Unsupported msg_type '{type_str}'. Add it to SourceManager._resolve_msg_type().")
