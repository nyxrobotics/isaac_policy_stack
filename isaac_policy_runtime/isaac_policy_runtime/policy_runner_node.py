from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np
import rclpy
from rclpy.node import Node

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

from .bundle.loader import load_bundle, PolicyBundle
from .io.source_manager import SourceManager
from .terms.base import RuntimeState
from .terms.registry import get_term_class
from .control.sinks.ros2_control_topic import Ros2ControlTopicSink


class PolicyRunner(Node):
    def __init__(self):
        super().__init__("policy_runner")

        self.declare_parameter("bundle_path", "")
        self.declare_parameter("use_tf", True)
        self.declare_parameter("strict", True)
        self.declare_parameter("log_io", True)

        bundle_path = str(self.get_parameter("bundle_path").value)
        self.use_tf = bool(self.get_parameter("use_tf").value)
        self.strict = bool(self.get_parameter("strict").value)
        self.log_io = bool(self.get_parameter("log_io").value)

        if not bundle_path:
            raise RuntimeError("Parameter 'bundle_path' is required")

        self.bundle: PolicyBundle = load_bundle(bundle_path)

        if ort is None:
            raise RuntimeError("onnxruntime is not installed. Install it with: pip install onnxruntime")

        self.sess = ort.InferenceSession(str(self.bundle.policy_onnx), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.output_name = self.sess.get_outputs()[0].name

        self.sources = SourceManager(self, self.bundle.robot_interface)
        self.state = RuntimeState(last_action=None)

        # Build term pipeline
        self.terms = []
        for obs in self.bundle.observations:
            TermCls = get_term_class(obs.func)
            term = TermCls(
                node=self,
                obs_spec=obs,
                robot_if=self.bundle.robot_interface,
                action_cfg=self.bundle.action_config,
                sources=self.sources,
                state=self.state,
                use_tf=self.use_tf,
                strict=self.strict,
            )
            self.terms.append(term)

        # Normalization (optional)
        self.obs_mean = None
        self.obs_std = None
        norm = self.bundle.obs_normalization
        if isinstance(norm, dict) and "mean" in norm and "std" in norm:
            self.obs_mean = np.asarray(norm["mean"], dtype=np.float32)
            self.obs_std = np.asarray(norm["std"], dtype=np.float32)

        # Control sink
        ctrl = self.bundle.robot_interface.control
        sink_type = str(ctrl.get("sink", "ros2_control_topic"))
        if sink_type != "ros2_control_topic":
            raise ValueError(f"Unsupported control sink '{sink_type}' (v1 supports ros2_control_topic only)")

        topic = str(ctrl.get("topic", "/joint_group_pos_controller/commands"))
        self.sink = Ros2ControlTopicSink(node=self, topic=topic)

        self.rate_hz = float(ctrl.get("rate_hz", 200.0))
        self.decimation = int(ctrl.get("decimation", 4))
        self._tick = 0

        if self.log_io:
            self.get_logger().info(f"Bundle: {self.bundle.root}")
            self.get_logger().info(f"Observations terms ({len(self.bundle.observations)}): {[o.name for o in self.bundle.observations]}")
            self.get_logger().info(f"Action joints ({len(self.bundle.action_config.joint_order)}): {self.bundle.action_config.joint_order}")
            self.get_logger().info(f"Control topic: {topic} @ rate_hz={self.rate_hz}, decimation={self.decimation}")

        period = 1.0 / self.rate_hz
        self.timer = self.create_timer(period, self._step)

    def _step(self):
        self._tick += 1
        if (self._tick % self.decimation) != 0:
            return

        obs_parts: List[np.ndarray] = []
        for t in self.terms:
            part = t.compute()
            part = np.asarray(part, dtype=np.float32).reshape(-1)
            # optional clip per term
            clip = t.obs_spec.clip
            if clip is not None:
                part = np.clip(part, clip[0], clip[1])
            obs_parts.append(part)

        obs = np.concatenate(obs_parts, axis=0).astype(np.float32, copy=False)

        if self.obs_mean is not None and self.obs_std is not None:
            if obs.shape != self.obs_mean.shape:
                raise RuntimeError(f"Obs shape {obs.shape} != normalization mean shape {self.obs_mean.shape}")
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-6)

        # ONNX inference expects batch dimension
        inp = obs.reshape(1, -1)
        out = self.sess.run([self.output_name], {self.input_name: inp})[0]
        action = np.asarray(out, dtype=np.float32).reshape(-1)

        # store last action (policy output)
        self.state.last_action = action.copy()

        # Decode joint position targets
        q_target = self._decode_joint_position(action)

        # Publish
        self.sink.publish_positions(q_target)

    def _decode_joint_position(self, action: np.ndarray) -> np.ndarray:
        cfg = self.bundle.action_config
        lo, hi = cfg.clip
        a = np.clip(action, lo, hi)

        scale = float(cfg.scale) if cfg.scale is not None else 1.0
        q = a * scale

        if bool(cfg.use_default_offset):
            defaults = dict((self.bundle.robot_interface.joints or {}).get("default_pos", {}) or {})
            # default_pos is keyed by policy joint names
            offset = np.asarray([float(defaults.get(j, 0.0)) for j in cfg.joint_order], dtype=np.float32)
            q = q + offset

        return q.astype(np.float32, copy=False)


def main():
    rclpy.init()
    node = PolicyRunner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
