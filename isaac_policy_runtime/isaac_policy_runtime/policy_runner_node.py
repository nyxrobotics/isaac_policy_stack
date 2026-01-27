from __future__ import annotations

import time
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
from .terms.joints import build_joint_maps
import isaac_policy_runtime.terms  # noqa: F401

# ROS 2 helper to query remote node parameters (e.g. controller joints order).
# Depending on distro, the class name may vary. We support both.
AsyncParameterClient = None
try:  # pragma: no cover
    from rclpy.parameter_client import AsyncParameterClient as _APC  # type: ignore
    AsyncParameterClient = _APC
except Exception:  # pragma: no cover
    try:
        from rclpy.parameter_client import AsyncParametersClient as _APCs  # type: ignore
        AsyncParameterClient = _APCs
    except Exception:
        AsyncParameterClient = None


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

        # Prefer GPU if available, fallback to CPU.
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self.sess = ort.InferenceSession(str(self.bundle.policy_onnx), providers=providers)
        except Exception:
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

        # Newer bundles: command_topic / controller_name (+ runtime joint order resolution via ROS params)
        # Backwards-compatible bundles: topic
        topic = str(ctrl.get("command_topic", ctrl.get("topic", "/joint_group_position_controller/commands")))
        self.sink = Ros2ControlTopicSink(node=self, topic=topic)

        # Resolve controller joint order at runtime (optional but recommended)
        self._cmd_joint_order_ros: list[str] | None = None
        self._cmd_to_policy_idx: np.ndarray | None = None
        self._resolve_command_joint_order_and_map()

        self.rate_hz = float(ctrl.get("rate_hz", 200.0))
        self.decimation = int(ctrl.get("decimation", 4))
        self._tick = 0

        if self.log_io:
            self.get_logger().info(f"Bundle: {self.bundle.root}")
            self.get_logger().info(
                f"Observations terms ({len(self.bundle.observations)}): {[o.name for o in self.bundle.observations]}"
            )
            self.get_logger().info(
                f"Action joints ({len(self.bundle.action_config.joint_order)}): {self.bundle.action_config.joint_order}"
            )
            self.get_logger().info(f"Control topic: {topic} @ rate_hz={self.rate_hz}, decimation={self.decimation}")
            if self._cmd_joint_order_ros is not None:
                self.get_logger().info(
                    "Controller joints resolved via ROS params "
                    f"({len(self._cmd_joint_order_ros)}): {self._cmd_joint_order_ros}"
                )

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
            clip = t.obs_spec.clip
            if clip is not None:
                part = np.clip(part, clip[0], clip[1])
            obs_parts.append(part)

        obs = np.concatenate(obs_parts, axis=0).astype(np.float32, copy=False)

        if self.obs_mean is not None and self.obs_std is not None:
            if obs.shape != self.obs_mean.shape:
                raise RuntimeError(f"Obs shape {obs.shape} != normalization mean shape {self.obs_mean.shape}")
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-6)

        inp = obs.reshape(1, -1)
        out = self.sess.run([self.output_name], {self.input_name: inp})[0]
        action = np.asarray(out, dtype=np.float32).reshape(-1)

        self.state.last_action = action.copy()

        q_target = self._decode_joint_position(action)
        self.sink.publish_positions(q_target)

    def _decode_joint_position(self, action: np.ndarray) -> np.ndarray:
        cfg = self.bundle.action_config
        lo, hi = cfg.clip
        a = np.clip(action, lo, hi)

        scale = float(cfg.scale) if cfg.scale is not None else 1.0
        q = a * scale

        if bool(cfg.use_default_offset):
            defaults = dict((self.bundle.robot_interface.joints or {}).get("default_pos", {}) or {})
            offset = np.asarray([float(defaults.get(j, 0.0)) for j in cfg.joint_order], dtype=np.float32)
            q = q + offset

        q_policy = q.astype(np.float32, copy=False)

        # Remap policy joint order -> controller command joint order (resolved at runtime).
        if self._cmd_to_policy_idx is None:
            # No controller mapping available: publish in policy order (legacy behavior).
            return q_policy

        q_cmd = np.zeros((int(self._cmd_to_policy_idx.shape[0]),), dtype=np.float32)
        for i, pol_idx in enumerate(self._cmd_to_policy_idx.tolist()):
            if pol_idx >= 0:
                q_cmd[i] = float(q_policy[int(pol_idx)])
            else:
                # Controller joint not part of policy DOFs (e.g., unactuated). Keep zero.
                q_cmd[i] = 0.0
        return q_cmd

    def _resolve_command_joint_order_and_map(self) -> None:
        """Resolve controller command joint order and build an index map into policy joint order.

        Runtime behavior:
          - Query: ros2 param get <controller_name> joints
          - Map: controller_ros_joint -> policy_joint_index

        We DO NOT put controller joint order into the bundle.
        """
        ctrl = self.bundle.robot_interface.control
        controller_name = str(ctrl.get("controller_name", ""))

        resolution = dict(ctrl.get("command_joint_order_resolution", {}) or {})
        param_name = str(resolution.get("param", "joints"))
        node_name = str(resolution.get("node", controller_name))

        if not node_name:
            return
        if AsyncParameterClient is None:
            self.get_logger().warning(
                "AsyncParameterClient is unavailable. Skipping controller joint order resolution; publishing in policy order."
            )
            return

        client = AsyncParameterClient(self, node_name)

        # rclpy provides wait_for_services()/services_are_ready() depending on version.
        ok = False
        if hasattr(client, "wait_for_services"):
            ok = bool(client.wait_for_services(timeout_sec=2.0))
        elif hasattr(client, "services_are_ready"):
            t0 = time.time()
            while (time.time() - t0) < 2.0 and not bool(client.services_are_ready()):
                rclpy.spin_once(self, timeout_sec=0.05)
            ok = bool(client.services_are_ready())

        if not ok:
            self.get_logger().warning(
                f"Parameter services not available for node '{node_name}'. Publishing in policy order."
            )
            return

        fut = client.get_parameters([param_name])
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        if fut.result() is None:
            self.get_logger().warning(
                f"Failed to read param '{param_name}' from node '{node_name}'. Publishing in policy order."
            )
            return

        params_msg = fut.result()
        values = getattr(params_msg, "values", None)
        if values is None or len(values) != 1:
            self.get_logger().warning(
                f"Unexpected response reading param '{param_name}' from node '{node_name}'. Publishing in policy order."
            )
            return

        v = values[0]
        joints_ros = list(getattr(v, "string_array_value", []) or [])
        if not joints_ros:
            self.get_logger().warning(
                f"Param '{param_name}' from '{node_name}' is empty or not a string array. Publishing in policy order."
            )
            return

        self._cmd_joint_order_ros = [str(x) for x in joints_ros]

        # Build ROS->policy map (optional). If missing, identity is used.
        ros_to_policy, _ = build_joint_maps(self.bundle.robot_interface.joints or {})
        policy_order = list(self.bundle.action_config.joint_order)
        pol_index = {name: i for i, name in enumerate(policy_order)}

        idx: list[int] = []
        missing: list[dict[str, str]] = []
        for rn in self._cmd_joint_order_ros:
            pn = ros_to_policy.get(rn, rn)
            if pn in pol_index:
                idx.append(int(pol_index[pn]))
            else:
                idx.append(-1)
                missing.append({"ros": rn, "policy": pn})

        if missing and self.strict:
            msg = "Controller joint list contains joints not present in policy_joint_order: " + ", ".join(
                [f"{m['ros']}-> {m['policy']}" for m in missing]
            )
            raise RuntimeError(msg)

        self._cmd_to_policy_idx = np.asarray(idx, dtype=np.int32)


def main():
    rclpy.init()
    node = PolicyRunner()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()