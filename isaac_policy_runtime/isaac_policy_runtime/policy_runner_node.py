from __future__ import annotations

import time
from typing import List
from typing import Optional

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
from .terms.joints import build_joint_maps, joint_state_to_policy_vector
import isaac_policy_runtime.terms  # noqa: F401

# ROS 2 helper to query remote node parameters (e.g. controller joints order).
# Depending on distro, the class name may vary. We support both.
try:  # pragma: no cover
    from rcl_interfaces.srv import GetParameters
except Exception:  # pragma: no cover
    GetParameters = None


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

        # Optional: publish targets relative to an offset.
        # If the bundle provides action offsets (action_offsets.yaml), we use them.
        # Otherwise, we fall back to capturing the current JointState.position at startup.
        self.control_relative = bool(ctrl.get("relative", False))
        self._control_offset: np.ndarray | None = None
        self._action_offsets: dict[str, float] = dict(ctrl.get("action_offsets", {}) or {})

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

        # Lazily capture initial joint offset for relative control.
        if self.control_relative and self._control_offset is None:
            self._control_offset = self._build_control_offset()

        q_target = self._decode_joint_position(action)
        if self.control_relative and self._control_offset is not None:
            if q_target.shape == self._control_offset.shape:
                q_target = q_target + self._control_offset
            else:
                self.get_logger().warning(
                    f"control.relative enabled but offset shape {self._control_offset.shape} != target {q_target.shape}; ignoring offset"
                )
        self.sink.publish_positions(q_target)

    def _build_control_offset(self) -> np.ndarray | None:
        """Build the outgoing command offset vector.

        Priority:
          1) Use action_offsets.yaml (loaded into robot_interface.control['action_offsets'])
          2) Fallback: capture current JointState.position

        Returns None if neither is available.
        """

        off = self._build_control_offset_from_action_offsets()
        if off is not None:
            return off
        return self._capture_initial_control_offset()

    def _build_control_offset_from_action_offsets(self) -> np.ndarray | None:
        if not self._action_offsets:
            return None

        # Build in controller joint order if resolved.
        if self._cmd_joint_order_ros is not None:
            ros_to_policy, _ = build_joint_maps(self.bundle.robot_interface.joints or {})
            out: list[float] = []
            for rn in self._cmd_joint_order_ros:
                pn = str(ros_to_policy.get(str(rn), str(rn)))
                out.append(float(self._action_offsets.get(pn, 0.0)))
            return np.asarray(out, dtype=np.float32)

        # Otherwise: build in policy action joint order.
        out = [float(self._action_offsets.get(j, 0.0)) for j in self.bundle.action_config.joint_order]
        return np.asarray(out, dtype=np.float32)

    def _capture_initial_control_offset(self) -> np.ndarray | None:
        """Capture initial joint positions in the outgoing command order.

        Returns None if joint_states is unavailable.
        """

        js = self.sources.get("joint_states")
        if js is None:
            return None

        # If we have controller joint order, capture in that order (ROS joint names).
        if self._cmd_joint_order_ros is not None:
            name_to_pos = {str(n): float(p) for n, p in zip(list(getattr(js, "name", []) or []), list(getattr(js, "position", []) or []))}
            out = [float(name_to_pos.get(rn, 0.0)) for rn in self._cmd_joint_order_ros]
            return np.asarray(out, dtype=np.float32)

        # Otherwise, fallback to policy joint order and apply ROS->policy name mapping.
        ros_to_policy, _ = build_joint_maps(self.bundle.robot_interface.joints or {})
        try:
            q0_policy = joint_state_to_policy_vector(
                ros_names=[str(x) for x in list(getattr(js, "name", []) or [])],
                values=[float(x) for x in list(getattr(js, "position", []) or [])],
                action_joint_order=list(self.bundle.action_config.joint_order),
                ros_to_policy=ros_to_policy,
                strict=False,
            )
        except Exception:
            return None
        return q0_policy

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

    def _get_remote_string_array_param(
        self, node_name: str, param_name: str, timeout_sec: float = 2.0
    ) -> Optional[list[str]]:
        """Read a remote string-array parameter via the standard get_parameters service.

        Works even when rclpy.parameter_client.AsyncParametersClient is unavailable.
        """
        if GetParameters is None:
            return None

        base = node_name if node_name.startswith("/") else f"/{node_name}"
        srv_name = f"{base}/get_parameters"

        client = self.create_client(GetParameters, srv_name)
        if not client.wait_for_service(timeout_sec=timeout_sec):
            return None

        req = GetParameters.Request()
        req.names = [param_name]

        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout_sec)
        resp = fut.result()
        if resp is None or not getattr(resp, "values", None) or len(resp.values) != 1:
            return None

        v = resp.values[0]
        arr = list(getattr(v, "string_array_value", []) or [])
        if not arr:
            return None
        return [str(x) for x in arr]
    

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
        
        joints_ros = self._get_remote_string_array_param(node_name=node_name, param_name=param_name, timeout_sec=2.0)
        if not joints_ros:
             self.get_logger().warning(
                 f"Param '{param_name}' from '{node_name}' is empty or not a string array. Publishing in policy order."
             )
             return
        
        self._cmd_joint_order_ros = joints_ros

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