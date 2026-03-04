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
from .terms.joints import build_joint_maps
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
        self.declare_parameter("log_every", 50)

        bundle_path = str(self.get_parameter("bundle_path").value)
        self.use_tf = bool(self.get_parameter("use_tf").value)
        self.strict = bool(self.get_parameter("strict").value)
        self.log_io = bool(self.get_parameter("log_io").value)
        self.log_every = int(self.get_parameter("log_every").value)
        if self.log_every <= 0:
            self.log_every = 1

        if not bundle_path:
            raise RuntimeError("Parameter 'bundle_path' is required")

        self.bundle: PolicyBundle = load_bundle(bundle_path)

        # Sanity-check: observation/action configs vs io_descriptors.yaml (if available).
        # This is ABI-critical. If mismatch is detected and strict=True, we raise.
        self._check_io_descriptor_consistency()

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
        for obs in self.bundle.observation_config.terms:
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

        # Drive policy inference and command publish strictly by dt (= 1 / rate_hz).
        self.rate_hz = float(ctrl.get("rate_hz", 200.0))
        self._tick = 0

        if self.log_io:
            self.get_logger().info(f"Bundle: {self.bundle.root}")
            obs_names = [o.name for o in self.bundle.observation_config.terms]
            self.get_logger().info(f"Observation terms ({len(obs_names)}): {obs_names}")

            # ---- Observation vector layout (concat order is ABI-critical)
            cursor = 0
            self.get_logger().info("Observation vector layout (concat order):")
            for o in self.bundle.observation_config.terms:
                dim = int(o.dim) if o.dim is not None else 0
                shape = list(o.shape or [])

                start_i = cursor
                end_i = cursor + dim  # exclusive
                self.get_logger().info(f"  [{start_i}:{end_i}) {o.name} shape={shape} dim={dim}")
                cursor = end_i

                # If this term has a meaningful element order (e.g., joints), print it.
                p = dict(o.params or {})
                jo = p.get("joint_order")
                off = p.get("offsets")
                if isinstance(jo, list) and len(jo) > 0:
                    # Print (term-local index -> joint name [+ offset]) mapping.
                    pairs = []
                    for k, jn in enumerate(jo):
                        if isinstance(off, list) and k < len(off):
                            pairs.append(f"{k}:{jn}(offset={float(off[k]):.6g})")
                        else:
                            pairs.append(f"{k}:{jn}")
                    self.get_logger().info(f"    element_order: {pairs}")
                    self.get_logger().info(f"    relative(default)={p.get('relative', None)}")

            expected = int(self.bundle.observation_config.total_dim)
            self.get_logger().info(f"Observation total_dim: computed={cursor}, observation_config.yaml={expected}")

            # ---- Action layout (policy output ABI)
            cfg = self.bundle.action_config
            self.get_logger().info(f"Action scale={cfg.scale}, clip={cfg.clip}, relative={cfg.relative}")
            if isinstance(cfg.joint_order, list) and len(cfg.joint_order) > 0:
                pairs = []
                for k, jn in enumerate(cfg.joint_order):
                    if isinstance(cfg.offset, list) and k < len(cfg.offset):
                        pairs.append(f"{k}:{jn}(offset={float(cfg.offset[k]):.6g})")
                    else:
                        pairs.append(f"{k}:{jn}")
                self.get_logger().info(f"Action element_order: {pairs}")
            if isinstance(cfg.offset, list):
                self.get_logger().info(f"Action offsets({len(cfg.offset)}): {cfg.offset}")

            self.get_logger().info(f"Control topic: {topic} @ rate_hz={self.rate_hz}")
            ctrl_rel = (self.bundle.robot_interface.control or {}).get("rel", None)
            if ctrl_rel is None:
                ctrl_rel = True
            self.get_logger().info(
                f"Control rel={bool(ctrl_rel)}: apply action_config offsets={'enabled' if bool(ctrl_rel) else 'disabled'}"
            )
            if self._cmd_joint_order_ros is not None:
                self.get_logger().info(
                    "Controller joints resolved via ROS params "
                    f"({len(self._cmd_joint_order_ros)}): {self._cmd_joint_order_ros}"
                )
        period = 1.0 / self.rate_hz
        self.timer = self.create_timer(period, self._step)

    def _check_io_descriptor_consistency(self) -> None:
        """Verify that action/observation configs match io_descriptors.yaml.

        This check ensures the policy ABI (term order, per-term shapes, joint orders, and offsets)
        matches what Isaac Lab exported.
        """
        io = self.bundle.io_descriptors
        if not isinstance(io, dict):
            return

        errors: list[str] = []
        warnings: list[str] = []

        # ---- Observations
        obs = io.get("observations")
        policy = obs.get("policy") if isinstance(obs, dict) else None
        if isinstance(policy, list):
            io_names = [str(t.get("name")) for t in policy if isinstance(t, dict)]
            cfg_names = [t.name for t in self.bundle.observation_config.terms]
            if io_names != cfg_names:
                errors.append(
                    "observation term order mismatch: io_descriptors.yaml vs observation_config.yaml\n"
                    f"  io:  {io_names}\n  cfg: {cfg_names}"
                )

            # Per-term shape + joint details
            by_name = {t.name: t for t in self.bundle.observation_config.terms}
            for t in policy:
                if not isinstance(t, dict):
                    continue
                name = str(t.get("name"))
                cfg = by_name.get(name)
                if cfg is None:
                    continue
                shape_raw = t.get("shape")
                if shape_raw is not None:
                    if isinstance(shape_raw, int):
                        io_shape = [int(shape_raw)]
                    else:
                        io_shape = [int(x) for x in list(shape_raw)]
                    cfg_shape = list(cfg.shape or [])
                    if io_shape != cfg_shape:
                        errors.append(f"observation shape mismatch for '{name}': io={io_shape} cfg={cfg_shape}")

                # Joint order + offsets are meaning-bearing.
                if isinstance(t.get("joint_names"), list):
                    io_joints = [str(x) for x in list(t.get("joint_names") or [])]
                    cfg_joints = list((cfg.params or {}).get("joint_order") or [])
                    if io_joints != cfg_joints:
                        errors.append(
                            f"observation joint order mismatch for '{name}': io={io_joints} cfg={cfg_joints}"
                        )

                # Offsets keys differ by term type.
                io_off = None
                if isinstance(t.get("joint_pos_offsets"), list):
                    io_off = [float(x) for x in list(t.get("joint_pos_offsets") or [])]
                if isinstance(t.get("joint_vel_offsets"), list):
                    io_off = [float(x) for x in list(t.get("joint_vel_offsets") or [])]
                if io_off is not None:
                    cfg_off = (cfg.params or {}).get("offsets")
                    cfg_off = [float(x) for x in list(cfg_off or [])] if isinstance(cfg_off, list) else []
                    if io_off != cfg_off:
                        errors.append(
                            f"observation offsets mismatch for '{name}': io={io_off} cfg={cfg_off}"
                        )
        else:
            warnings.append("io_descriptors.yaml has no observations.policy; skipping observation consistency check")

        # ---- Actions
        actions = io.get("actions")
        if isinstance(actions, list) and actions and isinstance(actions[0], dict):
            a0 = actions[0]
            io_joints = [str(x) for x in list(a0.get("joint_names") or [])]
            cfg_joints = list(self.bundle.action_config.joint_order or [])
            if io_joints and io_joints != cfg_joints:
                errors.append(f"action joint order mismatch: io={io_joints} cfg={cfg_joints}")

            if isinstance(a0.get("offset"), list) and self.bundle.action_config.offset is not None:
                io_off = [float(x) for x in list(a0.get("offset") or [])]
                cfg_off = [float(x) for x in list(self.bundle.action_config.offset or [])]
                if io_off != cfg_off:
                    errors.append(f"action offsets mismatch: io={io_off} cfg={cfg_off}")
        else:
            warnings.append("io_descriptors.yaml has no actions[0]; skipping action consistency check")

        for w in warnings:
            self.get_logger().warning(w)
        if errors:
            for e in errors:
                self.get_logger().error(e)
            if self.strict:
                raise RuntimeError("IO descriptor consistency check failed (strict=True).")

    def _step(self):
        self._tick += 1

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

        # Apply action offsets.
        #
        # The offset values live in action_config.yaml (exported from IO_descriptors.yaml).
        # robot_interface.yaml only toggles whether they are applied, via control.rel.
        ctrl = self.bundle.robot_interface.control or {}
        ctrl_rel = ctrl.get("rel", None)
        if ctrl_rel is None:
            # Default to True to match Isaac Lab's relative-action convention.
            ctrl_rel = True

        if bool(ctrl_rel):
            if isinstance(cfg.offset, list) and len(cfg.offset) == len(cfg.joint_order):
                q = q + np.asarray([float(x) for x in cfg.offset], dtype=np.float32)
            elif bool(cfg.use_default_offset):
                defaults = dict((self.bundle.robot_interface.joints or {}).get("default_pos", {}) or {})
                offset = np.asarray([float(defaults.get(j, 0.0)) for j in cfg.joint_order], dtype=np.float32)
                q = q + offset

        q_policy = q.astype(np.float32, copy=False)

        # Remap policy joint order -> controller command joint order (resolved at runtime).
        if self._cmd_to_policy_idx is None:
            # No controller mapping available: publish in policy order (legacy behavior).
            q_cmd = q_policy
            cmd_joint_names: list[str] | None = None
            self.get_logger().warning("No controller joint order mapping available. Publishing in policy joint order.")
        else:
            q_cmd = np.zeros((int(self._cmd_to_policy_idx.shape[0]),), dtype=np.float32)
            for i, pol_idx in enumerate(self._cmd_to_policy_idx.tolist()):
                if pol_idx >= 0:
                    q_cmd[i] = float(q_policy[int(pol_idx)])
                else:
                    # Controller joint not part of policy DOFs (e.g., unactuated). Keep zero.
                    q_cmd[i] = 0.0
            cmd_joint_names = list(self._cmd_joint_order_ros or []) if getattr(self, "_cmd_joint_order_ros", None) else None

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