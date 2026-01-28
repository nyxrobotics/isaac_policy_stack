from __future__ import annotations

import numpy as np

from .base import TermBase
from .joints import build_joint_maps, joint_state_to_policy_vector
from .registry import register


@register("joint_pos_rel")
@register("joint_pos")
class JointPosRel(TermBase):
    def compute(self) -> np.ndarray:
        src = str(self.wiring["source"])
        msg = self.sources.get(src)

        params = dict(self.obs_spec.params or {})
        joint_order = list(params.get("joint_order") or [])
        if not joint_order:
            # Fallback: policy action order (not ideal, but better than crashing)
            joint_order = list(self.action_cfg.joint_order or [])

        if msg is None:
            return np.zeros((len(joint_order),), dtype=np.float32)

        name_field = str(self.wiring.get("name_field", "name"))
        data_field = str(self.wiring.get("data_field", "position"))

        if not hasattr(msg, name_field) or not hasattr(msg, data_field):
            raise TypeError(f"JointState missing '{name_field}' or '{data_field}'")

        joints_cfg = self.robot_if.joints or {}
        ros_to_policy, _ = build_joint_maps(joints_cfg)

        q = joint_state_to_policy_vector(
            ros_names=list(getattr(msg, name_field)),
            values=list(getattr(msg, data_field)),
            joint_order=joint_order,
            ros_to_policy=ros_to_policy,
            strict=self.strict,
        )

        rel = bool(self.wiring.get("rel", params.get("relative", True)))

        # Log configuration once per node lifetime (per term instance).
        if not hasattr(self, "_logged_cfg"):
            self._logged_cfg = False  # type: ignore[attr-defined]
        if not self._logged_cfg:
            offsets = params.get("offsets")
            self.node.get_logger().info(f"[joint_pos] joint_order({len(joint_order)}): {joint_order}")
            if isinstance(offsets, list):
                self.node.get_logger().info(f"[joint_pos] offsets({len(offsets)}): {offsets}")
            self.node.get_logger().info(f"[joint_pos] relative={rel}, source={src}")
            self._logged_cfg = True  # type: ignore[attr-defined]

        if rel:
            offsets = params.get("offsets")
            if isinstance(offsets, list) and len(offsets) == len(joint_order):
                q = q - np.asarray([float(x) for x in offsets], dtype=np.float32)
            else:
                # Fallback: subtract robot default_pos (if present)
                defaults = dict((joints_cfg.get("default_pos") or {}))
                dq = []
                missing = []
                for pn, v in zip(joint_order, q.tolist()):
                    if pn in defaults:
                        dq.append(v - float(defaults[pn]))
                    else:
                        missing.append(pn)
                        dq.append(v)
                if self.strict and missing and len(defaults) > 0:
                    raise KeyError(f"default_pos missing entries for: {missing}")
                q = np.asarray(dq, dtype=np.float32)

        return q.astype(np.float32, copy=False)
