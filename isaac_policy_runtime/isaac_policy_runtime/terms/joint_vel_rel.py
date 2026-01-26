from __future__ import annotations

import numpy as np

from .base import Term
from .registry import register
from .joints import build_joint_maps, joint_state_to_policy_vector

@register("joint_vel_rel")
@register("joint_vel")
class JointVelRel(Term):
    def compute(self) -> np.ndarray:
        src = self.wiring["source"]
        msg = self.sources.get(src)
        if msg is None:
            return np.zeros((len(self.action_cfg.joint_order),), dtype=np.float32)

        if not hasattr(msg, "name") or not hasattr(msg, "velocity"):
            raise TypeError("JointState missing 'name' or 'velocity'")

        joints_cfg = self.robot_if.joints or {}
        ros_to_policy, _ = build_joint_maps(joints_cfg)

        qd = joint_state_to_policy_vector(
            ros_names=list(msg.name),
            values=list(msg.velocity),
            action_joint_order=self.action_cfg.joint_order,
            ros_to_policy=ros_to_policy,
            strict=self.strict,
        )

        # vel_rel in IsaacLab is often relative to default (usually 0), so rel flag is present for symmetry
        # If rel is false, return as-is.
        return qd.astype(np.float32, copy=False)
