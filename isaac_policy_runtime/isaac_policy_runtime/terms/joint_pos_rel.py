from __future__ import annotations

import numpy as np

from .base import TermBase
from .registry import register
from .joints import build_joint_maps, joint_state_to_policy_vector

@register("joint_pos_rel")
@register("joint_pos")
class JointPosRel(TermBase):
    def compute(self) -> np.ndarray:
        src = self.wiring["source"]
        msg = self.sources.get(src)
        if msg is None:
            return np.zeros((len(self.action_cfg.joint_order),), dtype=np.float32)

        if not hasattr(msg, "name") or not hasattr(msg, "position"):
            raise TypeError("JointState missing 'name' or 'position'")

        joints_cfg = self.robot_if.joints or {}
        ros_to_policy, _ = build_joint_maps(joints_cfg)

        q = joint_state_to_policy_vector(
            ros_names=list(msg.name),
            values=list(msg.position),
            action_joint_order=self.action_cfg.joint_order,
            ros_to_policy=ros_to_policy,
            strict=self.strict,
        )

        rel = bool(self.wiring.get("rel", True))
        if rel:
            defaults = dict((joints_cfg.get("default_pos") or {}))
            dq = []
            missing = []
            for pn, v in zip(self.action_cfg.joint_order, q.tolist()):
                if pn in defaults:
                    dq.append(v - float(defaults[pn]))
                else:
                    missing.append(pn)
                    dq.append(v)  # fallback: no subtraction
            if self.strict and missing and len(defaults) > 0:
                # only error if defaults were provided but incomplete
                raise KeyError(f"default_pos missing entries for: {missing}")
            q = np.asarray(dq, dtype=np.float32)

        return q.astype(np.float32, copy=False)
