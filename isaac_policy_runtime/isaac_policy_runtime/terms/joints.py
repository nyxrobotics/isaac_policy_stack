from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np

def build_joint_maps(joints_cfg: dict) -> tuple[dict[str, str], dict[str, str]]:
    # name_map is ROS -> policy
    ros_to_policy = dict(joints_cfg.get("name_map", {}) or {})
    # invert
    policy_to_ros = {p: r for r, p in ros_to_policy.items()}
    return ros_to_policy, policy_to_ros

def joint_state_to_policy_vector(
    ros_names: list[str],
    values: list[float],
    action_joint_order: list[str],
    ros_to_policy: dict[str, str],
    strict: bool,
) -> np.ndarray:
    # Build policy_name -> value dict from ROS message
    pol_vals: dict[str, float] = {}
    for rn, v in zip(ros_names, values):
        pn = ros_to_policy.get(rn, rn)
        pol_vals[pn] = float(v)

    out = []
    missing = []
    for pn in action_joint_order:
        if pn in pol_vals:
            out.append(pol_vals[pn])
        else:
            missing.append(pn)
            out.append(0.0)

    if strict and missing:
        raise KeyError(f"Missing joints in JointState for policy joint_order: {missing}")
    return np.asarray(out, dtype=np.float32)
