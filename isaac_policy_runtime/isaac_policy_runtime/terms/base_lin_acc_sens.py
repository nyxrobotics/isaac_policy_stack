from __future__ import annotations

import numpy as np
import math

from .base import TermBase
from .registry import register

def _quat_to_rot(qx, qy, qz, qw):
    # Returns rotation matrix from quaternion (x,y,z,w)
    # Right-handed, ROS convention.
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy+zz),     2*(xy - wz),     2*(xz + wy)],
        [    2*(xy + wz), 1 - 2*(xx+zz),     2*(yz - wx)],
        [    2*(xz - wy),     2*(yz + wx), 1 - 2*(xx+yy)],
    ], dtype=np.float32)

@register("base_lin_acc_sens")
class BaseLinAccSens(TermBase):
    def compute(self) -> np.ndarray:
        src = self.wiring["source"]
        msg = self.sources.get(src)
        if msg is None:
            return np.zeros((3,), dtype=np.float32)

        acc_field = self.wiring.get("vector_field", "linear_acceleration")
        ori_field = self.wiring.get("orientation_field", "orientation")
        provides_specific_force = bool(self.wiring.get("provides_specific_force", False))
        gmag = float(self.wiring.get("gravity_mag", 9.81))

        acc = getattr(msg, acc_field, None)
        ori = getattr(msg, ori_field, None)
        if acc is None:
            raise AttributeError(f"IMU message has no field '{acc_field}'")
        if ori is None:
            raise AttributeError(f"IMU message has no field '{ori_field}'")

        a = np.asarray([acc.x, acc.y, acc.z], dtype=np.float32)

        if provides_specific_force:
            return a

        # Compute gravity direction in IMU frame from orientation:
        # In ROS, IMU orientation is usually body->world rotation. Gravity in world is [0,0,-g].
        # Convert world gravity into body frame: g_b = R^T * g_w.
        R = _quat_to_rot(ori.x, ori.y, ori.z, ori.w)
        g_w = np.asarray([0.0, 0.0, -gmag], dtype=np.float32)
        g_b = R.T @ g_w

        # specific force = a - g (in body frame)
        # If the IMU already outputs linear_acceleration including gravity, this matches IsaacLab's (a - g).
        return (a - g_b).astype(np.float32, copy=False)
