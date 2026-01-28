from __future__ import annotations

import numpy as np

from .base import TermBase
from .registry import register

@register("base_ang_vel")
class BaseAngVel(TermBase):
    def compute(self) -> np.ndarray:
        src = self.wiring["source"]
        msg = self.sources.get(src)
        if msg is None:
            return np.zeros((3,), dtype=np.float32)

        field = self.wiring.get("data_field") or self.wiring.get("vector_field", "angular_velocity")
        v = getattr(msg, field, None)
        if v is None:
            raise AttributeError(f"IMU message has no field '{field}'")

        vec = np.asarray([v.x, v.y, v.z], dtype=np.float32)
        # v1: output in IMU frame unless TF is added later
        return vec
