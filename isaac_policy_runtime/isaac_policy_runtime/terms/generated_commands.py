from __future__ import annotations

import numpy as np

from .base import TermBase
from .registry import register

def _get_twist_attr(msg, path: str) -> float:
    obj = msg
    for part in path.split('.'):
        obj = getattr(obj, part)
    return float(obj)

@register("generated_commands")
@register("velocity_commands")
class VelocityCommands(TermBase):
    def compute(self) -> np.ndarray:
        src = self.wiring["source"]
        msg = self.sources.get(src)
        if msg is None:
            return np.zeros((3,), dtype=np.float32)

        mapping = self.wiring.get("mapping", {
            "lin_x": "linear.x",
            "lin_y": "linear.y",
            "ang_z": "angular.z",
        })

        vx = _get_twist_attr(msg, mapping["lin_x"])
        vy = _get_twist_attr(msg, mapping["lin_y"])
        wz = _get_twist_attr(msg, mapping["ang_z"])

        clip = self.wiring.get("clip")
        if clip:
            vx = max(min(vx, clip.get("lin_x", [vx, vx])[1]), clip.get("lin_x", [vx, vx])[0])
            vy = max(min(vy, clip.get("lin_y", [vy, vy])[1]), clip.get("lin_y", [vy, vy])[0])
            wz = max(min(wz, clip.get("ang_z", [wz, wz])[1]), clip.get("ang_z", [wz, wz])[0])

        return np.asarray([vx, vy, wz], dtype=np.float32)
