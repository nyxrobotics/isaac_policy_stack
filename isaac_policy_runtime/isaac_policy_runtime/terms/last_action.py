from __future__ import annotations

import numpy as np

from .base import Term
from .registry import register

@register("last_action")
class LastAction(Term):
    def compute(self) -> np.ndarray:
        if self.state.last_action is None:
            # size equals action dimension for joint_position policies
            n = len(self.action_cfg.joint_order)
            return np.zeros((n,), dtype=np.float32)
        return self.state.last_action.astype(np.float32, copy=False)
