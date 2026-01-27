from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from rclpy.node import Node

from ..bundle.schema import ObservationSpec
from ..bundle.schema import ActionConfig
from ..bundle.schema import RobotInterface
from ..io.source_manager import SourceManager


@dataclass
class RuntimeState:
    last_action: Optional[np.ndarray]


class TermBase:
    """Base class for all observation terms.

    A term computes one observation vector piece for the policy input.
    """

    def __init__(
        self,
        node: Node,
        obs_spec: ObservationSpec,
        robot_if: RobotInterface,
        action_cfg: ActionConfig,
        sources: SourceManager,
        state: RuntimeState,
        use_tf: bool,
        strict: bool,
    ) -> None:
        self.node = node
        self.obs_spec = obs_spec
        self.robot_if = robot_if
        self.action_cfg = action_cfg
        self.sources = sources
        self.state = state
        self.use_tf = use_tf
        self.strict = strict

        # Each term may declare required sources in obs_spec.term_inputs.
        # We don't enforce it at construction time here; individual terms can.
        self.term_inputs = getattr(obs_spec, "term_inputs", None)

        # Term-specific wiring from robot_interface.yaml
        if obs_spec.name not in robot_if.term_inputs:
            raise KeyError(f"term_inputs missing for term '{obs_spec.name}'")
        self.wiring: Dict[str, Any] = dict(robot_if.term_inputs[obs_spec.name])

        # Ensure required source subscription exists
        src = self.wiring.get("source")
        if not src:
            raise KeyError(f"term_inputs[{obs_spec.name}].source is required")
        
    
        self.sources.ensure_source(src)
    def compute(self) -> np.ndarray:
        raise NotImplementedError