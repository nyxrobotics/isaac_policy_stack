from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from rclpy.node import Node

from ..bundle.schema import ActionConfig, ObservationSpec, RobotInterface
from ..io.source_manager import SourceManager


@dataclass
class RuntimeState:
    last_action: Optional[np.ndarray]


class TermBase:
    """Base class for all observation terms.

    A term computes one observation vector piece for the policy input. Terms are
    concatenated in the order given by io_descriptor.json (exported from Isaac Lab).
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

        # Term-specific wiring from robot_interface.yaml
        if obs_spec.name not in (robot_if.term_inputs or {}):
            raise KeyError(f"term_inputs missing for term '{obs_spec.name}'")

        self.wiring: Dict[str, Any] = dict((robot_if.term_inputs or {})[obs_spec.name] or {})
        src = self.wiring.get("source")
        if not src:
            raise KeyError(f"term_inputs[{obs_spec.name}].source is required")

        # Pseudo-source: some terms (e.g., last_action) do not require a ROS topic.
        # Use source: "internal" in robot_interface.yaml.
        if str(src) != "internal":
            self.sources.ensure_source(str(src))

    def compute(self) -> np.ndarray:
        raise NotImplementedError
