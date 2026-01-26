from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

@dataclass(frozen=True)
class ObservationSpec:
    name: str
    func: str
    params: Dict[str, Any] | None = None
    clip: Tuple[float, float] | None = None

@dataclass(frozen=True)
class ActionConfig:
    type: str
    joint_order: List[str]
    scale: float | None = None
    use_default_offset: bool | None = None
    clip: Tuple[float, float] = (-1.0, 1.0)

@dataclass(frozen=True)
class SourceSpec:
    msg_type: str
    topic: str

@dataclass(frozen=True)
class RobotInterface:
    version: int
    frames: Dict[str, str]
    sources: Dict[str, SourceSpec]
    joints: Dict[str, Any]
    term_inputs: Dict[str, Dict[str, Any]]
    control: Dict[str, Any]
