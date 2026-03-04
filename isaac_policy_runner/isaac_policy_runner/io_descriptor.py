from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import yaml


@dataclass(frozen=True)
class JointMapping:
    joint_names: List[str]
    pos_offsets: Dict[str, float]
    vel_offsets: Dict[str, float]
    action_joint_names: List[str]
    action_offsets: Dict[str, float]
    action_scale: float


@dataclass(frozen=True)
class ObservationLayout:
    slices: Dict[str, Tuple[int, int]]
    total_size: int


@dataclass(frozen=True)
class PolicyIO:
    observation: ObservationLayout
    action_size: int
    joint: JointMapping


def load_io_descriptors(path: str) -> PolicyIO:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    obs_terms = (data.get("observations") or {}).get("policy") or []
    act_terms = data.get("actions") or []
    if not obs_terms:
        raise ValueError("IO_descriptors.yaml: observations.policy is empty")
    if not act_terms:
        raise ValueError("IO_descriptors.yaml: actions is empty")

    slices: Dict[str, Tuple[int, int]] = {}
    cursor = 0
    for t in obs_terms:
        name = t.get("name")
        shape = t.get("shape") or []
        size = int(shape[0]) if shape else 0
        if not name or size <= 0:
            raise ValueError(f"Invalid observation term: name={name}, shape={shape}")
        slices[name] = (cursor, size)
        cursor += size

    act0 = act_terms[0]
    act_shape = act0.get("shape") or []
    action_size = int(act_shape[0]) if act_shape else 0
    if action_size <= 0:
        raise ValueError("Invalid action shape in IO_descriptors.yaml")

    # joint_pos_rel / joint_vel_rel info
    pos_term = next((t for t in obs_terms if t.get("name") == "joint_pos_rel"), None)
    vel_term = next((t for t in obs_terms if t.get("name") == "joint_vel_rel"), None)
    if pos_term is None or vel_term is None:
        raise ValueError("IO_descriptors.yaml must contain joint_pos_rel and joint_vel_rel")

    joint_names = list(pos_term.get("joint_names") or [])
    pos_offsets_list = list(pos_term.get("joint_pos_offsets") or [])
    vel_offsets_list = list(vel_term.get("joint_vel_offsets") or [])

    pos_offsets: Dict[str, float] = {j: float(o) for j, o in zip(joint_names, pos_offsets_list)}
    vel_offsets: Dict[str, float] = {j: float(o) for j, o in zip(joint_names, vel_offsets_list)}

    action_joint_names = list(act0.get("joint_names") or [])
    action_offsets_list = list(act0.get("offset") or [0.0] * action_size)
    action_offsets: Dict[str, float] = {j: float(o) for j, o in zip(action_joint_names, action_offsets_list)}
    action_scale = float(act0.get("scale", 1.0))

    return PolicyIO(
        observation=ObservationLayout(slices=slices, total_size=cursor),
        action_size=action_size,
        joint=JointMapping(
            joint_names=joint_names,
            pos_offsets=pos_offsets,
            vel_offsets=vel_offsets,
            action_joint_names=action_joint_names,
            action_offsets=action_offsets,
            action_scale=action_scale,
        ),
    )
