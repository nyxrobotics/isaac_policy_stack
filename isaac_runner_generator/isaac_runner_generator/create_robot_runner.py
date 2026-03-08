
from __future__ import annotations

import argparse
from pathlib import Path


PKGXML_TEMPLATE = """<?xml version="1.0"?>
<package format="3">
  <name>{pkg}</name>
  <version>0.1.0</version>
  <description>Robot-specific Isaac policy runner bridges for {robot}.</description>
  <maintainer email="user@example.com">user</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_python</buildtool_depend>

  <exec_depend>rclpy</exec_depend>
  <exec_depend>std_msgs</exec_depend>
  <exec_depend>sensor_msgs</exec_depend>
  <exec_depend>nav_msgs</exec_depend>
  <exec_depend>geometry_msgs</exec_depend>
  <exec_depend>rcl_interfaces</exec_depend>
  <exec_depend>isaac_policy_runner</exec_depend>

  <exec_depend>launch</exec_depend>
  <exec_depend>launch_ros</exec_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
"""


SETUPCFG_TEMPLATE = """[develop]
script_dir=$base/lib/{pkg}
[install]
install_scripts=$base/lib/{pkg}
"""


SETUPPY_TEMPLATE = """from setuptools import setup

package_name = '{pkg}'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/{robot}_isaac_runner.launch.py']),
        ('share/' + package_name + '/model', [
            'model/IO_descriptors.yaml',
            'model/policy.onnx',
            'model/policy.pt',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Robot-specific Isaac policy runner bridges.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            '{robot}_observations_bridge_node = {pkg}.{robot}_observations_bridge_node:main',
            '{robot}_actions_bridge_node = {pkg}.{robot}_actions_bridge_node:main',
        ],
    },
)
"""


LAUNCH_TEMPLATE = r"""from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


# Remap table (edit the *right-hand side* as needed for your robot)
REMAPS = [
    ('/odom', '/odom'),
    ('/imu', '/imu'),
    ('/cmd_vel', '/cmd_vel'),
    ('/joint_states', '/joint_states'),
    ('/policy/observations', '/policy/observations'),
    ('/policy/actions', '/policy/actions'),
    ('/joint_group_position_controller/commands', '/joint_group_position_controller/commands'),
]


def generate_launch_description():
    model_dir = PathJoinSubstitution([FindPackageShare('{pkg}'), 'model'])

    return LaunchDescription([
        Node(
            package='{pkg}',
            executable='{robot}_observations_bridge_node',
            name='{robot}_observations_bridge_node',
            output='screen',
            parameters=[
                {'model_dir': model_dir},
                # Debug prints (similar to Isaac Lab)
                {'debug_print': True, 'debug_every_n': 1},
            ],
            remappings=REMAPS,
        ),
        Node(
            package='isaac_policy_runner',
            executable='isaac_policy_runner',
            name='isaac_policy_runner',
            output='screen',
            parameters=[{'model_dir': model_dir, 'use_internal_action_observation': True, 'debug_print': True, 'debug_every_n': 1}],
            remappings=REMAPS,
        ),
        Node(
            package='{pkg}',
            executable='{robot}_actions_bridge_node',
            name='{robot}_actions_bridge_node',
            output='screen',
            parameters=[
                {'model_dir': model_dir},
                # Hold default posture briefly at startup to match Isaac Lab reset
                {'startup_hold_sec': 1.0},
                # Debug prints (similar to Isaac Lab)
                {'debug_print': True, 'debug_every_n': 1},
            ],
            remappings=REMAPS,
        ),
    ])
"""


OBS_BRIDGE_TEMPLATE = r'''from __future__ import annotations

import datetime
import os
import numpy as np
import yaml

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray


def quat_to_rotmat(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1.0 - 2.0*(yy+zz), 2.0*(xy-wz),       2.0*(xz+wy)],
        [2.0*(xy+wz),       1.0 - 2.0*(xx+zz), 2.0*(yz-wx)],
        [2.0*(xz-wy),       2.0*(yz+wx),       1.0 - 2.0*(xx+yy)],
    ], dtype=np.float32)


class {Robot}ObservationsBridge(Node):
    def __init__(self) -> None:
        super().__init__('{robot}_observations_bridge_node')

        self.declare_parameter('model_dir', '')
        self.declare_parameter('debug_print', True)
        self.declare_parameter('debug_every_n', 1)
        self.declare_parameter('odom_twist_in_world_frame', False)
        self.debug_print = self.get_parameter('debug_print').get_parameter_value().bool_value
        self.debug_every_n = int(self.get_parameter('debug_every_n').get_parameter_value().integer_value) or 1
        self.odom_twist_in_world_frame = self.get_parameter('odom_twist_in_world_frame').get_parameter_value().bool_value
        self._debug_count = 0
        model_dir = self.get_parameter('model_dir').get_parameter_value().string_value
        if not model_dir:
            raise RuntimeError("Parameter 'model_dir' is required")

        io_path = os.path.join(model_dir, 'IO_descriptors.yaml')
        with open(io_path, 'r', encoding='utf-8') as f:
            self.io = yaml.safe_load(f) or {}

        # Debug: show observation joint order from IO_descriptors.yaml
        obs_terms = (self.io.get('observations') or {}).get('policy') or []
        pos_term = next((t for t in obs_terms if t.get('name') == 'joint_pos_rel'), None)
        self.policy_joint_names = list((pos_term or {}).get('joint_names') or [])
        if self.policy_joint_names:
            self.get_logger().info('IO observation joint_pos_rel order: ' + ', '.join(self.policy_joint_names))

        self.obs_terms = (self.io.get('observations') or {}).get('policy') or []
        if not self.obs_terms:
            raise RuntimeError('IO_descriptors.yaml: observations.policy is empty')

        # term lookup for overloads
        self.obs_term_by_name = {t.get('name'): t for t in self.obs_terms if t.get('name')}

        self.obs_slices = self._build_obs_slices(self.obs_terms)
        self.obs_size = sum(size for (_, size) in self.obs_slices.values())
        scene = self.io.get('scene') or {}
        if 'dt' in scene and scene['dt'] is not None:
            self.publish_dt = float(scene['dt'])
        else:
            physics_dt = float(scene.get('physics_dt', 0.005))
            decimation = int(scene.get('decimation', 1))
            self.publish_dt = physics_dt * decimation
        if self.publish_dt <= 0.0:
            self.publish_dt = 0.02


        act0 = (self.io.get('actions') or [{}])[0]
        self.act_size = int(((act0.get('shape') or [0])[0]) or 0)
        self.action_joint_names = list(act0.get('joint_names') or [])
        if self.action_joint_names:
            self.get_logger().info('IO action joint order: ' + ', '.join(self.action_joint_names))
        if self.action_joint_names:
            self.get_logger().info('IO action joint order: ' + ', '.join(self.action_joint_names))

        # caches
        self.base_lin_vel = np.zeros((3,), dtype=np.float32)
        self.odom_linear = np.zeros((3,), dtype=np.float32)
        self.base_ang_vel = np.zeros((3,), dtype=np.float32)
        self.projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self.rotation_body_to_world = np.eye(3, dtype=np.float32)
        self.generated_commands = np.zeros((3,), dtype=np.float32)
        self.last_action = np.zeros((self.act_size,), dtype=np.float32)

        # joint mapping (policy order)
        self.joint_pos_term = next((t for t in self.obs_terms if t.get('name') == 'joint_pos_rel'), None)
        self.joint_vel_term = next((t for t in self.obs_terms if t.get('name') == 'joint_vel_rel'), None)
        if self.joint_pos_term is None or self.joint_vel_term is None:
            raise RuntimeError("IO_descriptors.yaml must contain joint_pos_rel and joint_vel_rel")

        self.policy_joint_pos_names = list(self.joint_pos_term.get('joint_names') or [])
        self.policy_joint_vel_names = list(self.joint_vel_term.get('joint_names') or self.policy_joint_pos_names)
        self.policy_joint_names = list(self.policy_joint_pos_names)
        if self.policy_joint_pos_names:
            self.get_logger().info('IO observation joint_pos_rel order: ' + ', '.join(self.policy_joint_pos_names))
        if self.policy_joint_vel_names:
            self.get_logger().info('IO observation joint_vel_rel order: ' + ', '.join(self.policy_joint_vel_names))

        # Prefer per-term offsets from observation descriptors. If they are missing,
        # fall back to articulation defaults matched by joint name.
        art = ((self.io.get('articulations') or {}).get('robot') or {})
        art_joint_names = list(art.get('joint_names') or [])
        art_default_joint_pos = list(art.get('default_joint_pos') or [])
        art_default_joint_vel = list(art.get('default_joint_vel') or [])
        self.art_default_joint_pos_by_name = {
            name: float(art_default_joint_pos[i])
            for i, name in enumerate(art_joint_names)
            if i < len(art_default_joint_pos)
        }
        self.art_default_joint_vel_by_name = {
            name: float(art_default_joint_vel[i])
            for i, name in enumerate(art_joint_names)
            if i < len(art_default_joint_vel)
        }

        term_pos_offsets = list(self.joint_pos_term.get('joint_pos_offsets') or [])
        term_vel_offsets = list(self.joint_vel_term.get('joint_vel_offsets') or [])
        self.policy_joint_pos_offset_by_name = {}
        for i, jname in enumerate(self.policy_joint_pos_names):
            if i < len(term_pos_offsets) and term_pos_offsets[i] is not None:
                self.policy_joint_pos_offset_by_name[jname] = float(term_pos_offsets[i])
            else:
                self.policy_joint_pos_offset_by_name[jname] = float(self.art_default_joint_pos_by_name.get(jname, 0.0))

        self.policy_joint_vel_offset_by_name = {}
        for i, jname in enumerate(self.policy_joint_vel_names):
            if i < len(term_vel_offsets) and term_vel_offsets[i] is not None:
                self.policy_joint_vel_offset_by_name[jname] = float(term_vel_offsets[i])
            else:
                self.policy_joint_vel_offset_by_name[jname] = float(self.art_default_joint_vel_by_name.get(jname, 0.0))

        self.pub = self.create_publisher(Float32MultiArray, '/policy/observations', 10)

        self.create_subscription(Odometry, '/odom', self._cb_odom, 10)
        self.create_subscription(Imu, '/imu', self._cb_imu, 10)
        self.create_subscription(Twist, '/cmd_vel', self._cb_cmd_vel, 10)
        self.create_subscription(JointState, '/joint_states', self._cb_joint_states, 10)
        self.create_subscription(Float32MultiArray, '/policy/actions', self._cb_last_action, 10)
        self.have_odom = False
        self.have_imu = False
        self.have_cmd_vel = False
        self.have_joint_states = False
        self.latest_joint_state = None
        self._warned_missing_inputs = False
        self._warned_missing_joints = False
        self.input_topics_updated_ = False
        self.input_topics_received_ = False

        self.timer = self.create_timer(self.publish_dt, self._on_timer_publish)


        self.get_logger().info(
            f'Observation bridge ready: obs_size={self.obs_size}, publish_hz={1.0/self.publish_dt:.2f}, ' 
            f'odom_twist_in_world_frame={self.odom_twist_in_world_frame}'
        )

    def _build_obs_slices(self, terms):
        slices = {}
        cursor = 0
        for t in terms:
            name = t.get('name')
            shape = t.get('shape') or []
            size = int(shape[0]) if shape else 0
            if not name or size <= 0:
                raise RuntimeError(f'Invalid observation term: {name} / {shape}')
            slices[name] = (cursor, size)
            cursor += size
        return slices

    def _mark_input_topics_updated(self) -> None:
        self.input_topics_updated_ = True
        # Required inputs for observation publication. Optional terms like
        # cmd_vel / last_action should default to zeros until they arrive.
        self.input_topics_received_ = bool(
            self.have_odom and self.have_imu and self.have_joint_states
        )

    def _apply_overloads(self, term: str, vec: np.ndarray) -> np.ndarray:
        """Apply IO_descriptors.yaml observation overloads (scale/clip).

        Isaac Lab applies these overloads after term computation. We mirror that here.
        Supported forms:
          - scale: None | float | list[float]
          - clip: None | float (symmetric) | [min, max] | list[[min,max], ...]
        """
        t = self.obs_term_by_name.get(term)
        if not t:
            return vec
        ov = t.get('overloads') or {}

        out = vec.astype(np.float32, copy=True)

        scale = ov.get('scale', None)
        if scale is not None:
            if isinstance(scale, (int, float)):
                out *= float(scale)
            elif isinstance(scale, (list, tuple)):
                s = np.asarray(scale, dtype=np.float32)
                if s.size == out.size:
                    out *= s.reshape(out.shape)

        clip = ov.get('clip', None)
        if clip is not None:
            if isinstance(clip, (int, float)):
                c = float(clip)
                out = np.clip(out, -c, c)
            elif isinstance(clip, (list, tuple)):
                # [min, max]
                if len(clip) == 2 and all(isinstance(x, (int, float)) for x in clip):
                    out = np.clip(out, float(clip[0]), float(clip[1]))
                else:
                    # per-element [[min,max], ...]
                    c = np.asarray(clip, dtype=np.float32)
                    if c.shape == (out.size, 2):
                        lo = c[:, 0].reshape(out.shape)
                        hi = c[:, 1].reshape(out.shape)
                        out = np.minimum(np.maximum(out, lo), hi)
        return out

    def _cb_odom(self, msg: Odometry) -> None:
        tw = msg.twist.twist
        self.base_lin_vel[:] = [tw.linear.x, tw.linear.y, tw.linear.z]
        self.have_odom = True
        self._mark_input_topics_updated()

    def _cb_imu(self, msg: Imu) -> None:
        av = msg.angular_velocity
        self.base_ang_vel[:] = [av.x, av.y, av.z]

        q = msg.orientation
        R = quat_to_rotmat(q.x, q.y, q.z, q.w)

        # NOTE: adjust if your IMU frame differs
        # R is a rotation matrix, so R @ [0, 0, -1] is guaranteed to have norm 1.
        # Therefore, explicit normalization is unnecessary.
        self.projected_gravity[:] = (-R[:, 2])
        self.have_imu = True
        self._mark_input_topics_updated()

    def _cb_cmd_vel(self, msg: Twist) -> None:
        self.generated_commands[:] = [msg.linear.x, msg.linear.y, msg.angular.z]
        self.have_cmd_vel = True
        self._mark_input_topics_updated()

    def _cb_last_action(self, msg: Float32MultiArray) -> None:
        arr = np.asarray(msg.data, dtype=np.float32)
        if arr.size == self.act_size:
            self.last_action[:] = arr

    def _cb_joint_states(self, msg: JointState) -> None:
        self.latest_joint_state = msg
        self.have_joint_states = True
        self._mark_input_topics_updated()

    def _canonical_joint_name(self, name: str) -> str:
        if not name:
            return ''
        out = str(name).strip()
        if '/' in out:
            out = out.split('/')[-1]
        return out

    def _build_joint_state_lookup(self, msg: JointState) -> dict[str, int]:
        lookup = {}
        for i, name in enumerate(msg.name):
            lookup.setdefault(name, i)
            canonical = self._canonical_joint_name(name)
            lookup.setdefault(canonical, i)
        return lookup

    def _on_timer_publish(self) -> None:
        if not self.input_topics_received_:
            if not self._warned_missing_inputs:
                missing_inputs = []
                if not self.have_odom:
                    missing_inputs.append('odom')
                if not self.have_imu:
                    missing_inputs.append('imu')
                if not self.have_joint_states:
                    missing_inputs.append('joint_states')
                if missing_inputs:
                    self.get_logger().warning(
                        'Waiting for required inputs: ' + ', '.join(missing_inputs)
                    )
                self._warned_missing_inputs = True
            return

        if not self.input_topics_updated_:
            return

        # Publish once required inputs are ready. Optional terms such as
        # cmd_vel / last_action stay at zeros until they arrive.
        if self._warned_missing_inputs and (self.have_odom and self.have_imu and self.have_joint_states):
            self.get_logger().info('Required inputs ready. Starting observation publication; optional inputs default to zeros until received.')
            self._warned_missing_inputs = False

        msg = self.latest_joint_state
        name_to_idx = self._build_joint_state_lookup(msg) if msg is not None else {}

        joint_pos_rel = np.zeros((len(self.policy_joint_pos_names),), dtype=np.float32)
        joint_vel_rel = np.zeros((len(self.policy_joint_vel_names),), dtype=np.float32)

        missing_pos = []
        for k, jname in enumerate(self.policy_joint_pos_names):
            lookup_name = jname if jname in name_to_idx else self._canonical_joint_name(jname)
            if lookup_name not in name_to_idx:
                missing_pos.append(jname)
                continue
            i = name_to_idx[lookup_name]
            pos = float(msg.position[i]) if i < len(msg.position) else 0.0
            pos0 = float(self.policy_joint_pos_offset_by_name.get(jname, self.art_default_joint_pos_by_name.get(jname, 0.0)))
            joint_pos_rel[k] = pos - pos0

        missing_vel = []
        for k, jname in enumerate(self.policy_joint_vel_names):
            lookup_name = jname if jname in name_to_idx else self._canonical_joint_name(jname)
            if lookup_name not in name_to_idx:
                missing_vel.append(jname)
                continue
            i = name_to_idx[lookup_name]
            vel = float(msg.velocity[i]) if i < len(msg.velocity) else 0.0
            vel0 = float(self.policy_joint_vel_offset_by_name.get(jname, self.art_default_joint_vel_by_name.get(jname, 0.0)))
            joint_vel_rel[k] = vel - vel0

        missing = []
        for name in missing_pos + missing_vel:
            if name not in missing:
                missing.append(name)

        if missing and not self._warned_missing_joints:
            self.get_logger().warning('Missing joints in /joint_states: ' + ', '.join(missing))
            self._warned_missing_joints = True

        obs = np.zeros((self.obs_size,), dtype=np.float32)

        def put(term: str, vec: np.ndarray):
            if term not in self.obs_slices:
                return
            start, size = self.obs_slices[term]
            if vec.size != size:
                raise RuntimeError(f'Term size mismatch for {term}: {vec.size} != {size}')
            obs[start:start+size] = self._apply_overloads(term, vec)

        put('base_lin_vel', self.base_lin_vel)
        put('base_ang_vel', self.base_ang_vel)
        put('projected_gravity', self.projected_gravity)
        put('generated_commands', self.generated_commands)
        put('joint_pos_rel', joint_pos_rel)
        put('joint_vel_rel', joint_vel_rel)
        put('last_action', self.last_action)

        if self.debug_print:
            self._debug_count += 1
            if (self._debug_count % self.debug_every_n) == 0:
                now = datetime.datetime.now().strftime("%H:%M:%S.%f")
                self.get_logger().info(f'[{now}] [policy] observations: {obs}')
                self.get_logger().info(f'  base_lin_vel: {self.base_lin_vel}')
                self.get_logger().info(f'  base_ang_vel: {self.base_ang_vel}')
                self.get_logger().info(f'  projected_gravity: {self.projected_gravity}')
                self.get_logger().info(f'  generated_commands: {self.generated_commands}')
                self.get_logger().info(f'  joint_pos_rel: {joint_pos_rel}')
                self.get_logger().info(f'  joint_vel_rel: {joint_vel_rel}')
                self.get_logger().info(f'  last_action: {self.last_action}')

        out = Float32MultiArray()
        out.data = obs.tolist()
        self.pub.publish(out)
        self.input_topics_updated_ = False



def main() -> None:
    rclpy.init()
    node = {Robot}ObservationsBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
'''


ACT_BRIDGE_TEMPLATE = r"""from __future__ import annotations

import os
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterType

from std_msgs.msg import Float32MultiArray, Float64MultiArray


class {Robot}ActionsBridge(Node):
    def __init__(self) -> None:
        super().__init__('{robot}_actions_bridge_node')

        self.declare_parameter('model_dir', '')
        self.declare_parameter('startup_hold_sec', 1.0)
        self.declare_parameter('debug_print', True)
        self.declare_parameter('debug_every_n', 1)
        self.debug_print = self.get_parameter('debug_print').get_parameter_value().bool_value
        self.debug_every_n = int(self.get_parameter('debug_every_n').get_parameter_value().integer_value) or 1
        self._debug_count = 0
        model_dir = self.get_parameter('model_dir').get_parameter_value().string_value
        if not model_dir:
            raise RuntimeError("Parameter 'model_dir' is required")

        self.startup_hold_sec = float(self.get_parameter('startup_hold_sec').get_parameter_value().double_value)

        io_path = os.path.join(model_dir, 'IO_descriptors.yaml')
        with open(io_path, 'r', encoding='utf-8') as f:
            self.io = yaml.safe_load(f) or {}

        # Debug: show observation joint order from IO_descriptors.yaml
        obs_terms = (self.io.get('observations') or {}).get('policy') or []
        pos_term = next((t for t in obs_terms if t.get('name') == 'joint_pos_rel'), None)
        self.policy_joint_names = list((pos_term or {}).get('joint_names') or [])
        if self.policy_joint_names:
            self.get_logger().info('IO observation joint_pos_rel order: ' + ', '.join(self.policy_joint_names))

        # ----- Isaac Lab IO parsing (matches exported IO_descriptors.yaml) -----
        act0 = (self.io.get('actions') or [{}])[0]
        self.action_joint_names = list(act0.get('joint_names') or [])
        if self.action_joint_names:
            self.get_logger().info('IO action joint order: ' + ', '.join(self.action_joint_names))
        self.action_size = int(((act0.get('shape') or [len(self.action_joint_names)])[0]) or len(self.action_joint_names))
        self.action_full_path = str(act0.get('full_path') or '')
        self.action_name = str(act0.get('name') or '')
        self.action_offsets = list(act0.get('offset') or [0.0] * self.action_size)
        scale_cfg = act0.get('scale', 1.0)
        if isinstance(scale_cfg, (int, float)):
            self.action_scale = float(scale_cfg)
        elif isinstance(scale_cfg, (list, tuple)):
            self.action_scale = [float(x) for x in scale_cfg]
        else:
            self.action_scale = 1.0
        self.action_clip = act0.get('clip', None)
        alpha_cfg = act0.get('alpha', 1.0)
        if isinstance(alpha_cfg, (int, float)):
            self.action_alpha = float(alpha_cfg)
        elif isinstance(alpha_cfg, (list, tuple)):
            self.action_alpha = [float(x) for x in alpha_cfg]
        else:
            self.action_alpha = 1.0

        self.is_joint_position_to_limits = (
            'JointPositionToLimitsAction' in self.action_full_path
            or 'EMAJointPositionToLimitsAction' in self.action_full_path
        )
        self.is_ema_to_limits = 'EMAJointPositionToLimitsAction' in self.action_full_path

        art = (self.io.get('articulations') or {}).get('robot') or {}
        self.articulation_joint_names = list(art.get('joint_names') or [])
        self.default_joint_pos = list(art.get('default_joint_pos') or [])
        if len(self.articulation_joint_names) != len(self.default_joint_pos):
            raise RuntimeError(
                'articulations.robot.joint_names and default_joint_pos length mismatch: '
                f'{len(self.articulation_joint_names)} vs {len(self.default_joint_pos)}'
            )

        self.default_pos_by_name = {n: float(p) for n, p in zip(self.articulation_joint_names, self.default_joint_pos)}

        joint_limits = list(art.get('default_joint_pos_limits') or [])
        self.default_joint_limits_by_name = {}
        for n, lim in zip(self.articulation_joint_names, joint_limits):
            if isinstance(lim, (list, tuple)) and len(lim) >= 2:
                self.default_joint_limits_by_name[n] = (float(lim[0]), float(lim[1]))

        if self.is_joint_position_to_limits:
            self.get_logger().info(
                'Action target formula: target = unscale(clamp(raw * scale, -1, 1), joint_limits)'
            )
            if self.is_ema_to_limits:
                self.get_logger().info('EMA smoothing is enabled for action targets')
        else:
            self.get_logger().info(
                'Action target formula: target = raw * scale + offset (no extra default_joint_pos added)'
            )

        # Controller joint order is authoritative for /commands message layout.
        self.controller_joints = self._wait_for_controller_joints(
            node_name='/joint_group_position_controller',
            param_name='joints'
        )
        self.get_logger().info('Controller joints order: ' + ', '.join(self.controller_joints))

        self.action_index_by_name = {n: i for i, n in enumerate(self.action_joint_names)}

        # Precompute how to fill each controller joint:
        # - If joint is in actions: use policy output
        # - Else if joint is in articulations: ALWAYS use default_joint_pos (requested behavior)
        # - Else: warn and output 0.0
        self.ctrl_rule = []
        for j in self.controller_joints:
            if j in self.action_index_by_name:
                self.ctrl_rule.append(('action', int(self.action_index_by_name[j]), j))
            elif j in self.default_pos_by_name:
                self.ctrl_rule.append(('default', None, j))
            else:
                self.get_logger().warning(f"Controller joint '{j}' not found in IO_descriptors articulations; sending 0.0")
                self.ctrl_rule.append(('zero', None, j))

        self.pub = self.create_publisher(Float64MultiArray, '/joint_group_position_controller/commands', 10)
        self.sub = self.create_subscription(Float32MultiArray, '/policy/actions', self._cb_action, 10)

        self._prev_targets_by_action_index = {}
        for joint_name, act_i in self.action_index_by_name.items():
            self._prev_targets_by_action_index[int(act_i)] = float(self.default_pos_by_name.get(joint_name, 0.0))

        self._t0 = self.get_clock().now()

    def _wait_for_controller_joints(self, node_name: str, param_name: str):
        client = self.create_client(GetParameters, f'{node_name}/get_parameters')
        while not client.wait_for_service(timeout_sec=0.5):
            self.get_logger().info('Waiting for controller parameter service...')

        req = GetParameters.Request()
        req.names = [param_name]
        while True:
            fut = client.call_async(req)
            rclpy.spin_until_future_complete(self, fut)
            res = fut.result()
            if res is None or len(res.values) == 0:
                self.get_logger().info('Controller parameters not ready yet...')
                continue
            v = res.values[0]
            if v.type != ParameterType.PARAMETER_STRING_ARRAY:
                self.get_logger().warning(f'Controller joints param has unexpected type: {v.type}; retrying...')
                continue
            joints = list(v.string_array_value)
            if joints:
                return joints
            self.get_logger().info('Controller joints list empty; retrying...')

    def _clip_processed_action(self, processed: float, act_i: int) -> float:
        if self.action_clip is None:
            return processed

        c = self.action_clip
        try:
            if isinstance(c, (int, float)):
                bound = abs(float(c))
                return min(max(processed, -bound), bound)
            if isinstance(c, (list, tuple)) and act_i < len(c) and isinstance(c[act_i], (list, tuple)) and len(c[act_i]) >= 2:
                cmin = float(c[act_i][0])
                cmax = float(c[act_i][1])
                return min(max(processed, cmin), cmax)
            if isinstance(c, (list, tuple)) and len(c) >= 2 and not isinstance(c[0], (list, tuple)):
                cmin = float(c[0])
                cmax = float(c[1])
                return min(max(processed, cmin), cmax)
        except Exception:
            pass
        return processed

    def _scale_value(self, cfg, act_i: int, default: float = 1.0) -> float:
        if isinstance(cfg, (list, tuple)):
            if act_i < len(cfg):
                return float(cfg[act_i])
            return float(default)
        return float(cfg)

    def _offset_value(self, act_i: int, default: float = 0.0) -> float:
        if isinstance(self.action_offsets, (list, tuple)):
            if act_i < len(self.action_offsets):
                return float(self.action_offsets[act_i])
            return float(default)
        return float(self.action_offsets)

    def _compute_action_pipeline(self, act: np.ndarray, act_i: int) -> dict:
        a_raw = float(act[act_i])
        scale = self._scale_value(self.action_scale, act_i, 1.0)
        offset = self._offset_value(act_i, 0.0)
        scaled = a_raw * scale

        out = {
            'raw': a_raw,
            'scale': scale,
            'offset': offset,
            'scaled': scaled,
            'offset_applied': scaled,
            'clipped': scaled,
            'normalized': scaled,
            'target_pre_ema': scaled,
        }

        if self.is_joint_position_to_limits:
            clipped = self._clip_processed_action(scaled, act_i)
            normalized = min(max(clipped, -1.0), 1.0)
            target = normalized
            joint_name = self.action_joint_names[act_i] if act_i < len(self.action_joint_names) else None
            if joint_name is not None and joint_name in self.default_joint_limits_by_name:
                lo, hi = self.default_joint_limits_by_name[joint_name]
                target = 0.5 * (normalized + 1.0) * (hi - lo) + lo
            out['clipped'] = clipped
            out['normalized'] = normalized
            out['target_pre_ema'] = target
            return out

        offset_applied = scaled + offset
        clipped = self._clip_processed_action(offset_applied, act_i)
        out['offset_applied'] = offset_applied
        out['clipped'] = clipped
        out['normalized'] = clipped
        out['target_pre_ema'] = clipped
        return out

    def _action_to_target(self, act: np.ndarray, act_i: int) -> float:
        return float(self._compute_action_pipeline(act, act_i)['target_pre_ema'])

    def _default_targets(self) -> list[float]:
        # Default command for ALL controller joints.
        out: list[float] = []
        for _kind, _idx, j in self.ctrl_rule:
            out.append(float(self.default_pos_by_name.get(str(j), 0.0)))
        return out

    def _compute_targets_from_action(self, act: np.ndarray) -> list[float]:
        targets: list[float] = []
        for kind, idx, j in self.ctrl_rule:
            if kind == 'action':
                target = self._action_to_target(act, int(idx))
                if self.is_ema_to_limits:
                    alpha = self._scale_value(self.action_alpha, int(idx), 1.0)
                    alpha = min(max(alpha, 0.0), 1.0)
                    prev = float(self._prev_targets_by_action_index.get(int(idx), self.default_pos_by_name.get(str(j), 0.0)))
                    target = alpha * target + (1.0 - alpha) * prev
                    lim = self.default_joint_limits_by_name.get(str(j))
                    if lim is not None:
                        target = min(max(target, float(lim[0])), float(lim[1]))
                    self._prev_targets_by_action_index[int(idx)] = float(target)
                targets.append(target)
            elif kind == 'default':
                targets.append(float(self.default_pos_by_name.get(str(j), 0.0)))
            else:
                targets.append(0.0)
        return targets

    def _publish_targets(self, targets: list[float]) -> None:
        out = Float64MultiArray()
        out.data = [float(x) for x in targets]
        self.pub.publish(out)


    def _cb_action(self, msg: Float32MultiArray) -> None:
        act = np.asarray(msg.data, dtype=np.float32)
        if act.size != self.action_size:
            self.get_logger().warning(f'Action size mismatch: got {act.size}, expected {self.action_size}')
            return


        if self.debug_print:
            self._debug_count += 1
            if (self._debug_count % self.debug_every_n) == 0:
                self.get_logger().info(f"[policy] actions: {act}")
                pipelines = [self._compute_action_pipeline(act, i) for i in range(self.action_size)]
                target_pre_ema = np.array([p['target_pre_ema'] for p in pipelines], dtype=np.float32)
                self.get_logger().info(f"target_pre_ema: {target_pre_ema}")
                for i, p in enumerate(pipelines):
                    joint_name = self.action_joint_names[i] if i < len(self.action_joint_names) else f'joint_{i}'
                    self.get_logger().info(
                        f"  {joint_name}: raw={p['raw']:.6f}, scale={p['scale']:.6f}, offset={p['offset']:.6f}, "
                        f"scaled={p['scaled']:.6f}, offset_applied={p['offset_applied']:.6f}, "
                        f"clipped={p['clipped']:.6f}, normalized={p['normalized']:.6f}, "
                        f"target_pre_ema={p['target_pre_ema']:.6f}"
                    )

        targets = self._compute_targets_from_action(act)
        now = self.get_clock().now()
        t = (now - self._t0).nanoseconds * 1e-9
        if t < self.startup_hold_sec:
            self._publish_targets(self._default_targets())
        else:
            self._publish_targets(targets)


def main() -> None:
    rclpy.init()
    node = {Robot}ActionsBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
"""


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def _render(template: str, **kwargs) -> str:
    """Render template by literal token replacement.

    This avoids str.format() interpreting braces inside code snippets (e.g., f-strings, dicts).
    Only occurrences of {key} for provided keys are replaced.
    """
    out = template
    for k, v in kwargs.items():
        out = out.replace('{' + k + '}', str(v))
    return out


def create_robot_pkg(robot: str, out_dir: Path) -> Path:
    pkg = f"{robot}_isaac_runner"
    pkg_dir = out_dir / pkg
    if pkg_dir.exists():
        raise RuntimeError(f"Target already exists: {pkg_dir}")

    (pkg_dir / 'resource').mkdir(parents=True, exist_ok=True)
    (pkg_dir / pkg).mkdir(parents=True, exist_ok=True)
    (pkg_dir / 'launch').mkdir(parents=True, exist_ok=True)
    (pkg_dir / 'model').mkdir(parents=True, exist_ok=True)

    _write(pkg_dir / 'package.xml', _render(PKGXML_TEMPLATE, pkg=pkg, robot=robot))
    _write(pkg_dir / 'setup.cfg', _render(SETUPCFG_TEMPLATE, pkg=pkg))
    _write(pkg_dir / 'setup.py', _render(SETUPPY_TEMPLATE, pkg=pkg, robot=robot))
    _write(pkg_dir / 'resource' / pkg, '')
    _write(pkg_dir / pkg / '__init__.py', '')

    Robot = ''.join([p.capitalize() for p in robot.split('_')])
    _write(pkg_dir / pkg / f'{robot}_observations_bridge_node.py',
           _render(OBS_BRIDGE_TEMPLATE, robot=robot, Robot=Robot))
    _write(pkg_dir / pkg / f'{robot}_actions_bridge_node.py',
           _render(ACT_BRIDGE_TEMPLATE, robot=robot, Robot=Robot))

    _write(pkg_dir / 'launch' / f'{robot}_isaac_runner.launch.py',
           _render(LAUNCH_TEMPLATE, pkg=pkg, robot=robot))

    _write(pkg_dir / 'model' / 'IO_descriptors.yaml', '# Replace with Isaac Lab exported IO_descriptors.yaml\n')
    _write(pkg_dir / 'model' / 'policy.onnx', '')
    _write(pkg_dir / 'model' / 'policy.pt', '')

    return pkg_dir


def main() -> None:
    parser = argparse.ArgumentParser(description='Create a per-robot Isaac runner package.')
    parser.add_argument('--robot', required=True, help='robot name (e.g. my_bot)')
    parser.add_argument('--out', default='.', help='output directory (default: current directory)')
    args = parser.parse_args()

    robot = args.robot.strip()
    if not robot or not robot.replace('_', '').isalnum():
        raise SystemExit('Invalid robot name. Use letters/numbers/underscore.')

    out_dir = Path(args.out).resolve()
    pkg_dir = create_robot_pkg(robot, out_dir)
    print(f'Created: {pkg_dir}')
