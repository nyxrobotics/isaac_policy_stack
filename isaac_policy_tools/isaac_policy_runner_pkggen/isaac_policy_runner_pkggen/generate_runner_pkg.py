from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def _sanitize_pkg_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        raise ValueError("robot_name is empty after sanitization")
    return s


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _render_package_xml(pkg: str) -> str:
    return f'''<?xml version="1.0"?>
<package format="3">
  <name>{pkg}</name>
  <version>0.1.0</version>
  <description>Robot-specific wrapper package to launch isaac_policy_runtime with bundled policies.</description>

  <maintainer email="you@example.com">Your Name</maintainer>
  <license>BSD-3-Clause</license>

  <buildtool_depend>ament_python</buildtool_depend>

  <exec_depend>launch</exec_depend>
  <exec_depend>launch_ros</exec_depend>
  <exec_depend>isaac_policy_runtime</exec_depend>

  <export>
    <build_type>ament_python</build_type>
  </export>
</package>
'''


def _render_setup_py(pkg: str) -> str:
    # NOTE: We intentionally do NOT install bundle content here.
    # Bundle placement is handled by isaac_policy_export (separate command).
    return f'''from setuptools import setup

package_name = "{pkg}"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/isaac_policy_runner.launch.py"]),
        # We keep an empty bundle/ directory in source tree as a convention.
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Robot-specific wrapper package to launch isaac_policy_runtime with bundled policies.",
    license="BSD-3-Clause",
)
'''


def _render_setup_cfg() -> str:
    return '''[develop]
script_dir=$base/lib
[install]
install_scripts=$base/lib
'''


def _render_launch_py(pkg: str) -> str:
    return f'''from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    policy_name = LaunchConfiguration("policy_name")
    bundle_path = PathJoinSubstitution([
        FindPackageShare("{pkg}"),
        "bundle",
        policy_name,
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            "policy_name",
            description="Policy bundle name under share/{pkg}/bundle/<policy_name>.",
        ),
        Node(
            package="isaac_policy_runtime",
            executable="isaac_policy_runner",
            name="isaac_policy_runner",
            output="screen",
            parameters=[{{"bundle_path": bundle_path}}],
        ),
    ])
'''


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="isaac-generate-runner-pkg")
    ap.add_argument("--ws_src", required=True, help="Workspace src/ directory (e.g., ~/ros2_humble/src)")
    ap.add_argument("--robot_name", required=True, help="Robot name (e.g., kuroko)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing package directory")
    args = ap.parse_args(argv)

    ws_src = Path(args.ws_src).expanduser().resolve()
    if not ws_src.exists():
        raise FileNotFoundError(f"--ws_src not found: {ws_src}")

    robot = _sanitize_pkg_name(args.robot_name)
    pkg = f"{robot}_isaac_policy_runner"
    pkg_dir = ws_src / pkg

    if pkg_dir.exists():
        if not args.force:
            raise FileExistsError(f"{pkg_dir} exists (use --force)")
        shutil.rmtree(pkg_dir)

    # Create skeleton
    (pkg_dir / pkg).mkdir(parents=True, exist_ok=True)
    (pkg_dir / "launch").mkdir(parents=True, exist_ok=True)
    (pkg_dir / "bundle").mkdir(parents=True, exist_ok=True)  # placeholder dir
    (pkg_dir / "resource").mkdir(parents=True, exist_ok=True)

    _write_text(pkg_dir / "package.xml", _render_package_xml(pkg))
    _write_text(pkg_dir / "setup.py", _render_setup_py(pkg))
    _write_text(pkg_dir / "setup.cfg", _render_setup_cfg())
    _write_text(pkg_dir / pkg / "__init__.py", "")
    _write_text(pkg_dir / "launch" / "isaac_policy_runner.launch.py", _render_launch_py(pkg))
    _write_text(pkg_dir / "resource" / pkg, "")

    print(f"[OK] Generated package: {pkg_dir}")
    print(f"Build: colcon build --packages-select {pkg}")
    print(f"Run:   ros2 launch {pkg} isaac_policy_runner.launch.py policy_name:=<policy_name>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
