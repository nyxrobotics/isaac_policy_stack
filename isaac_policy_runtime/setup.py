from setuptools import setup

package_name = "isaac_policy_runtime"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name,
              package_name + ".bundle",
              package_name + ".io",
              package_name + ".io.sources",
              package_name + ".terms",
              package_name + ".control",
              package_name + ".control.sinks"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/runner_defaults.yaml"]),
        ("share/" + package_name + "/launch", ["launch/policy_runner.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Generic ROS 2 runtime for Isaac Lab-exported policy bundles (ONNX + descriptors).",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "policy_runner = isaac_policy_runtime.policy_runner_node:main",
        ],
    },
)
