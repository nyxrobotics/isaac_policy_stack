from setuptools import setup

package_name = "isaac_policy_runtime"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Runtime node for executing Isaac Lab policies in ROS 2",
    license="BSD-3-Clause",

    entry_points={
        "console_scripts": [
            # ★ これが無いと launch できない
            "isaac_policy_runner = isaac_policy_runtime.policy_runner_node:main",
        ],
    },
)
