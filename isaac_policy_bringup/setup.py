from setuptools import setup

package_name = "isaac_policy_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/bringup.launch.py"]),
        ("share/" + package_name + "/config", ["config/robots/example.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Launch files to run isaac_policy_runtime with a selected policy bundle.",
    license="BSD-3-Clause",
)
