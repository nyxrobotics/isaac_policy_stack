from setuptools import setup

package_name = 'isaac_policy_runner'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Generic Isaac Lab policy runner node (ONNX) for ROS 2 Humble.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'isaac_policy_runner = isaac_policy_runner.node:main',
        ],
    },
)
