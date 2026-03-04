from setuptools import setup

package_name = 'isaac_runner_generator'

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
    description='Generator that creates per-robot Isaac policy runner packages for ROS 2 Humble.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'create_robot_runner = isaac_runner_generator.create_robot_runner:main',
        ],
    },
)
