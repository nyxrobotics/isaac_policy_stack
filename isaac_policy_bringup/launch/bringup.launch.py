from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    bundle_path = LaunchConfiguration("bundle_path")
    use_tf = LaunchConfiguration("use_tf")
    strict = LaunchConfiguration("strict")
    log_io = LaunchConfiguration("log_io")

    return LaunchDescription([
        DeclareLaunchArgument("bundle_path", default_value=""),
        DeclareLaunchArgument("use_tf", default_value="true"),
        DeclareLaunchArgument("strict", default_value="true"),
        DeclareLaunchArgument("log_io", default_value="true"),
        Node(
            package="isaac_policy_runtime",
            executable="policy_runner",
            name="policy_runner",
            output="screen",
            parameters=[{
                "bundle_path": bundle_path,
                "use_tf": use_tf,
                "strict": strict,
                "log_io": log_io,
            }],
        ),
    ])
