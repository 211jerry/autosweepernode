import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    config_path = os.path.join(
        get_package_share_directory('autosweeper_robot'),
        'config',
        'sweeper_config.yaml'
    )

    sweeper_node = Node(
        package='autosweeper_robot',
        executable='sweeper_node',
        name='sweeper_node',
        parameters=[config_path],
        output='screen'
    )

    return LaunchDescription([
        sweeper_node
    ])