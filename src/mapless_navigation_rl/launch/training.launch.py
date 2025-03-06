import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, ExecuteProcess, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # パッケージディレクトリの取得
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    pkg_mapless_navigation_rl = get_package_share_directory('mapless_navigation_rl')
    
    # 引数の定義
    random_seed_arg = DeclareLaunchArgument(
        'random_seed',
        default_value='42',
        description='強化学習用のランダムシード'
    )
    
    # 設定変数
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    random_seed = LaunchConfiguration('random_seed')
    
    # TurtleBot3 Gazebo起動ファイルを含める
    turtlebot3_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_turtlebot3_gazebo, 'launch', 'empty_world.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )
    
    # TURTLEBOT3_MODEL環境変数の設定
    os.environ['TURTLEBOT3_MODEL'] = 'burger'
    
    # 訓練ノード
    training_node = Node(
        package='mapless_navigation_rl',
        executable='train_agent',
        name='train_agent',
        parameters=[{'random_seed': random_seed}],
        output='screen'
    )
    
    return LaunchDescription([
        random_seed_arg,
        turtlebot3_gazebo,
        training_node
    ])