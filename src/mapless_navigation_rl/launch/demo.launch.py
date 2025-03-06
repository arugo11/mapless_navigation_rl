import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # パッケージディレクトリの取得
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    pkg_mapless_navigation_rl = get_package_share_directory('mapless_navigation_rl')
    
    # 引数の宣言
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='models/dqn_latest/dqn_weights_final.h5',
        description='訓練済みモデルファイルのパス'
    )
    
    goal_x_arg = DeclareLaunchArgument(
        'goal_x',
        default_value='2.0',
        description='目標のX座標'
    )
    
    goal_y_arg = DeclareLaunchArgument(
        'goal_y',
        default_value='2.0',
        description='目標のY座標'
    )
    
    # 設定変数
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    model_path = LaunchConfiguration('model_path')
    goal_x = LaunchConfiguration('goal_x')
    goal_y = LaunchConfiguration('goal_y')
    
    # TURTLEBOT3_MODEL環境変数の設定
    os.environ['TURTLEBOT3_MODEL'] = 'burger'
    
    # TurtleBot3 Gazebo起動ファイルを含める
    turtlebot3_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_turtlebot3_gazebo, 'launch', 'empty_world.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time
        }.items()
    )
    
    # RLナビゲーターノード
    navigator_node = Node(
        package='mapless_navigation_rl',
        executable='rl_navigator',
        name='rl_navigator',
        parameters=[
            {'model_path': model_path},
            {'goal_x': goal_x},
            {'goal_y': goal_y}
        ],
        output='screen'
    )
    
    # 速度倍率の引数を追加
    sim_speed_arg = DeclareLaunchArgument(
        'sim_speed',
        default_value='1.0',
        description='シミュレーション速度の倍率'
    )

    # 他の変数の宣言と同様に、sim_speedを追加
    sim_speed = LaunchConfiguration('sim_speed')

    # シミュレーション速度コントローラーノード
    speed_controller_node = Node(
        package='mapless_navigation_rl',
        executable='sim_speed_controller',
        name='sim_speed_controller',
        parameters=[
            {'speed_factor': sim_speed}
        ],
        output='screen'
    )

    # LaunchDescriptionに追加
    return LaunchDescription([
        model_path_arg,
        goal_x_arg,
        goal_y_arg,
        sim_speed_arg,  # 追加
        turtlebot3_gazebo,
        navigator_node,
        speed_controller_node  # 追加
    ])