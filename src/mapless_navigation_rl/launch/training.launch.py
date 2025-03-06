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
    
    # インスタンスIDの追加
    instance_id_arg = DeclareLaunchArgument(
        'instance_id',
        default_value='0',
        description='トレーニングインスタンスのID'
    )
    
    # ワールド座標の引数を追加
    world_x_arg = DeclareLaunchArgument(
        'world_x',
        default_value='0.0',
        description='Gazeboワールドのx座標'
    )
    
    world_y_arg = DeclareLaunchArgument(
        'world_y',
        default_value='0.0',
        description='Gazeboワールドのy座標'
    )
    
    # 設定変数
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    random_seed = LaunchConfiguration('random_seed')
    world_x = LaunchConfiguration('world_x')
    world_y = LaunchConfiguration('world_y')
    instance_id = LaunchConfiguration('instance_id')
    
    # TURTLEBOT3_MODEL環境変数の設定
    os.environ['TURTLEBOT3_MODEL'] = 'burger'
    
    # TurtleBot3 Gazebo起動部分を変更
    # カスタムワールドファイルのパスを作成
    world_file = os.path.join(pkg_mapless_navigation_rl, 'worlds', 'colored_floor.world')

    # TurtleBot3 Gazebo起動ファイルを含める
    turtlebot3_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_turtlebot3_gazebo, 'launch', 'empty_world.launch.py')
        ),
        launch_arguments={
            'use_sim_time': use_sim_time,
            'world': world_file
        }.items()
    )
    
    # 訓練ノード
    training_node = Node(
        package='mapless_navigation_rl',
        executable='train_agent',
        name='train_agent',
        parameters=[
            {'random_seed': random_seed},
            {'world_x': world_x},
            {'world_y': world_y},
            {'instance_id': instance_id}
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
        random_seed_arg,
        world_x_arg,
        world_y_arg,
        instance_id_arg,
        sim_speed_arg,
        turtlebot3_gazebo,
        training_node,
        speed_controller_node
    ])