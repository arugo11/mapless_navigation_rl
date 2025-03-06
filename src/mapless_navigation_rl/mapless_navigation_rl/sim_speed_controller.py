#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter as RclpyParameter  # ノード内パラメータ更新用
from rcl_interfaces.msg import Parameter as InterfaceParameter, ParameterValue
from rcl_interfaces.srv import SetParameters  # srvからインポート
from std_srvs.srv import Empty

class SimSpeedController(Node):
    def __init__(self):
        super().__init__('sim_speed_controller')
        
        # ROS2パラメータ "sim_speed" を宣言（launch等から渡す場合はこの値が上書きされる）
        self.declare_parameter('sim_speed', 1.0)
        
        # Gazeboの物理プロパティ更新用サービスクライアント
        self.set_param_client = self.create_client(
            SetParameters, 
            '/gazebo/set_parameters'
        )
        
        # 速度変更用サービス（手動呼び出し用）
        self.set_speed_srv = self.create_service(
            Empty, 
            '~/set_speed', 
            self.set_speed_callback
        )
        
        current_speed = self.get_parameter('sim_speed').get_parameter_value().double_value
        self.get_logger().info('シミュレーション速度コントローラが起動しました')
        self.get_logger().info(f'現在の速度倍率: {current_speed}倍')
    
    def set_speed_callback(self, request, response):
        # ROS2パラメータから sim_speed の値を取得
        sim_speed = self.get_parameter('sim_speed').get_parameter_value().double_value
        
        # Gazeboの物理プロパティを更新
        self.set_gazebo_physics(sim_speed)
        
        self.get_logger().info(f'シミュレーション速度を {sim_speed} 倍に設定しました')
        return response
    
    def set_gazebo_physics(self, factor):
        # サービス利用可能を待つ
        while not self.set_param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Gazeboサービスを待っています...')
        
        # 環境依存ですが、ここでは以下のデフォルト値を仮定
        default_max_step_size = 0.001      # [秒]
        default_real_time_update_rate = 1000.0  # [Hz]
        # デフォルトの積は 0.001 * 1000 = 1 (1倍)
        # RTF を factor 倍にするため、real_time_update_rate を factor 倍に変更
        new_max_step_size = default_max_step_size
        new_real_time_update_rate = default_real_time_update_rate * factor
        
        self.get_logger().info(f'新しい物理パラメータ -> max_step_size: {new_max_step_size}, real_time_update_rate: {new_real_time_update_rate}')
        
        # ParameterValue の設定（double 型）
        param_value_step = ParameterValue()
        param_value_step.type = RclpyParameter.Type.DOUBLE.value
        param_value_step.double_value = new_max_step_size
        
        param_value_rate = ParameterValue()
        param_value_rate.type = RclpyParameter.Type.DOUBLE.value
        param_value_rate.double_value = new_real_time_update_rate
        
        # サービスリクエストの作成（2つのパラメータを一括で設定）
        request = SetParameters.Request()
        param_step = InterfaceParameter(name='physics.max_step_size', value=param_value_step)
        param_rate = InterfaceParameter(name='physics.real_time_update_rate', value=param_value_rate)
        request.parameters = [param_step, param_rate]
        
        future = self.set_param_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is not None:
            self.get_logger().info('Gazeboの物理プロパティが更新されました')
        else:
            self.get_logger().error('Gazeboの物理プロパティ更新に失敗しました')
        return result

def main(args=None):
    rclpy.init(args=args)
    
    # launchファイルやパラメータファイルから渡された "sim_speed" を使用するため、sys.argv の解析は行わない
    controller = SimSpeedController()
    
    # ROS2パラメータ "sim_speed" から値を取得
    sim_speed = controller.get_parameter('sim_speed').get_parameter_value().double_value
    controller.set_gazebo_physics(sim_speed)
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        # 終了前に物理プロパティをデフォルト（1.0倍、real_time_update_rate=1000）に戻す
        controller.set_gazebo_physics(1.0)
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
