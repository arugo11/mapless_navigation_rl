#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterValue
from rcl_interfaces.srv import SetParameters
from std_srvs.srv import Empty
from rclpy.parameter import Parameter
import sys

class SimSpeedController(Node):
    def __init__(self):
        super().__init__('sim_speed_controller')
        
        # パラメータの宣言
        self.declare_parameter('speed_factor', 1.0)
        
        # Gazeboパラメータ設定用クライアント
        self.set_param_client = self.create_client(
            SetParameters, 
            '/gazebo/set_parameters'
        )
        
        # 速度変更用サービス
        self.set_speed_srv = self.create_service(
            Empty, 
            '~/set_speed', 
            self.set_speed_callback
        )
        
        self.get_logger().info('シミュレーション速度コントローラが起動しました')
        self.get_logger().info('現在の速度倍率: 1.0倍')
    
    def set_speed_callback(self, request, response):
        # 現在のパラメータ値を取得
        speed_factor = self.get_parameter('speed_factor').get_parameter_value().double_value
        
        # Gazeboのリアルタイムファクターを設定
        self.set_gazebo_real_time_factor(speed_factor)
        
        self.get_logger().info(f'シミュレーション速度を{speed_factor}倍に設定しました')
        return response
    
    def set_gazebo_real_time_factor(self, factor):
        # Gazeboサービスの準備を待つ
        while not self.set_param_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Gazeboサービスを待っています...')
        
        # パラメータ値の設定
        param_value = ParameterValue()
        param_value.type = Parameter.Type.DOUBLE.value
        param_value.double_value = factor
        
        # リクエストの作成
        request = SetParameters.Request()
        param = Parameter()
        param.name = 'physics.real_time_factor'
        param.value = param_value
        request.parameters = [param]
        
        # サービスコール
        future = self.set_param_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        return future.result()

def main(args=None):
    rclpy.init(args=args)
    
    # コマンドライン引数から速度倍率を取得
    speed_factor = 1.0
    if len(sys.argv) > 1:
        try:
            speed_factor = float(sys.argv[1])
        except ValueError:
            print("エラー: 有効な浮動小数点数を指定してください")
            sys.exit(1)
    
    controller = SimSpeedController()
    
    # 引数が提供された場合、速度を設定
    if len(sys.argv) > 1:
        controller.get_parameter('speed_factor').value = speed_factor
        controller.set_gazebo_real_time_factor(speed_factor)
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        # 終了前に速度を通常に戻す
        controller.set_gazebo_real_time_factor(1.0)
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()