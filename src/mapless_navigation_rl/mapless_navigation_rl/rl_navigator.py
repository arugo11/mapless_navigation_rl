import rclpy
from rclpy.node import Node
import numpy as np
import os
import math
import time
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from visualization_msgs.msg import Marker  # マーカー用に追加
import tf2_ros

from mapless_navigation_rl.dqn_agent import DQNAgent

class RLNavigator(Node):
    def __init__(self):
        super().__init__('rl_navigator')
        
        # パラメータの宣言
        self.declare_parameter('model_path', 'models/dqn_latest/dqn_weights_final.h5')
        self.declare_parameter('goal_x', 2.0)
        self.declare_parameter('goal_y', 2.0)
        
        # パラメータの取得
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        goal_x = self.get_parameter('goal_x').get_parameter_value().double_value
        goal_y = self.get_parameter('goal_y').get_parameter_value().double_value
        
        # パブリッシャーとサブスクライバー
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.status_pub = self.create_publisher(String, '/rl_status', 10)
        # 目標マーカー用のパブリッシャーを追加
        self.goal_marker_pub = self.create_publisher(Marker, '/goal_marker', 10)
        
        # 変数の初期化
        self.scan_ranges = []
        self.min_obstacle_distance = 999.0
        self.robot_position = Point()
        self.robot_orientation = 0.0
        self.goal_position = Point(x=goal_x, y=goal_y, z=0.0)
        
        # 強化学習パラメータ
        self.num_laser_samples = 24
        self.scan_processed = np.zeros(self.num_laser_samples)
        self.state_size = self.num_laser_samples + 2  # LiDARサンプル + 目標距離と角度
        self.action_size = 5  # 左大きく, 左小さく, 直進, 右小さく, 右大きく
        
        # 定数
        self.goal_reached_threshold = 0.35  # メートル
        self.collision_threshold = 0.25  # メートル
        
        # エージェントの初期化
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.load_agent()
        
        # 目標マーカーの発行
        self.publish_goal_marker()
        
        # 行動実行用タイマー
        self.timer = self.create_timer(0.1, self.execute_action)
        # マーカー更新用タイマー
        self.marker_timer = self.create_timer(1.0, self.publish_goal_marker)
        
        self.get_logger().info('RLナビゲーターノード初期化完了!')
        self.get_logger().info(f'目標位置: x={self.goal_position.x}, y={self.goal_position.y}')
    
    def publish_goal_marker(self):
        # 目標位置のマーカーを作成
        marker = Marker()
        marker.header.frame_id = "odom"  # オドメトリフレームを基準
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "navigation_goals"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # マーカーの位置
        marker.pose.position.x = self.goal_position.x
        marker.pose.position.y = self.goal_position.y
        marker.pose.position.z = 0.1  # 地面よりわずかに上に表示
        marker.pose.orientation.w = 1.0
        
        # マーカーのサイズ
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        # マーカーの色 (赤)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        # マーカーの寿命 (永続)
        marker.lifetime.sec = 0
        
        # マーカーを発行
        self.goal_marker_pub.publish(marker)
    
    def load_agent(self):
        # 訓練済みモデルのロード
        if os.path.exists(self.model_path):
            self.agent.load(self.model_path)
            self.get_logger().info(f'モデルを {self.model_path} から読み込みました')
        else:
            self.get_logger().error(f'{self.model_path} にモデルが見つかりませんでした')
    
    def scan_callback(self, msg):
        # LaserScanメッセージの処理
        self.scan_ranges = np.array(msg.ranges)
        self.min_obstacle_distance = min(min(self.scan_ranges), 3.5)  # 3.5mで上限設定
        
        # LiDARスキャンをサンプリングして状態に使用
        angle_increment = len(self.scan_ranges) / self.num_laser_samples
        
        self.scan_processed = np.array([
            min(self.scan_ranges[int(i * angle_increment)], 3.5) 
            for i in range(self.num_laser_samples)
        ])
    
    def odom_callback(self, msg):
        # ロボットの位置と方向を取得
        self.robot_position = msg.pose.pose.position
        
        # クォータニオンからオイラー角に変換 (ヨー角のみ使用)
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.robot_orientation = math.atan2(siny_cosp, cosy_cosp)  # ヨー角
    
    def get_state(self):
        # 目標までの距離と角度を計算
        dx = self.goal_position.x - self.robot_position.x
        dy = self.goal_position.y - self.robot_position.y
        distance_to_goal = math.sqrt(dx*dx + dy*dy)
        
        # 目標に対する角度をロボットの向きに対して相対的に計算
        goal_angle = math.atan2(dy, dx)
        heading_error = goal_angle - self.robot_orientation
        
        # 角度を[-pi, pi]の範囲に正規化
        if heading_error > math.pi:
            heading_error -= 2 * math.pi
        elif heading_error < -math.pi:
            heading_error += 2 * math.pi
            
        # 状態ベクトルを構築
        state = np.concatenate([
            self.scan_processed,
            [distance_to_goal / 10.0],  # 距離を正規化
            [heading_error / math.pi]   # 角度を正規化
        ])
        
        return state
    
    def execute_action(self):
        # 現在の状態を取得
        state = self.get_state()
        
        # エージェントに行動を決定させる（訓練中ではない）
        action = self.agent.act(state, training=False)
        
        # 行動を実行
        self.set_action(action)
        
        # 目標到達や衝突をチェック
        self.check_navigation_status()
    
    def set_action(self, action):
        # 行動インデックスをロボットコマンドに変換
        twist = Twist()
        
        # 前進速度は固定
        twist.linear.x = 0.15
        
        # 行動に基づく角速度
        if action == 0:   # 左に大きく回転
            twist.angular.z = 1.5
        elif action == 1: # 左に小さく回転
            twist.angular.z = 0.75
        elif action == 2: # 直進
            twist.angular.z = 0.0
        elif action == 3: # 右に小さく回転
            twist.angular.z = -0.75
        elif action == 4: # 右に大きく回転
            twist.angular.z = -1.5
        
        # 速度指令を発行
        self.cmd_vel_pub.publish(twist)
    
    def check_navigation_status(self):
        # 目標までの距離を計算
        dx = self.goal_position.x - self.robot_position.x
        dy = self.goal_position.y - self.robot_position.y
        distance_to_goal = math.sqrt(dx*dx + dy*dy)
        
        # ステータスの発行
        status_msg = String()
        
        # 目標に到達したかチェック
        if distance_to_goal < self.goal_reached_threshold:
            status_msg.data = "目標に到達しました！"
            self.get_logger().info('目標に到達しました!')
            
            # ロボットを停止
            self.stop_robot()
            
        # 衝突したかチェック
        elif self.min_obstacle_distance < self.collision_threshold:
            status_msg.data = "衝突を検知しました！"
            self.get_logger().info('衝突を検知しました!')
            
            # ロボットを停止
            self.stop_robot()
        
        else:
            status_msg.data = f"ナビゲーション中: 距離={distance_to_goal:.2f}m, 最小障害物距離={self.min_obstacle_distance:.2f}m"
        
        self.status_pub.publish(status_msg)
    
    def stop_robot(self):
        # ロボットを停止させるためにゼロ速度を発行
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = RLNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()