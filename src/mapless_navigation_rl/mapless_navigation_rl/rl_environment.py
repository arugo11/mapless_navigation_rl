import rclpy
from rclpy.node import Node
import numpy as np
import time
import math
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import random
from visualization_msgs.msg import Marker

class TurtleBot3RLEnvironment(Node):
    def __init__(self):
        super().__init__('turtlebot3_rl_environment')
        
        # パブリッシャーとサブスクライバーの設定
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # 目標マーカーのパブリッシャーを追加
        self.goal_marker_pub = self.create_publisher(Marker, '/goal_marker', 10)
        
        # シミュレーションリセット用サービスクライアント
        self.reset_simulation_client = self.create_client(Empty, '/reset_simulation')
        
        # クラス変数の初期化
        self.scan_ranges = []
        self.min_obstacle_distance = 999.0
        self.robot_position = Point()
        self.robot_orientation = 0.0
        self.goal_position = Point()
        
        # 強化学習パラメータ
        self.num_laser_samples = 24  # サンプリングするLiDARビームの数
        self.scan_processed = np.zeros(self.num_laser_samples)
        self.state_size = self.num_laser_samples + 2  # LiDARサンプル + 目標距離と角度
        self.action_size = 5  # 左大きく, 左小さく, 直進, 右小さく, 右大きく
        
        # エピソードパラメータ
        self.episode_step = 0
        self.max_episode_steps = 500
        self.goal_reached_threshold = 0.35  # メートル
        self.collision_threshold = 0.25  # メートル
        
        # 報酬計算用の定数
        self.collision_penalty = -100.0
        self.goal_reward = 100.0
        self.step_penalty = -0.1
        self.approach_reward = 0.5
        self.min_distance_to_goal = float('inf')
        
        # マーカー更新タイマーを追加
        self.marker_timer = self.create_timer(1.0, self.publish_goal_marker)
        
        # すべてのサブスクライバーとパブリッシャーの準備を待つ
        self.get_logger().info('ROSトピックの準備中...')
        # トピックの準備を待つ
        self.get_logger().info('ROSトピックの準備中...')
        while not self.cmd_vel_pub.get_subscription_count() > 0 or \
              not self.scan_sub.get_publisher_count() > 0 or \
              not self.odom_sub.get_publisher_count() > 0:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info('環境の初期化完了!')
        self.get_logger().info('環境の初期化完了!')
        
    def scan_callback(self, msg):
        # LaserScanメッセージの処理
        if len(msg.ranges) == 0:
            self.get_logger().warn('空のLaserScanメッセージを受信しました')
            return
            
        self.scan_ranges = np.array(msg.ranges)
        
        # 無効な値（NaN、Infなど）を処理
        self.scan_ranges = np.nan_to_num(self.scan_ranges, nan=3.5, posinf=3.5, neginf=0.1)
        
        self.min_obstacle_distance = min(np.min(self.scan_ranges), 3.5)  # 3.5mで上限設定
        
        # LiDARスキャンをサンプリングして状態に使用
        angle_increment = len(self.scan_ranges) / self.num_laser_samples
        
        self.scan_processed = np.array([
            min(self.scan_ranges[int(i * angle_increment) % len(self.scan_ranges)], 3.5) 
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
        
    def reset(self):
        # 環境のリセット
        self.episode_step = 0
        self.min_distance_to_goal = float('inf')
        
        # シミュレーションのリセット
        while not self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('リセットサービスを待機中...')
        
        # リセットリクエストを非同期に送信
        future = self.reset_simulation_client.call_async(Empty.Request())
        
        # フューチャーが完了するまで待機
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        # リセット後に少し待機して安定させる
        time.sleep(1.0)
        
        # ランダムな目標位置を設定 (妥当な範囲内)
        self.goal_position.x = random.uniform(1.0, 3.0)
        self.goal_position.y = random.uniform(-1.5, 1.5)
        self.goal_position.z = 0.0
        
        self.get_logger().info(f'新しい目標設定: x={self.goal_position.x:.2f}, y={self.goal_position.y:.2f}')
        
        # 目標マーカーを更新
        self.publish_goal_marker()
        
        # センサーデータが更新されるまで待機
        self._wait_for_sensor_data()
        
        # 初期状態を取得
        state = self._get_state()
        return state

    def _wait_for_sensor_data(self):
        """センサーデータが更新されるまで待機する"""
        max_attempts = 10
        for i in range(max_attempts):
            # コールバックを処理
            rclpy.spin_once(self, timeout_sec=0.1)
            
            # LiDARデータが更新されているか確認
            if len(self.scan_ranges) > 0:
                return True
            
            time.sleep(0.1)
        
        self.get_logger().warn('センサーデータの更新を待機中にタイムアウトしました')
        return False
    def step(self, action):
        # 行動を実行して待機
        self._set_action(action)
        self.episode_step += 1
        time.sleep(0.1)  # 行動実行の時間をシミュレート
        
        # 現在の状態を取得し報酬を計算
        state = self._get_state()
        reward, done, info = self._compute_reward()
        
        return state, reward, done, info
    
    def _get_state(self):
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
    
    def _set_action(self, action):
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
    
    def _compute_reward(self):
        # デフォルト値
        reward = 0
        done = False
        info = {}
        
        # 目標までの距離を計算
        dx = self.goal_position.x - self.robot_position.x
        dy = self.goal_position.y - self.robot_position.y
        distance_to_goal = math.sqrt(dx*dx + dy*dy)
        
        # 目標に到達したかチェック
        if distance_to_goal < self.goal_reached_threshold:
            reward = self.goal_reward
            done = True
            info['success'] = True
            self.get_logger().info('目標に到達しました!')
        
        # 衝突したかチェック
        elif self.min_obstacle_distance < self.collision_threshold:
            reward = self.collision_penalty
            done = True
            info['collision'] = True
            self.get_logger().info('衝突を検知しました!')
        
        # タイムアウトをチェック
        elif self.episode_step >= self.max_episode_steps:
            done = True
            info['timeout'] = True
            self.get_logger().info('エピソードがタイムアウトしました!')
        
        # 目標に近づくための報酬
        else:
            # 各ステップに小さなペナルティを与え、より速い解決を促す
            reward += self.step_penalty
            
            # 目標に近づくことに対する報酬 - より安定した方法
            if hasattr(self, 'previous_distance') and self.previous_distance > 0:
                # 前回の距離との差分で報酬を計算（前進するほど報酬が大きい）
                distance_diff = self.previous_distance - distance_to_goal
                reward += self.approach_reward * distance_diff
            
            # 現在の距離を記録
            self.previous_distance = distance_to_goal
            
            # 最も近い距離を更新
            if distance_to_goal < self.min_distance_to_goal:
                self.min_distance_to_goal = distance_to_goal
            
            # 障害物に近すぎる場合の小さなペナルティ
            if self.min_obstacle_distance < 0.5:
                reward -= (0.5 - self.min_obstacle_distance) * 2.0
        
        info['distance'] = distance_to_goal
        info['step'] = self.episode_step
        
        return reward, done, info

    def publish_goal_marker(self):
        """目標位置のマーカーをRvizに表示する"""
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal_markers"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = self.goal_position.x
        marker.pose.position.y = self.goal_position.y
        marker.pose.position.z = 0.05  # 床より少し上に表示
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.4  # サイズ設定
        marker.scale.y = 0.4
        marker.scale.z = 0.02
        
        marker.color.r = 1.0  # 赤色
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8  # やや透明
        
        self.goal_marker_pub.publish(marker)