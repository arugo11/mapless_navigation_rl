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
        
        # 目標に近づくことに対する報酬 - 無限大になる可能性があるため修正
        if distance_to_goal < self.min_distance_to_goal:
            # 差分に上限を設ける
            distance_diff = min(self.min_distance_to_goal - distance_to_goal, 1.0)
            reward += self.approach_reward * distance_diff
            self.min_distance_to_goal = distance_to_goal
        
        # 障害物に近すぎる場合の小さなペナルティ
        if self.min_obstacle_distance < 0.5:
            reward -= (0.5 - self.min_obstacle_distance) * 2.0
    
    info['distance'] = distance_to_goal
    info['step'] = self.episode_step
    
    return reward, done, info
