import rclpy
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
import csv

from mapless_navigation_rl.rl_environment import TurtleBot3RLEnvironment
from mapless_navigation_rl.dqn_agent import DQNAgent

def main(args=None):
    # ROS2の初期化
    rclpy.init(args=args)
    
    # ノードの作成
    node = rclpy.create_node('rl_evaluator')
    
    # パラメータの宣言
    node.declare_parameter('model_path', 'models/dqn_latest/dqn_weights_final.h5')
    node.declare_parameter('num_trials', 100)
    
    # パラメータの取得
    model_path = node.get_parameter('model_path').get_parameter_value().string_value
    num_trials = node.get_parameter('num_trials').get_parameter_value().integer_value
    
    # 環境とエージェントの作成
    env = TurtleBot3RLEnvironment()
    agent = DQNAgent(env.state_size, env.action_size)
    
    # 訓練済みモデルのロード
    if os.path.exists(model_path):
        agent.load(model_path)
        node.get_logger().info(f'モデルを {model_path} から読み込みました')
    else:
        node.get_logger().error(f'{model_path} にモデルが見つかりませんでした')
        return
    
    # 評価結果保存用ディレクトリの作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = f"results/evaluation_{timestamp}"
    os.makedirs(eval_dir, exist_ok=True)
    
    # 評価用メトリクス
    success_count = 0
    collision_count = 0
    timeout_count = 0
    step_counts = []
    path_lengths = []
    
    # 結果用CSVファイルの作成
    with open(f"{eval_dir}/evaluation_results.csv", 'w', newline='') as csvfile:
        fieldnames = ['trial', 'result', 'steps', 'path_length', 'final_distance']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # 評価ループ
        for trial in range(num_trials):
            # 環境のリセット
            state = env.reset()
            
            # 現在のトライアルのメトリクス
            path_length = 0.0
            last_position = (env.robot_position.x, env.robot_position.y)
            trial_result = 'unknown'
            
            # エピソードの実行
            done = False
            step = 0
            while not done:
                # エージェントによる行動選択（探索なし）
                action = agent.act(state, training=False)
                
                # 環境内で行動を実行
                next_state, reward, done, info = env.step(action)
                
                # 経路長の更新
                current_position = (env.robot_position.x, env.robot_position.y)
                path_length += np.sqrt((current_position[0] - last_position[0])**2 + 
                                      (current_position[1] - last_position[1])**2)
                last_position = current_position
                
                # 状態の更新
                state = next_state
                step += 1
                
                # エピソードが終了したかチェック
                if done:
                    if info.get('success', False):
                        success_count += 1
                        trial_result = 'success'
                    elif info.get('collision', False):
                        collision_count += 1
                        trial_result = 'collision'
                    elif info.get('timeout', False):
                        timeout_count += 1
                        trial_result = 'timeout'
                    
                    # トライアルメトリクスの保存
                    step_counts.append(step)
                    path_lengths.append(path_length)
                    
                    # トライアル結果のログ記録
                    node.get_logger().info(
                        f"トライアル {trial+1}/{num_trials}: "
                        f"結果 = {trial_result}, "
                        f"ステップ数 = {step}, "
                        f"経路長 = {path_length:.2f}m, "
                        f"最終距離 = {info['distance']:.2f}m"
                    )
                    
                    # CSVに書き込み
                    writer.writerow({
                        'trial': trial + 1,
                        'result': trial_result,
                        'steps': step,
                        'path_length': round(path_length, 2),
                        'final_distance': round(info['distance'], 2)
                    })
            
            # ROS2コールバックのための時間を確保
            rclpy.spin_once(env, timeout_sec=0.1)
    
    # 評価メトリクスの計算
    success_rate = success_count / num_trials
    avg_steps = np.mean(step_counts) if step_counts else 0
    avg_path_length = np.mean(path_lengths) if path_lengths else 0
    
    # サマリー結果の表示
    node.get_logger().info(f"\n評価サマリー ({num_trials} トライアル):")
    node.get_logger().info(f"成功率: {success_rate:.2f} ({success_count}/{num_trials})")
    node.get_logger().info(f"衝突率: {collision_count/num_trials:.2f} ({collision_count}/{num_trials})")
    node.get_logger().info(f"タイムアウト率: {timeout_count/num_trials:.2f} ({timeout_count}/{num_trials})")
    node.get_logger().info(f"平均ステップ数: {avg_steps:.2f}")
    node.get_logger().info(f"平均経路長: {avg_path_length:.2f}m")
    
    # サマリーをファイルに保存
    with open(f"{eval_dir}/summary.txt", 'w') as f:
        f.write(f"評価サマリー ({num_trials} トライアル):\n")
        f.write(f"成功率: {success_rate:.2f} ({success_count}/{num_trials})\n")
        f.write(f"衝突率: {collision_count/num_trials:.2f} ({collision_count}/{num_trials})\n")
        f.write(f"タイムアウト率: {timeout_count/num_trials:.2f} ({timeout_count}/{num_trials})\n")
        f.write(f"平均ステップ数: {avg_steps:.2f}\n")
        f.write(f"平均経路長: {avg_path_length:.2f}m\n")
    
    # 結果のプロットと保存
    _plot_evaluation_results(step_counts, path_lengths, success_rate, eval_dir)
    
    # クリーンアップ
    env.destroy_node()
    node.destroy_node()
    rclpy.shutdown()

def _plot_evaluation_results(steps, path_lengths, success_rate, save_dir):
    # サブプロット付きの図を作成
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    
    # ステップ数分布のプロット
    axs[0].hist(steps, bins=20)
    axs[0].set_title('ステップ数の分布')
    axs[0].set_xlabel('ステップ数')
    axs[0].set_ylabel('カウント')
    axs[0].axvline(np.mean(steps), color='r', linestyle='dashed', linewidth=2)
    axs[0].text(np.mean(steps)*1.1, axs[0].get_ylim()[1]*0.9, f'平均: {np.mean(steps):.2f}')
    
    # 経路長分布のプロット
    axs[1].hist(path_lengths, bins=20)
    axs[1].set_title('経路長の分布')
    axs[1].set_xlabel('経路長 (m)')
    axs[1].set_ylabel('カウント')
    axs[1].axvline(np.mean(path_lengths), color='r', linestyle='dashed', linewidth=2)
    axs[1].text(np.mean(path_lengths)*1.1, axs[1].get_ylim()[1]*0.9, f'平均: {np.mean(path_lengths):.2f}')
    
    plt.suptitle(f'評価結果 (成功率: {success_rate:.2f})')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/evaluation_results.png")
    plt.close()

if __name__ == '__main__':
    main()