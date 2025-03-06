import rclpy
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import time

from mapless_navigation_rl.rl_environment import TurtleBot3RLEnvironment
from mapless_navigation_rl.dqn_agent import DQNAgent

def main(args=None):
    # ROS2の初期化
    rclpy.init(args=args)
    
    # 環境とエージェントの作成
    env = TurtleBot3RLEnvironment()
    agent = DQNAgent(env.state_size, env.action_size)
    
    # 訓練パラメータ
    num_episodes = 1000
    
    # メトリクストラッキング
    scores = []
    epsilon_history = []
    success_history = []
    
    # モデル保存用ディレクトリの作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/dqn_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # 訓練ループ
    for e in range(num_episodes):
        # 新しいエピソードのために環境をリセット
        state = env.reset()
        
        # 現在のエピソードのメトリクス
        episode_reward = 0
        step = 0
        
        while True:
            # エージェントによる行動選択
            action = agent.act(state)
            
            # 環境内で行動を実行
            next_state, reward, done, info = env.step(action)
            
            # エージェントのメモリに経験を保存
            agent.memorize(state, action, reward, next_state, done)
            
            # 状態とメトリクスの更新
            state = next_state
            episode_reward += reward
            step += 1
            
            # エピソードが終了したかチェック
            if done:
                # 定期的にターゲットモデルを更新
                if e % agent.update_target_frequency == 0:
                    agent.update_target_model()
                
                # エピソード結果をログに記録
                success = info.get('success', False)
                success_history.append(1 if success else 0)
                
                env.get_logger().info(
                    f"エピソード: {e+1}/{num_episodes}, "
                    f"スコア: {episode_reward:.2f}, "
                    f"イプシロン: {agent.epsilon:.4f}, "
                    f"ステップ数: {step}, "
                    f"成功: {success}"
                )
                
                # メトリクスの保存
                scores.append(episode_reward)
                epsilon_history.append(agent.epsilon)
                
                # 定期的にモデルを保存
                if (e+1) % 100 == 0:
                    agent.save(f"{model_dir}/dqn_weights_episode_{e+1}.h5")
                    
                    # メトリクスのプロットと保存
                    _plot_metrics(scores, epsilon_history, success_history, model_dir, e+1)
                
                break
            
            # エージェントを訓練
            agent.replay()
        
        # ROS2コールバックのための時間を確保
        rclpy.spin_once(env, timeout_sec=0.1)
    
    # 最終保存
    agent.save(f"{model_dir}/dqn_weights_final.h5")
    _plot_metrics(scores, epsilon_history, success_history, model_dir, num_episodes)
    
    # クリーンアップ
    env.destroy_node()
    rclpy.shutdown()
    
def _plot_metrics(scores, epsilon_history, success_history, save_dir, episode):
    # すべてのメトリクスをプロット
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # 報酬のプロット
    axs[0].plot(scores)
    axs[0].set_title('エピソード報酬')
    axs[0].set_xlabel('エピソード')
    axs[0].set_ylabel('報酬')
    
    # イプシロンのプロット
    axs[1].plot(epsilon_history)
    axs[1].set_title('探索率 (イプシロン)')
    axs[1].set_xlabel('エピソード')
    axs[1].set_ylabel('イプシロン')
    
    # 成功率のプロット (移動平均)
    window_size = min(100, len(success_history))
    if window_size > 0:
        success_rate = np.convolve(success_history, 
                                  np.ones(window_size)/window_size, 
                                  mode='valid')
        axs[2].plot(success_rate)
        axs[2].set_title(f'成功率 (移動平均, ウィンドウ={window_size})')
        axs[2].set_xlabel('エピソード')
        axs[2].set_ylabel('成功率')
        axs[2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/metrics_episode_{episode}.png")
    plt.close()

if __name__ == '__main__':
    main()