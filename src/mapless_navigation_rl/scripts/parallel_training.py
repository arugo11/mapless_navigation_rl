#!/usr/bin/env python3

import subprocess
import multiprocessing
import os
import argparse
import time
import signal
import sys
from datetime import datetime

# 終了時の処理
running_processes = []

def signal_handler(sig, frame):
    print('Ctrl+Cが押されました。すべてのプロセスを終了します。')
    for p in running_processes:
        if p.poll() is None:  # プロセスがまだ実行中かチェック
            p.terminate()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def run_training_instance(instance_id, num_episodes, log_dir, custom_args=None):
    """指定されたIDでトレーニングインスタンスを実行"""
    
    # ROS_DOMAIN_IDを設定して各プロセスが独立したROSネットワークを持つようにする
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(100 + instance_id)
    
    # トレーニング用ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/parallel_{instance_id}_{timestamp}"
    
    # ros2 launch コマンドを構築
    cmd = [
        "ros2", "launch", "mapless_navigation_rl", "training.launch.py"
    ]
    
    # カスタム引数があれば追加
    if custom_args:
        for arg in custom_args:
            cmd.append(arg)
    
    # プロセスを開始
    log_file = open(f"{log_dir}/training_instance_{instance_id}.log", "w")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    running_processes.append(process)
    print(f"インスタンス {instance_id} を起動しました (PID: {process.pid})")
    
    # プロセスの完了を待つ
    process.wait()
    log_file.close()
    
    print(f"インスタンス {instance_id} が終了しました")
    return instance_id

def main():
    parser = argparse.ArgumentParser(description='並列トレーニングの実行')
    parser.add_argument('--num_instances', type=int, default=4, 
                      help='並列に実行するトレーニングインスタンスの数')
    parser.add_argument('--num_episodes', type=int, default=1000, 
                      help='各インスタンスで実行するエピソード数')
    args = parser.parse_args()
    
    # ログディレクトリの作成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/parallel_training_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"並列トレーニングを開始します (インスタンス数: {args.num_instances})")
    
    # ThreadPoolを使用して並列に実行
    with multiprocessing.Pool(processes=args.num_instances) as pool:
        results = []
        for i in range(args.num_instances):
            # それぞれのインスタンスに異なるシード値を設定
            custom_args = ["--ros-args", "-p", f"random_seed:={i*100}"]
            result = pool.apply_async(
                run_training_instance,
                args=(i, args.num_episodes, log_dir, custom_args)
            )
            results.append(result)
            
            # 競合を防ぐため、次のインスタンス起動前に少し待機
            time.sleep(5)
        
        # すべての結果を待つ
        for result in results:
            instance_id = result.get()
            print(f"インスタンス {instance_id} の処理が完了しました")
    
    print("すべてのトレーニングインスタンスが完了しました")

if __name__ == "__main__":
    main()