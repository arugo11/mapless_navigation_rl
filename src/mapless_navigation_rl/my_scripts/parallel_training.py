def run_training_instance(instance_id, num_episodes, log_dir, random_seed=42):
    """指定されたIDでトレーニングインスタンスを実行"""
    
    # ROS_DOMAIN_IDを設定して各プロセスが独立したROSネットワークを持つようにする
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(100 + instance_id)
    
    # Gazeboワールドの設定を変更（各インスタンスで異なるワールドを使用）
    world_x = 2.0 + instance_id * 0.1
    world_y = 2.0 + instance_id * 0.1
    
    # トレーニング用ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/parallel_{instance_id}_{timestamp}"
    
    # ros2 launch コマンドを構築
    cmd = [
        "ros2", "launch", "mapless_navigation_rl", "training.launch.py",
        f"random_seed:={random_seed}",
        f"world_x:={world_x}",
        f"world_y:={world_y}",
        f"instance_id:={instance_id}"
    ]
    
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