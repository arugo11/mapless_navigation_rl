# TurtleBot3向け強化学習ベースのマップレスナビゲーション

このパッケージは、TurtleBot3ロボットのための強化学習（特にDQN: Deep Q-Network）を用いたマップレスナビゲーションを実装したものです。マップなしで障害物を回避しながら目標地点に到達するための自律的なナビゲーション能力を学習します。
## 注意
以下のREADMEはclaudeのGitHub連携機能を用いてclaudeのサービスを用いて記述したものです。
各コマンドの動作確認を行い、ハルシネーションなどがないことは確認済みです。
## 機能

- Deep Q-Networkを使用した強化学習ナビゲーション
- LiDARセンサーを使用した障害物検知と回避
- 事前マップなしでの目標地点へのナビゲーション
- 訓練済みモデルの評価
- 実環境での推論（ナビゲーション実行）
- Gazeboシミュレーション環境との連携

## 必要条件

- ROS 2 (Humble以上推奨)
- Gazebo
- TurtleBot3パッケージ
- Python 3.8以上
- TensorFlow 2.x
- NumPy, Matplotlib

## インストール方法

1. ROS 2ワークスペースのsrcディレクトリに移動します：

```bash
cd ~/ros2_ws/src/
```

2. このリポジトリをクローンします：

```bash
git clone https://github.com/username/mapless_navigation_rl.git
```

3. 依存パッケージをインストールします：

```bash
pip install tensorflow numpy matplotlib
```

4. ワークスペースをビルドします：

```bash
cd ~/ros2_ws
colcon build --packages-select mapless_navigation_rl
```

5. 環境変数を設定します：

```bash
source ~/ros2_ws/install/setup.bash
```

## 使用方法

### シミュレーション環境の起動

TurtleBot3のGazeboシミュレーション環境を起動します：

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo empty_world.launch.py
```

### エージェントの訓練

新しいDQNエージェントを訓練するには：

```bash
ros2 run mapless_navigation_rl train_agent
```

訓練は以下のパラメータをサポートしています：
- `random_seed`: 訓練の再現性のためのシード値（デフォルト: 42）

### エージェントの評価

訓練済みのモデルを評価するには：

```bash
ros2 run mapless_navigation_rl evaluate_agent --ros-args -p model_path:=models/dqn_latest/dqn_weights_final.h5 -p num_trials:=100
```

パラメータ：
- `model_path`: 評価するモデルのパス
- `num_trials`: 評価するエピソードの数

### ナビゲーションの実行

訓練済みのモデルを使用してナビゲーションを実行するには：

```bash
ros2 run mapless_navigation_rl rl_navigator --ros-args -p model_path:=models/dqn_latest/dqn_weights_final.h5 -p goal_x:=2.0 -p goal_y:=2.0
```

パラメータ：
- `model_path`: 使用するモデルのパス
- `goal_x`: 目標地点のX座標
- `goal_y`: 目標地点のY座標

## ファイル構成

- `dqn_agent.py`: DQNアルゴリズムの実装
- `rl_environment.py`: 強化学習環境の定義
- `train_agent.py`: エージェント訓練のメインスクリプト
- `evaluate_agent.py`: モデル評価のスクリプト
- `rl_navigator.py`: 訓練済みモデルによるナビゲーション実行

## 実装詳細

### 状態空間

- LiDARスキャンの24方向のサンプル（障害物検知用）
- 目標地点までの正規化された距離
- 目標地点へのヘディング角度（正規化済み）

### 行動空間

5つの離散的な行動：
- 左に大きく回転
- 左に小さく回転
- 直進
- 右に小さく回転
- 右に大きく回転

### 報酬関数

- 目標地点に到達：+100
- 障害物との衝突：-100
- 各ステップのペナルティ：-0.1
- 目標地点に近づくボーナス：+0.5 * 距離減少

## カスタマイズ

学習パラメータや環境設定は、各Pythonファイル内のコンストラクタで調整できます：

```python
# DQNハイパーパラメータの例
self.gamma = 0.99       # 割引率
self.epsilon = 1.0      # 探索率（初期値）
self.epsilon_decay = 0.9995  # 探索率の減衰
self.learning_rate = 0.001  # 学習率
```

## 結果の可視化

訓練中のメトリクスは定期的に`models/dqn_[timestamp]/`ディレクトリに保存されます：
- エピソード報酬
- 探索率（イプシロン）
- 成功率の移動平均

評価結果は`results/evaluation_[timestamp]/`ディレクトリに保存されます：
- ステップ数の分布
- 経路長の分布
- 成功率、衝突率、タイムアウト率の統計
