import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # ハイパーパラメータの調整
        self.gamma = 0.99  # 割引率
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995  # より緩やかな減衰
        self.learning_rate = 0.001
        self.update_target_frequency = 1000  # ステップ単位での更新に変更
        self.target_update_counter = 0
        
        # メモリ (経験再生バッファ)
        self.memory = deque(maxlen=100000)
        self.batch_size = 64
        
        # ネットワーク
        self.model = self._build_model()  # Q-Network
        self.target_model = self._build_model()  # Target Q-Network
        self.update_target_model()
        
        # メトリクス
        self.loss_history = []
        
    def _build_model(self):
        # Deep-Q学習用のニューラルネットワークモデル
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        # モデルの重みをターゲットモデルにコピー
        self.target_model.set_weights(self.model.get_weights())
    
    def memorize(self, state, action, reward, next_state, done):
        # 経験をメモリに保存
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        # イプシロンでランダム行動または最適行動を選択
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self):
        # メモリからの経験でエージェントを訓練
        if len(self.memory) < self.batch_size:
            return
        
        # ミニバッチをランダムにサンプリング
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # ターゲットQ値の計算
        target = rewards + (1 - dones) * self.gamma * np.amax(
            self.target_model.predict(next_states, verbose=0), axis=1
        )
        
        # 現在のQ値を取得し、選択した行動のターゲットで更新
        target_f = self.model.predict(states, verbose=0)
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        
        # モデルの訓練
        history = self.model.fit(states, target_f, epochs=1, verbose=0)
        self.loss_history.append(history.history['loss'][0])
        
        # ターゲットネットワークの更新カウンター
        self.target_update_counter += 1
        if self.target_update_counter >= self.update_target_frequency:
            self.update_target_model()
            self.target_update_counter = 0
            
        # 探索と活用のトレードオフのためにイプシロンを減衰
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, filepath):
        try:
            self.model.load_weights(filepath)
            self.target_model.load_weights(filepath)
            print(f"モデルを {filepath} から読み込みました")
        except:
            print(f"{filepath} にモデルが見つかりませんでした")
    
    def save(self, filepath):
        self.model.save_weights(filepath)
        print(f"モデルを {filepath} に保存しました")