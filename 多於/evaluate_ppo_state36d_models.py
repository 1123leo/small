import pandas as pd  # 用於數據分析
import matplotlib.pyplot as plt
import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import torch
import random
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from collections import deque
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

class SmoothFlappyEnv(gym.Wrapper):
    def __init__(self, env, n_stack=4):
        super().__init__(env)

        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)

        self.last_score = 0
        self.last_action = 0

        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(9 * n_stack,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)

        self.last_score = 0
        self.last_action = 0

        features = self._extract(obs)
        self.frames.clear()
        for _ in range(self.n_stack):
            self.frames.append(features)

        return self._get_obs(), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        reward = 0.05

        score = info.get("score", 0)
        if score > self.last_score:
            reward += 20.0
            self.last_score = score

        if terminated or truncated:
            reward = -15.0

        bird_y = obs[9]
        bird_vel = obs[10]

        p1_x, p1_top, p1_bot = obs[3], obs[4], obs[5]

        if p1_x < -0.1:
            gap_center = (obs[7] + obs[8]) / 2.0
        else:
            gap_center = (p1_top + p1_bot) / 2.0

        dist = abs(bird_y - gap_center)
        reward += 0.1 * np.exp(-dist * 5)

        if bird_vel < -0.6 and bird_y > gap_center:
            reward -= 0.1

        if action == 0 and abs(bird_y - gap_center) < 0.1:
            reward += 0.05

        self.last_action = action

        self.frames.append(self._extract(obs))

        return self._get_obs(), reward, terminated, truncated, info

    def _extract(self, obs):
        bird_y = obs[9]
        bird_vel = obs[10]

        p1_x, p1_top, p1_bot = obs[3], obs[4], obs[5]
        p2_x, p2_top, p2_bot = obs[6], obs[7], obs[8]

        if p1_x < -0.1:
            target_x, target_top, target_bot = p2_x, p2_top, p2_bot
        else:
            target_x, target_top, target_bot = p1_x, p1_top, p1_bot

        gap_center = (target_top + target_bot) / 2.0

        return np.array([
            bird_y,
            bird_vel,
            target_x,
            bird_y - gap_center,
            target_top,
            target_bot,
            bird_vel * target_x,
            1.0 if bird_vel < -0.5 else 0.0,
            p2_x if p1_x < 0 else 0
        ], dtype=np.float32)

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)

# 1. 載入最新的 V2 模型
model = PPO.load("best_model")    
    # 創建測試環境（帶 wrapper）
test_env_raw = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
test_env = SmoothFlappyEnv(test_env_raw, n_stack=4)
# ... (前面的環境定義與模型載入部分保持不變) ...

test_episodes = 100  # 增加測試輪次
scores = []
steps_list = []

print(f"開始進行 {test_episodes} 輪大規模測試...")
# 🔍 診斷專用：分析 0 分種子 (修正版)
zero_score_seeds = [1019, 1049, 1079, 1089, 1099]

print("🔍 開始診斷 0 分種子的死亡現場...\n")

for seed in zero_score_seeds:
    obs, _ = test_env.reset(seed=seed)
    done = False
    step_count = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        step_count += 1
        
        if done:
            # 直接從當前 obs (36維) 提取最後一幀的 9 個特徵
            last_features = obs[-9:] 
            
            bird_y = last_features[0]
            bird_vel = last_features[1]
            target_y_diff = last_features[3] # bird_y - gap_center
            
            print(f"--- 種子 {seed} ---")
            print(f"存活步數: {step_count}")
            print(f"死亡高度 (y): {bird_y:.4f}")
            print(f"垂直速度 (vel): {bird_vel:.4f}")
            print(f"與中心距離: {target_y_diff:.4f}")
            
            # 根據 Flappy Bird 環境常見數值判斷
            if bird_y < -0.8: 
                print("💀 診斷結果：直接墜地 (開場完全沒跳或跳太少)")
            elif bird_y > 0.8:
                print("💀 診斷結果：飛太高 (開場跳過頭)")
            else:
                print("💀 診斷結果：精準碰撞 (管子高度刁鑽或時機不對)")
            print("-" * 20)
            break

for ep in range(test_episodes):
    obs, _ = test_env.reset(seed=ep + 1000)  # 使用不同的種子區間
    done = False
    step_count = 0
    
    while not done and step_count < 50000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        step_count += 1
        
        if done:
            final_score = info.get('score', 0)
            scores.append(final_score)
            steps_list.append(step_count)
            if (ep + 1) % 10 == 0:
                print(f"進度: {ep + 1}/{test_episodes} | 當前得分: {final_score}")
            break

# 數據分析
df = pd.DataFrame({'Score': scores, 'Steps': steps_list})

print("\n" + "="*30)
print(f"📊 測試報告 (N={test_episodes})")
print(f"平均得分 (Mean):   {df['Score'].mean():.2f}")
print(f"中位數 (Median):   {df['Score'].median():.2f}")
print(f"標準差 (Std Dev):  {df['Score'].std():.2f}")
print(f"最高得分 (Max):    {df['Score'].max()}")
print(f"最低得分 (Min):    {df['Score'].min()}")
print(f"穩定度 (CV%):      {(df['Score'].std()/df['Score'].mean()*100):.1f}% (越低越穩)")
print("="*30)

# 繪製得分分佈圖
plt.figure(figsize=(10, 5))
plt.hist(scores, bins=20, color='skyblue', edgecolor='black')
plt.title('Score Distribution Over 100 Episodes')
plt.xlabel('Score (Pipes)')
plt.ylabel('Frequency')
plt.show()

test_env.close()