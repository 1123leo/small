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
            p2_x if p1_x < 0.0 else 0
        ], dtype=np.float32)

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)

# 1. 載入最新的 V2 模型
model = PPO.load("best_model")    
    # 創建測試環境（帶 wrapper）
test_env_raw = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
test_env = SmoothFlappyEnv(test_env_raw, n_stack=4)
    
scores = []
for ep in range(5):  # 測試5輪
    obs, _ = test_env.reset(seed=ep + 1019)  # 用不同 seed
    done = False
    score = 0
    step_count = 0
        
    while not done and step_count < 10000000:  # 防止無限循環
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        step_count += 1
            
        if terminated or truncated:
            final_score = info.get('score', 0)
            scores.append(final_score)
            print(f"測試輪次 {ep+1}: {final_score} 根管子 (存活 {step_count} 步)")
            break
    
print(f"\n測試結果:")
print(f"  平均: {np.mean(scores):.1f} 根")
print(f"  最高: {max(scores)} 根")
print(f"  最低: {min(scores)} 根")
    
test_env.close()