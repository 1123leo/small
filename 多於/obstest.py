import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import torch
import random
import os
from collections import deque
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
# 診斷腳本：確認 obs 索引正確
env = gym.make("FlappyBird-v0", use_lidar=False)
obs, _ = env.reset()

print(f"bird_y (obs[9]): {obs[9]:.3f} - 應該在 0.3~0.7 之間")
print(f"bird_vel (obs[10]): {obs[10]:.3f} - 初始應該接近 0")

# 不斷跳躍，觀察變化
for i in range(10):
    obs, _, _, _, _ = env.step(1 if i % 5 == 0 else 0)  # 間隔跳躍
    print(f"Step {i}: y={obs[9]:.3f}, vel={obs[10]:.3f}, next_pipe_x={obs[3]:.3f}")
