import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import torch
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gc

# 1. 完整 Seed 固定
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

SEED = 42
set_seed(SEED)

# 2. 定義動態衰減函數，防止訓練末期鎖死
def linear_schedule(initial_value, min_value=1e-4):
    def func(progress_remaining: float) -> float:
        # progress_remaining 從 1.0 降到 0.0
        return max(initial_value * progress_remaining, min_value)
    return func

# 3. 自定義環境：強化獎勵邏輯
class CustomFlappyBirdEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_score = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 修正：大幅降低生存獎勵，避免 AI 變成「膽小鬼」
        reward = 0.05 
        
        # 修正：通過水管獎勵稍微提高
        current_score = info.get('score', 0)
        if current_score > self.last_score:
            reward += 20.0
            self.last_score = current_score
            
        # 修正：增加死亡懲罰，讓模型明確知道撞擊是負面的
        if terminated:
            reward = -15.0
            
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.last_score = 0.0
        return self.env.reset(**kwargs)

# 4. 記憶體清理回調
class MemoryCleanupCallback(BaseCallback):
    def __init__(self, cleanup_freq=100_000, verbose=0):
        super().__init__(verbose)
        self.cleanup_freq = cleanup_freq
    def _on_step(self) -> bool:
        if self.n_calls % self.cleanup_freq == 0:
            gc.collect()
        return True

# 5. 建立環境
def make_env(rank, seed=SEED):
    def _init():
        env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=True)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        return Monitor(CustomFlappyBirdEnv(env))
    return _init

n_envs = 4
train_env = DummyVecEnv([make_env(i) for i in range(n_envs)])

eval_env = Monitor(CustomFlappyBirdEnv(gym.make("FlappyBird-v0", render_mode=None, use_lidar=True)))
eval_env.reset(seed=999)
eval_env.action_space.seed(999)
eval_env.observation_space.seed(999)

# 6. 模型初始化：調整超參數以增加穩定性與探索性
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=linear_schedule(1e-3, min_value=1e-4), # 保留底線
    clip_range=linear_schedule(0.2, min_value=0.05),    # 保留底線
    n_steps=2048,           # 每次更新前的步數
    batch_size=128,         # 每次優化的樣本數
    ent_coef=0.01,          # 增加探索，防止策略過於僵硬
    gae_lambda=0.95,
    gamma=0.99,
    verbose=1,
    seed=SEED,
    tensorboard_log="./tb_logs/"
)

# 7. 回調設定
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best_model",
    log_path="./logs/",
    eval_freq=max(50_000 // n_envs, 1),
    deterministic=True,
    render=False
)
checkpoint_callback = CheckpointCallback(
    save_freq=max(100_000 // n_envs, 1),
    save_path="./models/",
    name_prefix="ppo_flappybird_optimized"
)
memory_callback = MemoryCleanupCallback(cleanup_freq=200_000)

# 8. 開始訓練
print("開始優化訓練...")
model.learn(
    total_timesteps=1000_000,
    callback=[eval_callback, checkpoint_callback, memory_callback],
    tb_log_name="ppo1"
)

model.save("ppo_flappybird_final")
train_env.close()
eval_env.close()
print("訓練完成！")
