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

# =========================================================
# 1. 完整 Seed 固定
# =========================================================
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

# =========================================================
# 2. Wrapper (保持 36 維特徵提取)
# =========================================================
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
            reward += 0.2
            self.last_score = score
        if terminated or truncated:
            reward = -0.15  # 稍微加重死亡懲罰以維持穩定性

        bird_y = obs[9]
        bird_vel = obs[10]
        p1_x, p1_top, p1_bot = obs[3], obs[4], obs[5]

        if p1_x < -0.1:
            gap_center = (obs[7] + obs[8]) / 2.0
        else:
            gap_center = (p1_top + p1_bot) / 2.0

        dist = abs(bird_y - gap_center)
        reward += 0.2 * np.exp(-dist * 5) # 強化中心對齊獎勵

        if bird_vel < -0.6 and bird_y > gap_center:
            reward -= 0.1
        if action == 0 and abs(bird_y - gap_center) < 0.1:
            reward += 0.05

        self.last_action = action
        self.frames.append(self._extract(obs))
        return self._get_obs(), reward, terminated, truncated, info

    def _extract(self, obs):
        bird_y, bird_vel = obs[9], obs[10]
        p1_x, p1_top, p1_bot = obs[3], obs[4], obs[5]
        p2_x, p2_top, p2_bot = obs[6], obs[7], obs[8]
        if p1_x < -0.1:
            target_x, target_top, target_bot = p2_x, p2_top, p2_bot
        else:
            target_x, target_top, target_bot = p1_x, p1_top, p1_bot
        gap_center = (target_top + target_bot) / 2.0
        return np.array([
            bird_y, bird_vel, target_x, bird_y - gap_center,
            target_top, target_bot, bird_vel * target_x,
            1.0 if bird_vel < -0.5 else 0.0,
            p2_x if p1_x < 0 else 0
        ], dtype=np.float32)

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)

# =========================================================
# 3. 環境與調度器建立
# =========================================================
def make_env(rank, seed=42):
    def _init():
        env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)
        return Monitor(SmoothFlappyEnv(env, n_stack=4))
    return _init

train_env = DummyVecEnv([make_env(i) for i in range(8)])
eval_env = DummyVecEnv([make_env(0, seed=999)])

# 金隼控制：學習率與 Clip Range 的雙重鎖定點
TOTAL_STEPS = 10000000
TRIGGER_STEP = 0

def lr_schedule(initial_value, min_value=0.0003, total_timesteps=TOTAL_STEPS, plateau_steps=TRIGGER_STEP):
    def func(progress_remaining: float) -> float:
        current_step = (1.0 - progress_remaining) * total_timesteps
        if current_step < plateau_steps:
            return initial_value
        else:
            decay_progress = (current_step - plateau_steps) / (total_timesteps - plateau_steps)
            return initial_value - (initial_value - min_value) * decay_progress
    return func

# def clip_schedule(progress_remaining: float) -> float:
#     current_step = (1.0 - progress_remaining) * TOTAL_STEPS
#     # 28 萬步之後，將 Clip Range 從 0.4 強制降到 0.1，鎖死斜率
#     return 0.4 if current_step < TRIGGER_STEP else 0.1

# =========================================================
# 4. PPO 模型 (參數精準打擊版)
# =========================================================
model = PPO(
    "MlpPolicy",
    train_env,
    # learning_rate=1e-3,
    learning_rate=lr_schedule(1e-3, 3e-4),
    clip_range=0.2,  # 導入動態 Clip 控制
    n_steps=2048,
    batch_size=1024,            # 縮小 batch_size 讓更新更頻繁且穩定
    gamma=0.995,
    ent_coef=0.07,             # 降低初始熵，配合 36 維觀測更精準
    gae_lambda=0.95,
    verbose=1,
    seed=SEED,
    tensorboard_log="./tb_logs/",
    policy_kwargs=dict(
        net_arch=[128, 128, 64],
        activation_fn=torch.nn.ReLU
    ),
    n_epochs=5,               # 增加 epoch 提高穩定度
)

# =========================================================
# 5. Callback 與 訓練
# =========================================================
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./models/best/",
    eval_freq=16384,           # 提高評估頻率，更早捕捉 320k 附近的巔峰
    deterministic=True,
    render=False,
    n_eval_episodes=10
)

checkpoint_cb = CheckpointCallback(
    save_freq=50000,
    save_path="./models/checkpoints/",
    name_prefix="ppo_flappy"
)

model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_cb, checkpoint_cb])
model.save("ppo_500plus_final")

print("✅ 訓練完成！")