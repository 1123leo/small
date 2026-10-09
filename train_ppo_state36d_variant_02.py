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


SEED = 42
set_seed(SEED)

# =========================================================
# 2. Wrapper（36維：9維 × 4幀）
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

# =========================================================
# 3. 環境建立（完全固定 seed）
# =========================================================
def make_env(rank, seed=42):
    def _init():
        env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)

        env.reset(seed=seed + rank)
        env.action_space.seed(seed + rank)
        env.observation_space.seed(seed + rank)

        return Monitor(SmoothFlappyEnv(env, n_stack=4))
    return _init


train_env =DummyVecEnv (
    [make_env(i) for i in range(8)]
)

eval_env = DummyVecEnv([
    make_env(0, seed=999)
])
def lr_schedule(initial_value, min_value=0.0003, total_timesteps=700000, plateau_steps=300000):
    """
    精確控制步數的學習率調度器
    - initial_value: 一開始的最高學習率
    - min_value: 最終降到的最低學習率
    - total_timesteps: 你在 model.learn 填的總步數
    - plateau_steps: 平台期要維持多少步才開始下降
    """
    def func(progress_remaining: float) -> float:
        # 計算當前進度（從 0.0 增加到 1.0）
        current_progress = 1.0 - progress_remaining
        # 換算成當前大約步數
        current_step = current_progress * total_timesteps
        
        if current_step < plateau_steps:
            # 1. 平台期：還沒到指定步數，維持原價
            return initial_value
        else:
            # 2. 衰減期：計算剩餘步數的比例，線性降至 min_value
            # 在衰減區間內的進度 (從 0.0 變到 1.0)
            decay_progress = (current_step - plateau_steps) / (total_timesteps - plateau_steps)
            return initial_value - (initial_value - min_value) * decay_progress
            
    return func

# def lr_schedule(progress):
#     # #     # 前 50% 快速學習，後 50% 微調
#     if progress > 0.7:
#         return 1e-3 * (progress * 2)
    
# def lr_schedule(initial_value, min_value=0.0005):
#     def func(progress_remaining: float) -> float:
#     #     # progress_remaining 從 1.0 降到 0.0
#         return max(initial_value * progress_remaining, min_value)
#     return func

# =========================================================
# 4. PPO 模型
# =========================================================
model = PPO(
    "MlpPolicy",
    train_env,
    #learning_rate=1e-3,
    learning_rate=lr_schedule(1e-3, 3e-4, 700000, 300000),
    #(0.001, min_value=0.0005),
    n_steps=2048,
    batch_size=1024,
    gamma=0.995,
    ent_coef=0.07,
    clip_range=0.4,
    gae_lambda=0.95,
    verbose=1,
    seed=SEED,
    tensorboard_log="./tb_logs/",
    policy_kwargs=dict(
        net_arch=[128, 128, 64],
        activation_fn=torch.nn.ReLU
    ),
    n_epochs=5,
)

# =========================================================
# 5. Callback
# =========================================================
eval_cb = EvalCallback(
    eval_env,
    best_model_save_path="./models/best/",
    eval_freq=20000,
    deterministic=True,
    render=False,
    n_eval_episodes=10
)

checkpoint_cb = CheckpointCallback(
    save_freq=50000,
    save_path="./models/checkpoints/",
    name_prefix="ppo_flappy"
)

# =========================================================
# 6. 訓練
# =========================================================
model.learn(total_timesteps=70_0000, callback=[eval_cb, checkpoint_cb])
model.save("ppo_500plus_final")
    
        # ========== 測試最佳模型（修正版）==========
print("✅ 訓練完成！測試最佳模型...")
    
    
    # 載入最佳模型
best_model = PPO.load("./models/500plus/best_model")
    
    # 創建測試環境（帶 wrapper）
test_env_raw = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
test_env = SmoothFlappyEnv(test_env_raw, n_stack=4)
    
scores = []
for ep in range(5):  # 測試5輪
    obs, _ = test_env.reset(seed=ep + 100)  # 用不同 seed
    done = False
    score = 0
    step_count = 0
        
    while not done and step_count < 10000:  # 防止無限循環
        action, _ = best_model.predict(obs, deterministic=True)
        obs, reward, done, trun, info = test_env.step(action)
        step_count += 1
            
        if done or trun:
            final_score = info.get('score', 0)
            scores.append(final_score)
            print(f"測試輪次 {ep+1}: {final_score} 根管子 (存活 {step_count} 步)")
            break
    
print(f"\n測試結果:")
print(f"  平均: {np.mean(scores):.1f} 根")
print(f"  最高: {max(scores)} 根")
print(f"  最低: {min(scores)} 根")
    
test_env.close()
