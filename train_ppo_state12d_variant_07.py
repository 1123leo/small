import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import gc
import os
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# =========================================================
# 1. 精準對齊環境 (索引對齊：解決 50 步魔咒)
# =========================================================
class CV12FlappyEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(12,), dtype=np.float32
        )
        self.last_score = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_score = 0
        return self._build_obs(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # 基礎獎勵 (讓 AI 知道活著是有價值的)
        reward = 0.05 
        
        # --- 邊界懲罰：防止 AI 原地躺平 ---
        bird_y = obs[9]
        

        # --- 死亡懲罰：維持高壓 ---
        if terminated or truncated:
            reward = -10.0 

        # --- 過關獎勵：調降至 5.0，穩定 Explained Variance ---
        score = info.get("score", 0)
        if score > self.last_score:
            reward += 10.0  
            self.last_score = score

        return self._build_obs(obs), reward, terminated, truncated, info

    def _build_obs(self, obs):
        # 官方索引對齊：100% 確保 bird_y 在 [9]
        p1_x, p1_y_t, p1_y_b = obs[3], obs[4], obs[5]
        p2_x, p2_y_t, p2_y_b = obs[6], obs[7], obs[8]
        bird_y, bird_v, bird_r = obs[9], obs[10], obs[11]

        gap1_cy = (p1_y_t + p1_y_b) / 2.0
        gap2_cy = (p2_y_t + p2_y_b) / 2.0
        
        # 正規化輸出
        return np.array([
            bird_y / 512.0,
            bird_v / 10.0,
            p1_x / 288.0,
            gap1_cy / 512.0,
            (p1_y_b - p1_y_t) / 512.0,
            p2_x / 288.0,
            gap2_cy / 512.0,
            (p2_y_b - p2_y_t) / 512.0,
            (bird_y - gap1_cy) / 512.0,
            (bird_y - gap2_cy) / 512.0,
            bird_r / 90.0,
            (p1_x - p2_x) / 288.0
        ], dtype=np.float32)

# =========================================================
# 2. 訓練設定
# =========================================================
if __name__ == "__main__":
    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./tb_logs/", exist_ok=True)

    env_kwargs = {"render_mode": None, "use_lidar": False}
    # 加深網路為 [128, 128, 128] 以處理 200 根水管所需的複雜預判
    policy_kwargs = dict(net_arch=[128, 128, 128]) 

    train_env = make_vec_env(
        lambda: CV12FlappyEnv(gym.make("FlappyBird-v0", **env_kwargs)),
        n_envs=4
    )
    eval_env = CV12FlappyEnv(gym.make("FlappyBird-v0", **env_kwargs))

    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=2e-3,   # 微調級學習率
        n_steps=1024,
        batch_size=256,
        gamma=0.99,
        ent_coef=0.005,      # 降低隨機探索，固定肌肉記憶
        clip_range=0.3,      # 縮小更新幅度，確保不崩潰
        verbose=1,
        tensorboard_log="./tb_logs/"
    )

    callbacks = [
        EvalCallback(eval_env, best_model_save_path="./models/best/", eval_freq=20000),
        CheckpointCallback(save_freq=100000, save_path="./models/"),
    ]

    print("🚀 訓練啟動：目標突破 50 步死線，直衝 200 根水管...")
    model.learn(total_timesteps=2_000_000, callback=callbacks)
    model.save("ppo_flappy_cv12_final")