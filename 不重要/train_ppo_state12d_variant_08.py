import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import torch
import random
import os
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# =========================================================
# 1. 全域隨機種子設定 (確保實驗可重複性)
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 讓 cuDNN 運算結果確定化 (會稍微犧牲一點點速度，但對比最準)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)

# =========================================================
# 2. 環境 Wrapper (修正 Reset 漏洞與獎勵縮放)
# =========================================================
class CV12FlappyEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(12,), dtype=np.float32
        )
        self.last_score = 0

    def reset(self, seed=None, options=None):
        # 核心修正：必須將 seed 傳遞給底層環境
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_score = 0
        return self._build_obs(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # --- 獎勵設計區 (建議將數值縮小 10 倍以提升穩定性) ---
        reward = 0.01  # 生存獎勵 (原 0.05 -> 0.01)
        
        # 死亡懲罰
        if terminated or truncated:
            reward = -2.5  # 原 -20.0 -> -2.0

        # 過關獎勵
        score = info.get("score", 0)
        if score > self.last_score:
            reward += 2  # 原 15.0 -> 1.5
            self.last_score = score

        # 引導獎勵：鼓勵鳥靠近水管缺口中心 (可選，有助於突破 50 步)
        # bird_y = obs[9]
        # gap1_cy = (obs[4] + obs[5]) / 2.0
        # reward -= 0.001 * abs(bird_y - gap1_cy) 

        return self._build_obs(obs), reward, terminated, truncated, info

    def _build_obs(self, obs):
        # 官方索引對齊
        p1_x, p1_y_t, p1_y_b = obs[3], obs[4], obs[5]
        p2_x, p2_y_t, p2_y_b = obs[6], obs[7], obs[8]
        bird_y, bird_v, bird_r = obs[9], obs[10], obs[11]

        gap1_cy = (p1_y_t + p1_y_b) / 2.0
        gap2_cy = (p2_y_t + p2_y_b) / 2.0
        
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
# 3. 訓練主程式
# =========================================================
if __name__ == "__main__":
    # 建立目錄
    os.makedirs("./models/best/", exist_ok=True)
    os.makedirs("./tb_logs/", exist_ok=True)

    env_kwargs = {"render_mode": None, "use_lidar": False}
    policy_kwargs = dict(net_arch=[128, 128, 128]) 

    # 訓練環境 (加入 Seed)
    train_env = make_vec_env(
        lambda: CV12FlappyEnv(gym.make("FlappyBird-v0", **env_kwargs)),
        n_envs=4,
        seed=SEED
    )

    # 評估環境
    eval_env = CV12FlappyEnv(gym.make("FlappyBird-v0", **env_kwargs))
    eval_env.reset(seed=SEED)

    # PPO 模型設定
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=2e-3,   # 建議降至 3e-4，2e-3 在 RL 中極容易跑飛
        n_steps=1024,
        batch_size=512,
        gamma=0.99,
        ent_coef=0.06,        # 稍微提高探索，防止太快變成 PPO_19 的死腦筋
        clip_range=0.7,       # 標準 PPO 常用 0.2
        verbose=1,
        seed=SEED,            # 核心：固定模型種子
        tensorboard_log="./tb_logs/"
    )

    # Callbacks
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path="./models/best/", 
        log_path="./tb_logs/",
        eval_freq=10000, 
        deterministic=True, 
        render=False
    )
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path="./models/")

    print(f"🚀 實驗啟動 (Seed: {SEED})")
    model.learn(
        total_timesteps=2_000_000, 
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="PPO"
    )
    
    model.save("ppo_flappy_final_v1")
    print("✅ 訓練完成！")