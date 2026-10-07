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
# 1. 全域隨機種子與工具函式
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def linear_schedule(initial_value: float):
    """ 學習率線性衰減：隨著訓練進度(1.0 -> 0.0)降低學習率 """
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func

SEED = 42
set_seed(SEED)

# =========================================================
# 2. 環境 Wrapper (針對 2000 步長度優化)
# =========================================================
class CV12FlappyEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # 12 維特徵向量
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(12,), dtype=np.float32
        )
        self.last_score = 0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_score = 0
        return self._build_obs(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # --- 長時序獎勵設計 ---
        # 1. 穩定的生存獎勵
        reward = 0.01 
        
        # 2. 死亡懲罰 (適中即可，過重會導致鳥變得很膽小不敢飛)
        if terminated or truncated:
            reward = -2.0 

        # 3. 過關獎勵
        score = info.get("score", 0)
        if score > self.last_score:
            reward += 1.5 
            self.last_score = score

        # 4. 核心引導獎勵：鼓勵鳥保持在下一個缺口的中心
        # 這是達成 2000 步的秘密，防止在空窗期隨機飄移
        bird_y = obs[9]
        gap1_cy = (obs[4] + obs[5]) / 2.0
        dist_to_gap = abs(bird_y - gap1_cy) / 512.0
        reward -= 0.1 * dist_to_gap  # 距離中心越遠扣分越多

        return self._build_obs(obs), reward, terminated, truncated, info

    def _build_obs(self, obs):
        # 索引對齊與歸一化 (Normalization)
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
    os.makedirs("./models/best/", exist_ok=True)
    os.makedirs("./tb_logs/", exist_ok=True)

    env_kwargs = {"render_mode": None, "use_lidar": False}
    # 增加網路深度以處理更複雜的時序邏輯
    policy_kwargs = dict(net_arch=[128, 128])

    # 訓練環境
    train_env = make_vec_env(
        lambda: CV12FlappyEnv(gym.make("FlappyBird-v0", **env_kwargs)),
        n_envs=4,
        seed=SEED
    )

    # 評估環境
    eval_env = CV12FlappyEnv(gym.make("FlappyBird-v0", **env_kwargs))

    # PPO 模型設定 (針對長時序目標優化)
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=linear_schedule(5e-4), # 使用衰減學習率，初始設為稍大的 5e-4
        n_steps=2048,          # 增加單次更新的樣本長度
        batch_size=128,        # 較小的 batch 有助於捕捉細微特徵
        gamma=0.999,           # 核心修改：極高的折扣係數以看見未來
        ent_coef=0.005,        # 降低探索，中後期以穩定為主
        clip_range=0.2,        # 收緊 Clip，確保穩定性 (不使用 0.3)
        gae_lambda=0.95,
        verbose=1,
        seed=SEED,
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

    print(f"🚀 目標 2000 步實驗啟動 (Seed: {SEED}, Gamma: {model.gamma})")
    model.learn(
        total_timesteps=2_000_000, 
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="PPO_LongTerm"
    )
    
    model.save("ppo_flappy_2000step_v1")
    print("✅ 訓練完成！")