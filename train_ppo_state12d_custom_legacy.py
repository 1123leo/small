import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import gc
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

# =========================================================
# 1. CV ⇄ Gym 對齊的 12 維環境
# =========================================================
class CV12FlappyEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )

        self.last_score = 0
        self.last_bird_y = 0.0
        self.last_pipe_x1 = 0.0
        self.last_pipe_x2 = 0.0

    def reset(self, **kwargs):
        self.last_score = 0
        obs, info = self.env.reset(**kwargs)

        bird_y = obs[9]
        self.last_bird_y = bird_y

        pipe1_x = obs[3]
        pipe2_x = obs[6]

        self.last_pipe_x1 = pipe1_x
        self.last_pipe_x2 = pipe2_x

        return self._build_obs(obs), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

    # === 一定先初始化 ===
        reward = 0.02
        score = info.get("score", self.last_score)

    # === 死亡懲罰（小而固定）===
        if terminated or truncated:
            reward = -5.0
        else:
        # ===== 方向性 shaping（關鍵）=====
            bird_y = obs[9]
            bird_vy = bird_y - self.last_bird_y

            pipe1_top, pipe1_bot = obs[4], obs[5]
            gap1_cy = (pipe1_top + pipe1_bot) / 2
            dy1 = bird_y - gap1_cy

        # 1️⃣ 越接近中心越好
            reward += 1.0 - abs(dy1) / 800.0

        # 2️⃣ 如果「方向錯誤」，給懲罰（這行是破關關鍵）
            reward += -0.3 * np.sign(dy1) * bird_vy

    # === 過管獎勵 ===
            score = info.get("score", 0)
        if score > self.last_score:
            reward += 10.0
            self.last_score = score

        return self._build_obs(obs), reward, terminated, truncated, info


    def _build_obs(self, obs):
        # 原始 obs（非 lidar）定義
        

        bird_y = obs[9]
        bird_vy = bird_y - self.last_bird_y

        pipe1_x, pipe1_top, pipe1_bot = obs[3], obs[4], obs[5]
        pipe2_x, pipe2_top, pipe2_bot = obs[6], obs[7], obs[8]

        pipe1_vx = pipe1_x - self.last_pipe_x1
        pipe2_vx = pipe2_x - self.last_pipe_x2

        gap1_cy = (pipe1_top + pipe1_bot) / 2
        gap2_cy = (pipe2_top + pipe2_bot) / 2

        dy1 = bird_y - gap1_cy
        dy2 = bird_y - gap2_cy

        self.last_bird_y = bird_y
        self.last_pipe_x1 = pipe1_x
        self.last_pipe_x2 = pipe2_x
        

        # ===== 正規化（與 CV deploy 完全一致）=====
        return np.array([
            bird_y / 800.0,
            bird_vy / 10.0,

            pipe1_x / 500.0,
            gap1_cy / 800.0,
            (pipe1_bot - pipe1_top) / 800.0,
            pipe1_vx / 10.0,

            pipe2_x / 500.0,
            gap2_cy / 800.0,
            (pipe2_bot - pipe2_top) / 800.0,
            pipe2_vx / 10.0,

            dy1 / 800.0,
            dy2 / 800.0
        ], dtype=np.float32)


# =========================================================
# 2. GC Callback
# =========================================================
class MemoryCleanupCallback(BaseCallback):
    def __init__(self, freq=200_000):
        super().__init__()
        self.freq = freq

    def _on_step(self):
        if self.n_calls % self.freq == 0:
            gc.collect()
        return True


# =========================================================
# 3. 建立環境
# =========================================================
env_kwargs = {"render_mode": None, "use_lidar": False}

train_env = make_vec_env(
    lambda: CV12FlappyEnv(gym.make("FlappyBird-v0", **env_kwargs)),
    n_envs=4
)

eval_env = CV12FlappyEnv(gym.make("FlappyBird-v0", **env_kwargs))


# =========================================================
# 4. PPO（已為 12 維調過）
# =========================================================
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=256,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    verbose=1,
    tensorboard_log="./tb_logs/"
)

model.learn(
    total_timesteps=500_000,
    callback=[
        EvalCallback(eval_env, eval_freq=50_000),
        CheckpointCallback(save_freq=100_000, save_path="./models/"),
        MemoryCleanupCallback()
    ]
)

model.save("ppo_flappy_cv12")
print("✅ 訓練完成（12 維，MDP，可實戰）")
