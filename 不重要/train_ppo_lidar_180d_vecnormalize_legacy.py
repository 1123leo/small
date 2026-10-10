import os
import gc
from pathlib import Path

import gymnasium as gym
import flappy_bird_gymnasium  # ensures envs are registered
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback

# ---------------------------
# Helper: robust env creation (tries common names)
# ---------------------------
def make_flappy_env(render_mode=None, use_lidar=True):
    candidates = ["FlappyBird-v0"]
    last_err = None
    for name in candidates:
        try:
            env = gym.make(name, render_mode=render_mode, use_lidar=use_lidar)
            return env
        except Exception as e:
            last_err = e
    raise RuntimeError(f"No FlappyBird environment found. Last error: {last_err}")

# ---------------------------
# Memory cleanup callback
# ---------------------------
class MemoryCleanupCallback(BaseCallback):
    def __init__(self, cleanup_freq: int = 200_000, verbose: int = 0):
        super().__init__(verbose)
        self.cleanup_freq = int(cleanup_freq)

    def _on_step(self) -> bool:
        if self.n_calls > 0 and (self.n_calls % self.cleanup_freq == 0):
            gc.collect()
            if self.verbose:
                print(f"[MemoryCleanup] gc.collect() at step {self.n_calls}")
        return True

# ---------------------------
# Training config
# ---------------------------
OUTPUT_DIR = Path("./flappy_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_ENVS = 8
TOTAL_TIMESTEPS = 3_000_000  # 建議長訓練，視資源調整
EVAL_FREQ = 100_000
CHECKPOINT_FREQ = 100_000
TB_LOG = "./tb_logs_flappy"

# ---------------------------
# Build envs
# ---------------------------
# training envs: vectorized, no render
def _init():
    return make_flappy_env(render_mode=None, use_lidar=True)

train_vec = make_vec_env(_init, n_envs=N_ENVS)
train_vec = VecMonitor(train_vec)
train_vec = VecNormalize(train_vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

# eval env (single)
eval_env = make_flappy_env(render_mode=None, use_lidar=True)
# NOTE: Do NOT wrap eval_env with VecNormalize from train (we keep separate for eval stability)

# ---------------------------
# PPO model (stable settings for long-run training)
# ---------------------------
policy_kwargs = dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])])

model = PPO(
    "MlpPolicy",
    train_vec,
    learning_rate=5e-5,
    n_steps=4096,            # 長序列 -> 更穩定的長期回報估計
    batch_size=1024,
    n_epochs=10,
    gamma=0.999,
    gae_lambda=0.97,
    ent_coef=0.02,           # 保持微量探索，避免完全僵化
    clip_range=0.2,
    policy_kwargs=policy_kwargs,
    verbose=1,
    tensorboard_log=TB_LOG,
    device="auto",
)

# ---------------------------
# Callbacks
# ---------------------------
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=str(OUTPUT_DIR / "best_model"),
    log_path=str(OUTPUT_DIR / "eval_logs"),
    eval_freq=EVAL_FREQ,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
)

checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=str(OUTPUT_DIR / "checkpoints"),
    name_prefix="ppo_flappy"
)

memory_callback = MemoryCleanupCallback(cleanup_freq=200_000, verbose=1)

# ---------------------------
# Train
# ---------------------------
if __name__ == "__main__":
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[eval_callback, checkpoint_callback, memory_callback],
            tb_log_name="ppo_flappy_lidar"
        )
        # save final model and VecNormalize stats
        model.save(str(OUTPUT_DIR / "ppo_flappy_final"))
        # Save VecNormalize statistics so we can use same normalization at test time
        train_vec.save(str(OUTPUT_DIR / "vec_normalize.pkl"))
    finally:
        train_vec.close()
        eval_env.close()
        gc.collect()
