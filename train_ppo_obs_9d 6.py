#!/usr/bin/env python3
"""統一規格的 Flappy Bird PPO 訓練程式；此檔使用 9 維單幀狀態。"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import random
import time
from collections import deque
from importlib import metadata
from pathlib import Path
from typing import Any

# 必須在匯入 PyTorch 前設定 CUDA 確定性運算所需環境變數。
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

try:
    import flappy_bird_gymnasium  # noqa: F401; registers FlappyBird-v0
    import gymnasium as gym
    import numpy as np
    import torch
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少 PPO 執行套件。請先安裝 PyTorch、gymnasium、flappy-bird-gymnasium、"
        "stable-baselines3；若要量測程序 RAM，另安裝 psutil。"
    ) from exc

try:
    import psutil  # optional: process RAM measurement
except ImportError:
    psutil = None

# 三支檔案只更改這一個常數；其餘訓練／評估規格完全相同。
OBSERVATION_MODE = "9d"

# 實驗控制條件：三種 observation 必須使用同一份設定。
N_ENVS = 4
DEFAULT_TOTAL_TIMESTEPS = 1_000_0000
N_STEPS = 2_048
BATCH_SIZE = 256
N_EPOCHS = 10
LEARNING_RATE = 0.0001
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.07
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
HIDDEN_LAYERS = [128, 128]
EVAL_EVERY_STEPS = 100_000
CHECKPOINT_EVERY_STEPS = 100_000
EVAL_EPISODES = 10
EVAL_SEED_BASE = 10_000
MAX_EVAL_STEPS = 50_000

MODE_SPEC = {
    "9d": {"expected_dim": 9, "use_lidar": False, "frame_stack": 1},
    "36d": {"expected_dim": 36, "use_lidar": False, "frame_stack": 4},
    "180d": {"expected_dim": 180, "use_lidar": True, "frame_stack": 0},
}


def set_all_seeds(seed: int) -> None:
    """固定 Python、NumPy 與 PyTorch 隨機種子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None


class SharedRewardWrapper(gym.Wrapper):
    """三組共用的獎勵：每步 +0.05、過管 +20、回合結束 -15。"""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.last_score = 0

    def reset(self, **kwargs: Any):
        self.last_score = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        observation, _environment_reward, terminated, truncated, info = self.env.step(action)
        reward = 0.05
        score = info.get("score", self.last_score)
        if score > self.last_score:
            reward += 0.2
            self.last_score = score
        if terminated or truncated:
            reward = -0.15  # 稍微加重死亡懲罰以維持穩定性
        return observation, reward, terminated, truncated, info


class StateObservationWrapper(gym.Wrapper):
    """把非 Lidar 原始狀態轉成 9 個特徵，並可堆疊最近數幀。"""

    def __init__(self, env: gym.Env, frame_stack: int):
        super().__init__(env)
        if int(np.prod(env.observation_space.shape)) < 11:
            raise ValueError(
                "use_lidar=False 的原始 observation 維度不足；"
                "此 wrapper 需要原始索引 3–10。"
            )
        self.frame_stack = frame_stack
        self.frames: deque[np.ndarray] = deque(maxlen=frame_stack)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9 * frame_stack,),
            dtype=np.float32,
        )

    @staticmethod
    def _features(observation: np.ndarray) -> np.ndarray:
        raw = np.asarray(observation, dtype=np.float32).reshape(-1)
        bird_y, bird_velocity = raw[9], raw[10]
        pipe1_x, pipe1_top, pipe1_bottom = raw[3], raw[4], raw[5]
        pipe2_x, pipe2_top, pipe2_bottom = raw[6], raw[7], raw[8]

        if pipe1_x < -0.1:
            target_x, target_top, target_bottom = pipe2_x, pipe2_top, pipe2_bottom
        else:
            target_x, target_top, target_bottom = pipe1_x, pipe1_top, pipe1_bottom
        gap_center = (target_top + target_bottom) / 2.0

        return np.asarray(
            [
                bird_y,
                bird_velocity,
                target_x,
                bird_y - gap_center,
                target_top,
                target_bottom,
                bird_velocity * target_x,
                1.0 if bird_velocity < -0.5 else 0.0,
                pipe2_x if pipe1_x < 0.0 else 0.0,
            ],
            dtype=np.float32,
        )

    def _stacked(self) -> np.ndarray:
        return np.concatenate(tuple(self.frames), axis=0).astype(np.float32, copy=False)

    def reset(self, **kwargs: Any):
        observation, info = self.env.reset(**kwargs)
        features = self._features(observation)
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(features.copy())
        return self._stacked(), info

    def step(self, action: Any):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._features(observation))
        return self._stacked(), reward, terminated, truncated, info


def make_single_env() -> gym.Env:
    spec = MODE_SPEC[OBSERVATION_MODE]
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=spec["use_lidar"])
    env = SharedRewardWrapper(env)
    if spec["frame_stack"]:
        env = StateObservationWrapper(env, frame_stack=spec["frame_stack"])
    env = Monitor(env)

    actual_dim = int(np.prod(env.observation_space.shape))
    if actual_dim != spec["expected_dim"]:
        env.close()
        raise ValueError(
            f"{OBSERVATION_MODE} 預期輸入 {spec['expected_dim']} 維，"
            f"但目前環境產生 {actual_dim} 維；請檢查 flappy-bird-gymnasium 版本。"
        )
    return env


class ScoreEvaluationCallback(BaseCallback):
    """以固定 seed 評估遊戲管數，並輸出每次檢查點的逐次摘要 CSV。"""

    def __init__(self, output_csv: Path, eval_every: int, episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.output_csv = output_csv
        self.eval_every = eval_every
        self.episodes = episodes
        self.next_eval_at = eval_every
        self.started_at = time.perf_counter()

    def _on_step(self) -> bool:
        while self.num_timesteps >= self.next_eval_at:
            scores: list[int] = []
            lengths: list[int] = []
            for episode in range(self.episodes):
                env = make_single_env()
                try:
                    observation, _ = env.reset(seed=EVAL_SEED_BASE + episode)
                    score = 0
                    length = 0
                    for length in range(1, MAX_EVAL_STEPS + 1):
                        action, _ = self.model.predict(observation, deterministic=True)
                        observation, _reward, terminated, truncated, info = env.step(action)
                        score = int(info.get("score", score))
                        if terminated or truncated:
                            break
                    scores.append(score)
                    lengths.append(length)
                finally:
                    env.close()

            row = {
                "timesteps": int(self.num_timesteps),
                "episodes": self.episodes,
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "median_score": float(np.median(scores)),
                "max_score": int(max(scores, default=0)),
                "mean_episode_steps": float(np.mean(lengths)),
                "eval_seed_base": EVAL_SEED_BASE,
                "elapsed_wall_seconds": time.perf_counter() - self.started_at,
            }
            is_new = not self.output_csv.exists()
            with self.output_csv.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(row))
                if is_new:
                    writer.writeheader()
                writer.writerow(row)

            self.logger.record("eval/score_mean", row["mean_score"])
            self.logger.record("eval/score_std", row["std_score"])
            self.logger.record("eval/score_median", row["median_score"])
            if self.verbose:
                print(
                    f"[eval] step={self.num_timesteps:,}, "
                    f"mean score={row['mean_score']:.2f} ± {row['std_score']:.2f}"
                )
            self.next_eval_at += self.eval_every
        return True


class ResourceMonitorCallback(BaseCallback):
    """選配取樣程序 RAM；psutil 未安裝時仍可正常訓練。"""

    def __init__(self, sample_every_calls: int = 250):
        super().__init__()
        self.sample_every_calls = sample_every_calls
        self.peak_rss_mb: float | None = None
        self.process = None
        if psutil:
            try:
                self.process = psutil.Process(os.getpid())
            except psutil.Error as exc:
                print(f"[resource] RAM sampling unavailable: {exc}")

    def _on_step(self) -> bool:
        if self.process and self.n_calls % self.sample_every_calls == 0:
            try:
                rss_mb = self.process.memory_info().rss / (1024**2)
                self.peak_rss_mb = max(self.peak_rss_mb or 0.0, rss_mb)
            except psutil.Error as exc:
                print(f"[resource] RAM sampling stopped: {exc}")
                self.process = None
        return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"統一規格 Flappy Bird PPO 訓練（{OBSERVATION_MODE} observation）"
    )
    parser.add_argument("--seed", type=int, default=42, help="隨機種子；三組使用相同 seed 配對比較")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=DEFAULT_TOTAL_TIMESTEPS,
        help="總環境步數；比較時三組必須使用相同值",
    )
    args = parser.parse_args()
    if args.total_timesteps <= 0:
        parser.error("--total-timesteps 必須大於 0")

    set_all_seeds(args.seed)
    spec = MODE_SPEC[OBSERVATION_MODE]
    output_dir = Path(__file__).resolve().parent / "models" / "standardized" / f"obs_{OBSERVATION_MODE}_seed_{args.seed}.6"
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(
            f"輸出目錄已有資料，為避免混合或覆寫結果而停止：{output_dir}；"
            "請先封存既有資料，或選用其他 seed。"
        )
    checkpoint_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / "tensorboard"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([make_single_env for _ in range(N_ENVS)])
    train_env.seed(args.seed)
    train_env.action_space.seed(args.seed)

    memory_callback = ResourceMonitorCallback()
    try:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            policy_kwargs={
                "net_arch": {"pi": HIDDEN_LAYERS, "vf": HIDDEN_LAYERS},
                "activation_fn": torch.nn.Tanh,
            },
            seed=args.seed,
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            device="auto",
        )
        if torch.cuda.is_available() and str(model.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        callbacks = CallbackList(
            [
                CheckpointCallback(
                    save_freq=max(CHECKPOINT_EVERY_STEPS // N_ENVS, 1),
                    save_path=str(checkpoint_dir),
                    name_prefix=f"ppo_obs_{OBSERVATION_MODE}",
                ),
                ScoreEvaluationCallback(
                    output_csv=output_dir / "eval_scores.csv",
                    eval_every=EVAL_EVERY_STEPS,
                    episodes=EVAL_EPISODES,
                    verbose=1,
                ),
                memory_callback,
            ]
        )

        started_at = time.perf_counter()
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            tb_log_name=f"ppo_obs_{OBSERVATION_MODE}_seed_{args.seed}",
        )
        elapsed = time.perf_counter() - started_at
        model.save(str(output_dir / "final_model"))

        gpu_peak_mb = None
        if torch.cuda.is_available() and str(model.device).startswith("cuda"):
            gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024**2)

        summary = {
            "observation_mode": OBSERVATION_MODE,
            "observation_dimension": spec["expected_dim"],
            "observation_definition": (
                "9 hand-engineered state features from one frame"
                if OBSERVATION_MODE == "9d"
                else "the same 9 state features stacked across 4 frames"
                if OBSERVATION_MODE == "36d"
                else "raw FlappyBird-v0 Lidar observation"
            ),
            "use_lidar": spec["use_lidar"],
            "frame_stack": spec["frame_stack"],
            "reward": {"per_step": 0.05, "pipe_pass": 20.0, "episode_end": -15.0},
            "seed": args.seed,
            "n_envs": N_ENVS,
            "total_timesteps_requested": args.total_timesteps,
            "total_timesteps_actual": int(model.num_timesteps),
            "ppo": {
                "learning_rate": LEARNING_RATE,
                "n_steps": N_STEPS,
                "batch_size": BATCH_SIZE,
                "n_epochs": N_EPOCHS,
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "clip_range": CLIP_RANGE,
                "ent_coef": ENT_COEF,
                "vf_coef": VF_COEF,
                "max_grad_norm": MAX_GRAD_NORM,
                "hidden_layers": HIDDEN_LAYERS,
                "activation": "Tanh",
            },
            "evaluation": {
                "episodes_per_check": EVAL_EPISODES,
                "every_timesteps": EVAL_EVERY_STEPS,
                "fixed_seed_base": EVAL_SEED_BASE,
            },
            "elapsed_wall_seconds_including_evaluation": elapsed,
            "environment_steps_per_wall_second": model.num_timesteps / elapsed if elapsed else None,
            "peak_process_rss_mb": memory_callback.peak_rss_mb,
            "peak_gpu_allocated_mb": gpu_peak_mb,
            "device": str(model.device),
            "system": {
                "platform": platform.platform(),
                "processor": platform.processor() or None,
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "gymnasium": package_version("gymnasium"),
                "stable_baselines3": package_version("stable-baselines3"),
                "flappy_bird_gymnasium": package_version("flappy-bird-gymnasium"),
            },
            "notes": [
                "三種 observation 除表徵外共用獎勵、PPO 超參數、環境數、訓練步數和評估 seeds。",
                "9D 是單幀人工特徵，36D 是同一組特徵的四幀堆疊，180D 是不同型態的 Lidar；不是單純巢狀資訊量實驗。",
                "RAM 峰值需安裝 psutil 才會記錄；硬體比較必須在同一台機器及相同執行環境進行。",
            ],
        }
        (output_dir / "training_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"訓練完成：{output_dir}")
        print(f"步數={model.num_timesteps:,}；耗時={elapsed:.1f}s；裝置={model.device}")
    finally:
        train_env.close()
        gc.collect()


if __name__ == "__main__":
    main()
