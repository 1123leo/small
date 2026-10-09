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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# =========================================================
# 關鍵創新：FrameStack + 平滑獎勵塑形
# =========================================================
class SmoothFlappyEnv(gym.Wrapper):
    """
    目標：500+ 根管子
    策略：
    1. 9維特徵（正確velocity）+ 4幀堆疊 = 36維輸入
    2. 獎勵塑形：懲罰劇烈動作，獎勵平滑飛行
    3. 鼓勵保持在管口中心線附近
    """
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        
        self.n_stack = n_stack
        self.frames = deque(maxlen=n_stack)
        self.last_score = 0
        self.last_action = 0  # 用於懲罰連續跳躍   # 用於計算垂直速度
        
        # 觀測空間：9維 × 4幀 = 36維
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(9 * n_stack,), dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_score = 0
        self.last_action = 0
        self.last_y = obs[9]  # bird_y
        
        # 用初始狀態填充所有幀
        features = self._extract(obs)
        for _ in range(self.n_stack):
            self.frames.append(features)
            
        return self._get_obs(), info
    
    def step(self, action):
        obs, _, done, trun, info = self.env.step(action)
        
        # ===== 關鍵：平滑獎勵塑形 =====
        bird_y = obs[9]
        bird_vel = obs[10]
        
        # 1. 基礎存活獎勵（降低，避免「膽小鬼」策略）
        reward = 0.05
        
        # 2. 過管大獎勵
        score = info.get('score', 0)
        if score > self.last_score:
            reward +=20.0  # 大幅提高過管獎勵
            self.last_score = score
        
        # 3. 死亡懲罰
        if done:
            reward = -15.0
        
        # 4. 🎯 關鍵創新：「管口中心線」獎勵（鼓勵對齊）
        # 提取下一根管子的上下位置
        p1_x, p1_top, p1_bot = obs[3], obs[4], obs[5]
        if p1_x < -0.1:  # 已過管，看下下根
            gap_center = (obs[7] + obs[8]) / 2.0
        else:
            gap_center = (p1_top + p1_bot) / 2.0
        
        # 與管口中心線的距離（越小越好）
        dist_to_center = abs(bird_y - gap_center)
        # 獎勵靠近中心（高斯獎勵）
        center_bonus = 0.1 * np.exp(-dist_to_center * 5)  # 在中心時 ≈0.1，偏離時快速衰減
        reward += center_bonus
        
        # 5. 🎯 關鍵創新：「平滑飛行」懲罰（懲罰劇烈跳躍）
        # 如果上一幀跳了，這一幀也跳，懲罰（避免連續跳導致飛太高）
        # if action == 1 and self.last_action == 1:
        #     reward -= 0.5  # 連續跳躍懲罰
        
        # 6. 速度懲罰：懲罰過快下降（除非必要）
        if bird_vel < -0.6 and bird_y > gap_center:  # 快速下降且已經低於中心
            reward -= 0.1
        
        # 7. 「節能」獎勵：鼓勵不跳（滑翔）當位置合適時
        if action == 0 and abs(bird_y - gap_center) < 0.1:
            reward += 0.05  # 位置好且滑翔 = 獎勵
        
        self.last_action = action
        self.last_y = bird_y
        
        # 存儲幀
        self.frames.append(self._extract(obs))
        
        return self._get_obs(), reward, done, trun, info
    
    def _extract(self, obs):
        """提取單幀 9 維特徵"""
        bird_y = obs[9]
        bird_vel = obs[10]  # 正確索引
        p1_x, p1_top, p1_bot = obs[3], obs[4], obs[5]
        p2_x, p2_top, p2_bot = obs[6], obs[7], obs[8]
        
        # 智能管線選擇
        if p1_x < -0.1:
            target_x, target_top, target_bot = p2_x, p2_top, p2_bot
        else:
            target_x, target_top, target_bot = p1_x, p1_top, p1_bot
            
        gap_center = (target_top + target_bot) / 2.0
        
        return np.array([
            bird_y,
            bird_vel,
            target_x,
            bird_y - gap_center,      # 與中心偏差（關鍵！）
            target_top,
            target_bot,
            bird_vel * target_x,      # 預判項
            1.0 if bird_vel < -0.5 else 0.0,
            p2_x if p1_x < 0 else 0   # 下下根管距離
        ], dtype=np.float32)
    
    def _get_obs(self):
        """返回堆疊的 36 維觀測"""
        return np.concatenate(list(self.frames), axis=0)

# =========================================================
# 主程式：針對 500+ 優化的超參數
# =========================================================
if __name__ == "__main__":
    os.makedirs("./models/500plus/", exist_ok=True)
    
    def make_env():
        return SmoothFlappyEnv(
            gym.make("FlappyBird-v0", render_mode=None, use_lidar=False),
            n_stack=4
        )
    
    # 更多並行環境（16個），提高樣本多樣性
    train_env = make_vec_env(make_env, n_envs=8, seed=42)
    
    # 評估環境
    eval_env = DummyVecEnv([lambda: Monitor(make_env())])
    
    # 學習率調度：快速探索 → 穩定收斂
    # def lr_schedule(progress):
    # #     # 前 50% 快速學習，後 50% 微調
    #     if progress > 0.4:
    #         return 1e-3 * (progress * 2)  # 後期降低
    #     return 3e-4
    # def linear_schedule(initial_value, min_value=1e-5):
    #     def func(progress_remaining: float) -> float:
    #     # progress_remaining 從 1.0 降到 0.0
    #         return max(initial_value * progress_remaining, min_value)
    #     return func
    
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,  # 固定學習率，穩定訓練
        #learning_rate=lr_schedule,
        #learning_rate=linear_schedule(2e-3, min_value=1e-4), 
        n_steps=2048,           # 更長的軌跡（捕捉長期依賴）
        batch_size=1024,         # 大批量穩定更新
        gamma=0.995,            # 更重視長期回報（原本0.99）
        ent_coef=0.03,          # 適度探索
        clip_range=0.2,        # 適中限制
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log="./tb_logs/",
        policy_kwargs=dict(
            net_arch=[128, 128, 64],  # 更深網路處理36維時序
            activation_fn=torch.nn.ReLU
        ),
        n_epochs=5,             # 每批數據訓練更多輪
    )
    
    # 每 20K 步評估，保存最佳
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="./models/500plus/",
        eval_freq=20000,
        deterministic=True,
        render=False,
        n_eval_episodes=10      # 評估10輪取平均，更穩定
    )
    
    # 每 50K 步保存檢查點
    checkpoint_cb = CheckpointCallback(
        save_freq=50000,
        save_path="./models/500plus/",
        name_prefix="ppo_flappy"
    )
    
    
    model.learn(total_timesteps=60_0000, callback=[eval_cb, checkpoint_cb])
    model.save("ppo_500plus_final")
    
        # ========== 測試最佳模型（修正版）==========
    print("✅ 訓練完成！測試最佳模型...")
    
    from stable_baselines3 import PPO
    
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
