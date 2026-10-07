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
            bird_y, bird_vel, target_x, bird_y - gap_center,
            target_top, target_bot, bird_vel * target_x,
            1.0 if bird_vel < -0.5 else 0.0,
            p2_x if p1_x < 0.0 else 0
        ], dtype=np.float32)

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)

# =========================================================
# 詳細調試器
# =========================================================
class DebugRecorder:
    def __init__(self):
        self.events = []
        self.feature_history = []
        
    def log_step(self, step, bird_y, bird_vel, p1_x, target_x, gap_center, 
                 action, predicted_prob, reward, obs_features):
        self.events.append({
            'step': step,
            'bird_y': bird_y,
            'bird_vel': bird_vel,
            'p1_x': p1_x,
            'target_x': target_x,
            'gap_center': gap_center,
            'action': action,
            'predicted_prob': predicted_prob,
            'reward': reward,
        })
        self.feature_history.append(obs_features.copy())
    
    def analyze_failure(self, final_y, final_vel, gap_top, gap_bottom):
        """分析死亡原因"""
        if final_y < gap_top:
            return "撞上管道（鸟在管道上方）"
        elif final_y > gap_bottom:
            return "撞下管道（鸟在管道下方）"
        else:
            return "未知原因"
    
    def print_last_n_steps(self, n=10):
        """打印最后N步的决策过程"""
        print(f"\n最后 {n} 步决策过程：")
        print("步数 | 鸟高度 | 鸟速度 | P1距离 | 目标距 | 间隙中心 | 行动 | 概率 | 奖励")
        print("-" * 100)
        for event in self.events[-n:]:
            gap_center = event['gap_center']
            dist_to_gap = abs(event['bird_y'] - gap_center)
            print(f"{event['step']:4d} | {event['bird_y']:6.3f} | {event['bird_vel']:6.3f} | "
                  f"{event['p1_x']:6.3f} | {event['target_x']:6.3f} | "
                  f"{event['gap_center']:6.3f} | {'跳' if event['action'] else '不'} | "
                  f"{event['predicted_prob']:.3f} | {event['reward']:+6.2f}")

# =========================================================
# 主程序
# =========================================================
model = PPO.load("ppo_flappybird/policy")
test_env_raw = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
test_env = SmoothFlappyEnv(test_env_raw, n_stack=4)

print("=" * 100)
print("模型过水管调查 - 详细分析")
print("=" * 100)

scores = []
failures = {}

for ep in range(5):
    print(f"\n【第 {ep+1} 集】" + "=" * 80)
    obs, _ = test_env.reset(seed=ep + 1019)
    done = False
    score = 0
    step_count = 0
    recorder = DebugRecorder()
    
    while not done and step_count < 10000000:
        # 获取模型预测
        action, _states = model.predict(obs, deterministic=True)
        
        # 获取模型的动作概率
        with torch.no_grad():
            action_prob = model.policy.get_distribution(obs.reshape(1, -1)).distribution.probs
        
        # 执行步骤
        next_obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated
        
        # 提取当前状态信息
        raw_obs = test_env.env.get_observation()
        bird_y = raw_obs[9]
        bird_vel = raw_obs[10]
        p1_x = raw_obs[3]
        p2_x = raw_obs[6]
        target_x = p2_x if p1_x < -0.1 else p1_x
        p1_top, p1_bot = raw_obs[4], raw_obs[5]
        p2_top, p2_bot = raw_obs[7], raw_obs[8]
        target_top = p2_top if p1_x < -0.1 else p1_top
        target_bot = p2_bot if p1_x < -0.1 else p1_bot
        gap_center = (target_top + target_bot) / 2.0
        
        # 记录
        recorder.log_step(step_count, bird_y, bird_vel, p1_x, target_x, gap_center,
                         int(action), float(action_prob[0, action]), float(reward), obs)
        
        obs = next_obs
        step_count += 1
        
        if done:
            final_score = info.get('score', 0)
            scores.append(final_score)
            
            # 分析失败原因
            failure_reason = recorder.analyze_failure(bird_y, bird_vel, target_top, target_bot)
            failures[final_score] = failure_reason
            
            print(f"【结果】得分: {final_score} | 存活步数: {step_count} | 原因: {failure_reason}")
            
            # 如果得分 <= 2，打印详细信息
            if final_score <= 2:
                print(f"\n【警告】得分不理想，可能的问题：")
                print(f"  - 鸟最终高度: {bird_y:.4f}")
                print(f"  - 鸟最终速度: {bird_vel:.4f}")
                print(f"  - 管道间隙: {target_top:.4f} - {target_bot:.4f}")
                print(f"  - 最后10步决策:")
                recorder.print_last_n_steps(10)

test_env.close()

print("\n" + "=" * 100)
print("【汇总统计】")
print("=" * 100)
print(f"平均得分: {np.mean(scores):.2f} 根管子")
print(f"最高得分: {max(scores)} 根")
print(f"最低得分: {min(scores)} 根")
print(f"标准差: {np.std(scores):.2f}")

print("\n【失败原因分布】")
reason_counts = {}
for reason in failures.values():
    reason_counts[reason] = reason_counts.get(reason, 0) + 1
for reason, count in reason_counts.items():
    print(f"  {reason}: {count} 次")

print("\n【诊断建议】")
if max(scores) <= 3:
    print("⚠️  模型无法稳定通过管道")
    print("   可能原因：")
    print("   1. falling_flag 始终为0（速度阈值过高）")
    print("   2. 学习率/奖励设置不优")
    print("   3. 模型训练不充分")
    print("   建议：")
    print("   - 降低 bird_vel 的 falling_flag 阈值（从 -0.5 改为 -0.2）")
    print("   - 增加奖励信号强度")
    print("   - 用更多步数重新训练")
elif max(scores) <= 10:
    print("⚠️  模型偶尔能通过，但不稳定")
    print("   建议：微调参数并增加训练步数")
else:
    print("✓ 模型表现不错！")
