import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np

def analyze_obs():
    env = gym.make("FlappyBird-v0", render_mode=None, use_lidar=False)
    
    print("=" * 60)
    print("自動診斷: 通過自動跳躍來識別各維度含義")
    print("=" * 60)
    
    # 測試 1: 不跳不跳躍，觀察自然掉落
    print("\n【測試 1：不跳躍，觀察 30 步】")
    obs, _ = env.reset(seed=42)
    print(f"初始狀態：")
    for i in range(len(obs)):
        print(f"  obs[{i:2d}]: {obs[i]:+.4f}")
    
    no_jump_history = [obs.copy()]
    for step in range(30):
        obs, _, done, _, _ = env.step(0)  # 不跳躍
        no_jump_history.append(obs.copy())
        if done:
            print(f"  死亡於 step {step}")
            break
    
    # 測試 2: 在特定步驟跳躍，觀察變化
    print("\n【測試 2：在 step 10, 20, 30 跳躍】")
    obs, _ = env.reset(seed=42)
    jump_history = [obs.copy()]
    jump_steps = [10, 20, 30]  # 在這些步驟跳躍
    
    for step in range(40):
        action = 1 if step in jump_steps else 0
        prev_obs = obs.copy()
        obs, _, done, _, _ = env.step(action)
        
        if action == 1:
            print(f"\n  >>> Step {step}: 跳躍!")
            print(f"      跳躍前: ", end="")
            for i in [0, 1, 2, 10, 11]:  # 重點觀察這些索引
                if i < len(prev_obs):
                    print(f"[{i}]={prev_obs[i]:+.2f} ", end="")
            print(f"\n      跳躍後: ", end="")
            for i in [0, 1, 2, 10, 11]:
                if i < len(obs):
                    print(f"[{i}]={obs[i]:+.2f} ", end="")
            print()
        
        jump_history.append(obs.copy())
        if done:
            print(f"  死亡於 step {step}")
            break
    
    env.close()
    
    # 分析: 找出跳躍時變化最大的維度
    print("\n" + "=" * 60)
    print("【分析結果】")

    
    print("=" * 60)

    print(f"{'維度':>6} | {'不跳躍變化':>12} | {'跳躍時變化':>12} | {'推測含義':>20}")
    print("-" * 60)
    
    for i in range(len(obs)):
        # 不跳躍時的變化 (step 5-10)
        no_jump_change = abs(no_jump_history[10][i] - no_jump_history[5][i]) if len(no_jump_history) > 10 else 0
        
        # 跳躍時的變化 (step 10 跳躍前後)
        if len(jump_history) > 11:
            jump_change = abs(jump_history[11][i] - jump_history[10][i])
        else:
            jump_change = 0
        
        meaning = ""
        if i == 0:
            meaning = "bird_y (高度)"
        elif jump_change > 0.3:  # 跳躍時劇烈變化
            meaning = "*** bird_vel (速度) ***"
        elif no_jump_history[0][i] > 0.9 and no_jump_history[5][i] < no_jump_history[0][i]:
            meaning = "next_pipe_x (距離)"
        elif abs(no_jump_history[0][i] - 0.5) < 0.1:
            meaning = "可能是 bias/常數"
        
        print(f"[{i:2d}]   | {no_jump_change:>12.4f} | {jump_change:>12.4f} | {meaning:>20}")

    # 關鍵結論
    print("\n" + "=" * 60)
    print("【關鍵發現】")
    print("=" * 60)
    
    # 找出跳躍時變化最大的索引
    max_jump_idx = 0
    max_jump_val = 0
    for i in range(len(obs)):
        if len(jump_history) > 11:
            change = abs(jump_history[11][i] - jump_history[10][i])
            if change > max_jump_val:
                max_jump_val = change
                max_jump_idx = i
    
    print(f"跳躍時變化最大的維度是: [{max_jump_idx}]")
    print(f"變化量: {max_jump_val:.4f}")
    print(f"\n這應該就是 bird_vel (垂直速度)")
    print(f"建議的 9 維特徵應該使用:")
    print(f"  - bird_y: [0] (= {jump_history[0][0]:.4f})")
    print(f"  - bird_vel: [{max_jump_idx}] (= {jump_history[0][max_jump_idx]:.4f})")
    print(f"  - next_pipe_x: [3] (如果值會遞減)")
    
    return max_jump_idx  # 返回 velocity 的索引

if __name__ == "__main__":
    vel_idx = analyze_obs()
    print(f"\n請記住: velocity 的索引是 [{vel_idx}]")