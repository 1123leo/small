from playwright.sync_api import sync_playwright
import numpy as np
from stable_baselines3 import PPO
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

model = PPO.load("best_model")  # 你的幾何模型

def get_game_state(page):
    """
    注入 JavaScript 抓取遊戲內部變數。
    注意：不同網站的 Flappy Bird 變數名不同，請在瀏覽器 F12 Console 確認。
    """
    return page.evaluate("""() => {
        // 以常見的網頁版 Flappy Bird (Phaser) 為例
        // 你可能需要根據實際網站調整這段
        try {
            // 有些版本會暴露到 window 物件
            const bird = game.bird || bird || {};
            const pipes = game.pipes || pipes || [];
            
            return {
                birdY: bird.y || 0,
                birdVel: bird.velocity || bird.body?.velocity?.y || 0,
                nextPipeX: pipes.length > 0 ? pipes[0].x : 500,
                nextPipeGapY: pipes.length > 0 ? pipes[0].gapCenter || pipes[0].y : 250,
                alive: true
            };
        } catch(e) {
            return null;
        }
    }""")

def state_to_obs(state):
    """轉成和你訓練時完全一致的 obs 格式"""
    return np.array([
        state['birdY'] / 512.0,
        state['birdVel'] / 10.0,
        state['nextPipeX'] / 288.0,
        state['nextPipeGapY'] / 512.0,
        # ... 補到和訓練時一樣的維度
    ], dtype=np.float32)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page(viewport={"width": 600, "height": 800})
    page.goto("https://flappybird.io/")  # 換成你的目標網站
    
    print("5 秒後啟動 AI，請先切換到瀏覽器視窗...")
    time.sleep(5)
    
    # 點擊畫面開始遊戲（座標請依實際調整）
    page.mouse.click(300, 400)
    time.sleep(1)
    
    while True:
        state = get_game_state(page)
        if state is None:
            time.sleep(0.05)
            continue
        
        obs = state_to_obs(state)
        action, _ = model.predict(obs, deterministic=True)
        
        if action == 1:
            while True:
                state = get_state(page)
                if state is None:
                    time.sleep(0.02)
                    continue
    
                obs = state_to_obs(state)
                action, _ = model.predict(obs, deterministic=True)
    
    # ===== 除錯輸出 =====
                print(f"obs={obs.round(2)}, action={action}, vel={state['birdVel']:.2f}, pipeX={state['nextPipeX']:.1f}")
    
        if action == 1:
            print(">>> 應該要跳！")
        page.keyboard.press("Space")
    
        time.sleep(0.016)# 控制頻率約 20 FPS 就夠了
