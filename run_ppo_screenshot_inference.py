import argparse
import ctypes
import importlib
import time
import numpy as np
import cv2
import mss
from collections import deque
from ctypes import wintypes
from stable_baselines3 import PPO

# --- 關鍵對齊參數 ---
TARGET_FRAME_WIDTH = 500
TARGET_FRAME_HEIGHT = 800
STACK_SIZE = 4
FEATURE_DIM = 9
# 速度放大係數 (根據 gymnasium 的數值分佈估算，必要時可微調此值)
VELOCITY_SCALE = 15.0  

# HSV 閾值 (需根據你的螢幕顯示效果微調)
BIRD_LOWER = np.array([20, 100, 100])
BIRD_UPPER = np.array([40, 255, 255])
PIPE_LOWER = np.array([35, 50, 50])
PIPE_UPPER = np.array([85, 255, 255])

class FeatureBuilder:
    def __init__(self, stack_size=4):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)
        self.y_history = deque(maxlen=5)
        self.last_obs = None
        self.kernel = np.ones((5, 5), np.uint8)

    def extract_features9(self, frame_bgra):
        h, w = frame_bgra.shape[:2]
        # 1. 遮蔽地板 (底部 15%)，避免綠色地板干擾水管偵測
        frame_bgra[int(h * 0.85):, :] = 0
        
        bgr = cv2.cvtColor(frame_bgra, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # 2. 偵測小鳥
        bird_mask = cv2.inRange(hsv, BIRD_LOWER, BIRD_UPPER)
        bird_cnts, _ = cv2.findContours(bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not bird_cnts:
            return None
        
        bird_cnt = max(bird_cnts, key=cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(bird_cnt)
        bird_center_y = (by + bh / 2.0) / h
        bird_center_x = (bx + bw / 2.0) / w

        # 3. 計算速度 (並對齊 Gym 的量級)
        self.y_history.append(bird_center_y)
        if len(self.y_history) >= 2:
            # 這裡乘以 VELOCITY_SCALE 是為了讓視覺版的速度與訓練時的 obs[10] 接近
            bird_vel = (self.y_history[-1] - self.y_history[-2]) * VELOCITY_SCALE
        else:
            bird_vel = 0.0

        # 4. 偵測水管
        pipe_mask = cv2.inRange(hsv, PIPE_LOWER, PIPE_UPPER)
        pipe_cnts, _ = cv2.findContours(pipe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pipes = []
        for c in pipe_cnts:
            if cv2.contourArea(c) < 500: continue
            px, py, pw, ph = cv2.boundingRect(c)
            pipes.append({'x': px/w, 'y': py/h, 'w': pw/w, 'h': ph/h, 'bottom': (py+ph)/h})

        # 5. 尋找目標水管 (target_x > bird_center_x)
        # 如果畫面上沒水管，給予模擬環境重置時的預設值
        target_x, target_top, target_bottom = 1.0, 0.4, 0.6 
        
        if pipes:
            # 簡化邏輯：抓取鳥右側最近的上下管邊界
            right_pipes = [p for p in pipes if p['x'] > bird_center_x - 0.05]
            if right_pipes:
                # 依 X 軸排序找最近的
                right_pipes.sort(key=lambda p: p['x'])
                target_x = right_pipes[0]['x']
                # 這裡假設最近的有兩塊（上管跟下管），取中間縫隙
                # 實際應用中建議加入更穩健的 Gap 計算
                # ... (簡化示意)
                target_top = 0.4
                target_bottom = 0.6

        gap_center = (target_top + target_bottom) / 2.0

        # 6. 組合成 9 維特徵 (順序必須與訓練時完全一致)
        feat = np.array([
            bird_center_y,          # bird_y
            bird_vel,               # bird_vel
            target_x,               # target_x
            bird_center_y - gap_center, # bird_minus_gap_center
            target_top,             # target_top
            target_bottom,          # target_bottom
            bird_vel * target_x,    # bird_vel_mul_target_x
            1.0 if bird_vel < -0.5 else 0.0, # falling_flag
            0.0                     # p2_x_if_p1_behind (推論時可先給 0)
        ], dtype=np.float32)
        
        return feat

    def get_obs(self, frame_bgra):
        feat = self.extract_features9(frame_bgra)
        if feat is None:
            return self.last_obs if self.last_obs is not None else None
        
        if len(self.frames) == 0:
            for _ in range(self.stack_size): self.frames.append(feat)
        else:
            self.frames.append(feat)
            
        self.last_obs = np.concatenate(list(self.frames), axis=0)
        return self.last_obs

def main():
    model = PPO.load("models/best/best_model") # 載入你訓練好的模型
    extractor = FeatureBuilder(stack_size=4)
    
    with mss.mss() as sct:
        # 請根據你 Pygame 視窗在螢幕上的位置設定 monitor
        monitor = {"top": 100, "left": 100, "width": TARGET_FRAME_WIDTH, "height": TARGET_FRAME_HEIGHT}
        
        print("🤖 AI 代理啟動... 按 'q' 退出")
        while True:
            start_time = time.time()
            
            # 擷取畫面
            img = np.array(sct.grab(monitor))
            obs = extractor.get_obs(img)
            
            if obs is not None:
                action, _ = model.predict(obs, deterministic=True)
                # 這裡你可以整合 pyautogui 或 pydirectinput 來模擬按鍵
                if action == 1:
                    # pydirectinput.press('space') # 舉例
                    pass
                
                # 除錯資訊 (這對解決「往上飛」至關重要)
                # print(f"BirdY: {obs[0]:.2f} | Vel: {obs[1]:.2f} | Action: {action}")

            # 控制 FPS
            elapsed = time.time() - start_time
            time.sleep(max(0, 1/60 - elapsed))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()