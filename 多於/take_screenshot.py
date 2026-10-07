import mss
import cv2
import numpy as np

print("截取当前屏幕...")
with mss.mss() as sct:
    monitor = sct.monitors[1]  # 主屏幕
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    cv2.imwrite('real_game_screenshot.png', frame)
    print(f"✓ 截图保存为 real_game_screenshot.png ({frame.shape[1]}x{frame.shape[0]})")
