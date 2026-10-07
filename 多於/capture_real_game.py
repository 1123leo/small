import mss
import cv2
import numpy as np
import time

# 截取真实游戏画面
print("准备截游戏画面，5秒后开始...")
time.sleep(5)

with mss.mss() as sct:
    monitor = sct.monitors[1]  # 主屏幕
    screenshot = sct.grab(monitor)
    
    # 转换为 OpenCV 格式
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    
    # 保存
    cv2.imwrite('real_game_screenshot.png', frame)
    print(f"✓ 截图保存为 real_game_screenshot.png")
    print(f"  分辨率: {frame.shape[1]}x{frame.shape[0]}")
    
    # 显示预览
    cv2.imshow('Game Screenshot', frame)
    print("按任意键关闭...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
