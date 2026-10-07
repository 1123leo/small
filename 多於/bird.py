import mss
import cv2
import numpy as np
with mss.mss() as sct:
    # 先截全螢幕
    full = np.array(sct.grab(sct.monitors[1]))
    cv2.imshow("screen", cv2.resize(full, (960, 540)))
    cv2.waitKey(0)
    
    # 或用滑鼠點擊找座標
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked: x={x}, y={y}")
    
    cv2.namedWindow("screen")
    cv2.setMouseCallback("screen", mouse_callback)
    cv2.imshow("screen", cv2.resize(full, (960, 540)))
    cv2.waitKey(0)