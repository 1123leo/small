import cv2
import mss
import numpy as np

REGION = {"top": 60, "left": 640, "width": 640, "height": 860}

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"點擊座標: x={x}, y={y}")
        # 在視窗標題顯示座標，方便你記下來
        cv2.setWindowTitle("Check Ground", f"Ground Y = {y}")

with mss.mss() as sct:
    cv2.namedWindow("Check Ground")
    cv2.setMouseCallback("Check Ground", mouse_callback)
    
    print("請點擊遊戲中的地面位置...")
    while True:
        img = np.array(sct.grab(REGION))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        cv2.imshow("Check Ground", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()