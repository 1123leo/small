import mss
import cv2
import numpy as np

region = {'top': 50, 'left': 100, 'width': 500, 'height': 800}
with mss.mss() as sct:
    screenshot = sct.grab(region)
    img = np.array(screenshot)
    cv2.imwrite("game_screenshot.png", img)
img = cv2.imread("game_screenshot.png")
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
print("點擊鳥或管道，獲取HSV值")
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"HSV at ({x}, {y}): {hsv[y, x]}")
cv2.imshow("Image", img)
cv2.setMouseCallback("Image", mouse_callback)
cv2.waitKey(0)
cv2.destroyAllWindows()

