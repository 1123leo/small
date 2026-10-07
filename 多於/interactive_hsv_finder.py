"""
交互式 HSV 范围查找工具
使用滑块调整，查看实时掩膜效果
"""
import cv2
import numpy as np

image = cv2.imread('game_screenshot.png')
if image is None:
    print("Error: Could not read game_screenshot.png")
    exit(1)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print(f"图片尺寸: {image.shape}")
print("使用滑块调整 HSV 范围，查看掩膜中的检测效果")
print("黄色（鸟）通常应该清晰孤立地出现在左侧")
print("灰色（管道）通常应该在中右区域")
print()

def update_yellow(val):
    h_low = cv2.getTrackbarPos('H_low_yellow', 'Yellow Control')
    h_high = cv2.getTrackbarPos('H_high_yellow', 'Yellow Control')
    s_low = cv2.getTrackbarPos('S_low_yellow', 'Yellow Control')
    s_high = cv2.getTrackbarPos('S_high_yellow', 'Yellow Control')
    v_low = cv2.getTrackbarPos('V_low_yellow', 'Yellow Control')
    v_high = cv2.getTrackbarPos('V_high_yellow', 'Yellow Control')
    
    lower = (h_low, s_low, v_low)
    upper = (h_high, s_high, v_high)
    mask = cv2.inRange(hsv, lower, upper)
    
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('Yellow Mask', mask)
    cv2.imshow('Yellow Result', result)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"\r黄色: H({h_low}-{h_high}) S({s_low}-{s_high}) V({v_low}-{v_high}) | 物体数: {len(contours)}", end='')

def update_gray(val):
    h_low = cv2.getTrackbarPos('H_low_gray', 'Gray Control')
    h_high = cv2.getTrackbarPos('H_high_gray', 'Gray Control')
    s_low = cv2.getTrackbarPos('S_low_gray', 'Gray Control')
    s_high = cv2.getTrackbarPos('S_high_gray', 'Gray Control')
    v_low = cv2.getTrackbarPos('V_low_gray', 'Gray Control')
    v_high = cv2.getTrackbarPos('V_high_gray', 'Gray Control')
    
    lower = (h_low, s_low, v_low)
    upper = (h_high, s_high, v_high)
    mask = cv2.inRange(hsv, lower, upper)
    
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('Gray Mask', mask)
    cv2.imshow('Gray Result', result)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"\r灰色: H({h_low}-{h_high}) S({s_low}-{s_high}) V({v_low}-{v_high}) | 物体数: {len(contours)}", end='')

# 创建窗口
cv2.namedWindow('Yellow Control', cv2.WINDOW_NORMAL)
cv2.namedWindow('Gray Control', cv2.WINDOW_NORMAL)

# 黄色滑块 (初始值根据经验设置)
cv2.createTrackbar('H_low_yellow', 'Yellow Control', 10, 180, update_yellow)
cv2.createTrackbar('H_high_yellow', 'Yellow Control', 40, 180, update_yellow)
cv2.createTrackbar('S_low_yellow', 'Yellow Control', 20, 255, update_yellow)
cv2.createTrackbar('S_high_yellow', 'Yellow Control', 220, 255, update_yellow)
cv2.createTrackbar('V_low_yellow', 'Yellow Control', 30, 255, update_yellow)
cv2.createTrackbar('V_high_yellow', 'Yellow Control', 230, 255, update_yellow)

# 灰色滑块 (S 很低，V 中等)
cv2.createTrackbar('H_low_gray', 'Gray Control', 0, 180, update_gray)
cv2.createTrackbar('H_high_gray', 'Gray Control', 180, 180, update_gray)
cv2.createTrackbar('S_low_gray', 'Gray Control', 0, 255, update_gray)
cv2.createTrackbar('S_high_gray', 'Gray Control', 100, 255, update_gray)
cv2.createTrackbar('V_low_gray', 'Gray Control', 50, 255, update_gray)
cv2.createTrackbar('V_high_gray', 'Gray Control', 190, 255, update_gray)

# 首次更新
update_yellow(0)
update_gray(0)

print("\n按 'q' 或 'ESC' 退出。查看窗口中的掩膜，调整滑块直到只看到目标颜色。")
print("===========================================")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# 获取最终参数
print("\n\n=== 最终参数 ===")
h_low_y = cv2.getTrackbarPos('H_low_yellow', 'Yellow Control')
h_high_y = cv2.getTrackbarPos('H_high_yellow', 'Yellow Control')
s_low_y = cv2.getTrackbarPos('S_low_yellow', 'Yellow Control')
s_high_y = cv2.getTrackbarPos('S_high_yellow', 'Yellow Control')
v_low_y = cv2.getTrackbarPos('V_low_yellow', 'Yellow Control')
v_high_y = cv2.getTrackbarPos('V_high_yellow', 'Yellow Control')

h_low_g = cv2.getTrackbarPos('H_low_gray', 'Gray Control')
h_high_g = cv2.getTrackbarPos('H_high_gray', 'Gray Control')
s_low_g = cv2.getTrackbarPos('S_low_gray', 'Gray Control')
s_high_g = cv2.getTrackbarPos('S_high_gray', 'Gray Control')
v_low_g = cv2.getTrackbarPos('V_low_gray', 'Gray Control')
v_high_g = cv2.getTrackbarPos('V_high_gray', 'Gray Control')

print(f"BIRD_LOWER = ({h_low_y}, {s_low_y}, {v_low_y})")
print(f"BIRD_UPPER = ({h_high_y}, {s_high_y}, {v_high_y})")
print(f"PIPE_LOWER = ({h_low_g}, {s_low_g}, {v_low_g})")
print(f"PIPE_UPPER = ({h_high_g}, {s_high_g}, {v_high_g})")

cv2.destroyAllWindows()
