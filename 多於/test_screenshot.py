import os
import sys
import cv2
import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# 从图片读取并测试 OpenCV 检测
image_path = 'game_screenshot.png'
if not os.path.exists(image_path):
    image_path = 'auto_game_screenshot.png'

if not os.path.exists(image_path):
    print("Error: game_screenshot.png and auto_game_screenshot.png not found")
    exit(1)

# 读取图片
image = cv2.imread(image_path)
if image is None:
    print(f"Error: Could not read {image_path}")
    exit(1)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, w = image.shape[:2]

print(f"图片尺寸: {w}x{h}")

# HSV 阈值 (来自 CNN test gym mss.py)
BIRD_LOWER = np.array([15, 80, 80])
BIRD_UPPER = np.array([30, 200, 200])

PIPE_LOWER = np.array([0, 0, 80])
PIPE_UPPER = np.array([180, 30, 150])

# 鸟检测
bird_mask = cv2.inRange(hsv, BIRD_LOWER, BIRD_UPPER)
bird_contours, _ = cv2.findContours(bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"\n=== 鸟检测 ===")
print(f"检测到 {len(bird_contours)} 个黄色物体")

bird_x, bird_y = None, None
if len(bird_contours) > 0:
    # 找最大的黄色物体（鸟）
    largest_bird = max(bird_contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_bird)
    M = cv2.moments(largest_bird)
    if M['m00'] > 0:
        bird_x = int(M['m10'] / M['m00'])
        bird_y = int(M['m01'] / M['m00'])
        print(f"鸟位置: ({bird_x}, {bird_y}), 面积: {area}")

# 管道检测
pipe_mask = cv2.inRange(hsv, PIPE_LOWER, PIPE_UPPER)
pipe_contours, _ = cv2.findContours(pipe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"\n=== 管道检测 ===")
print(f"检测到 {len(pipe_contours)} 个灰色物体")

pipe_list = []
for cnt in pipe_contours:
    area = cv2.contourArea(cnt)
    if area > 100:  # 最小面积
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
            pipe_list.append({
                'x': cx, 'y': cy, 'area': area,
                'rect': (x, y, w_cnt, h_cnt),
                'top': y, 'bottom': y + h_cnt
            })
            print(f"管道: x={cx}, y={cy}, 面积={area}, 高度={h_cnt}")

# 分析管道
print(f"\n=== 管道分析 ===")
if len(pipe_list) >= 2:
    # 按 x 坐标排序
    pipe_list.sort(key=lambda p: p['x'])
    p1 = pipe_list[0]
    p2 = pipe_list[1]
    print(f"P1 (左管道): x={p1['x']}")
    print(f"P2 (右管道): x={p2['x']}")
elif len(pipe_list) == 1:
    print(f"只检测到 1 个管道")
else:
    print(f"没有检测到管道！")

# 绘制检测结果
result = image.copy()

# 绘制鸟
if bird_x is not None and bird_y is not None:
    cv2.circle(result, (bird_x, bird_y), 10, (0, 255, 255), 2)
    cv2.putText(result, f"Bird ({bird_x}, {bird_y})", (bird_x - 40, bird_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

# 绘制管道
colors = [(255, 0, 0), (0, 0, 255)]
for i, pipe in enumerate(pipe_list):
    x, y, w_cnt, h_cnt = pipe['rect']
    cv2.rectangle(result, (x, y), (x + w_cnt, y + h_cnt), colors[i % len(colors)], 2)
    cv2.putText(result, f"P{i+1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 1)

# 保存结果
cv2.imwrite("detection_result.png", result)
print("结果已保存为 detection_result.png")
