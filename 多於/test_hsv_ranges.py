"""
可视化 OpenCV 检测结果 - 原图 vs 掩膜 vs 结果
"""
import os
import sys
import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

IMAGE_PATH = 'game_screenshot.png'
if not os.path.exists(IMAGE_PATH):
    IMAGE_PATH = 'auto_game_screenshot.png'

image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Error: Could not read {IMAGE_PATH}")
    exit(1)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print(f"图片尺寸: {image.shape} (高, 宽, 通道)")
print("正在测试不同的 HSV 范围...\n")

# 测试不同的范围
test_cases = [
    # (name, bird_lower, bird_upper, pipe_lower, pipe_upper)
    ("Conservative", (15, 50, 50), (35, 255, 255), (0, 0, 100), (180, 50, 200)),
    ("Moderate", (10, 30, 40), (45, 250, 250), (0, 0, 50), (180, 80, 200)),
    ("Aggressive", (5, 20, 20), (50, 255, 255), (0, 0, 40), (180, 100, 220)),
]

for name, bird_lower, bird_upper, pipe_lower, pipe_upper in test_cases:
    print(f"=== {name} ===")
    print(f"BIRD:  {bird_lower} ~ {bird_upper}")
    print(f"PIPE:  {pipe_lower} ~ {pipe_upper}")
    
    # 黄色（鸟）掩膜
    bird_mask = cv2.inRange(hsv, bird_lower, bird_upper)
    bird_contours, _ = cv2.findContours(bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bird_valid = [c for c in bird_contours if cv2.contourArea(c) > 20]
    
    # 灰色（管道）掩膜
    pipe_mask = cv2.inRange(hsv, pipe_lower, pipe_upper)
    pipe_contours, _ = cv2.findContours(pipe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pipe_valid = [c for c in pipe_contours if cv2.contourArea(c) > 50]
    
    print(f"检测到鸟: {len(bird_valid)} 个 (总: {len(bird_contours)})")
    print(f"检测到管: {len(pipe_valid)} 个 (总: {len(pipe_contours)})")
    print()
    
    # 创建可视化
    vis = image.copy()
    
    # 画出鸟
    for contour in bird_valid:
        cv2.drawContours(vis, [contour], 0, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # 画出管子
    for contour in pipe_valid:
        cv2.drawContours(vis, [contour], 0, (255, 0, 0), 2)
    
    # 保存结果
    output_file = f'hsv_test_{name.lower()}.png'
    cv2.imwrite(output_file, vis)
    print(f"已保存: {output_file}\n")

print("已生成 3 个测试结果图片。请从最接近的那个中选择参数。")
