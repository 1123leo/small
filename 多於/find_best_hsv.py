"""
精细调整 HSV 范围 - 目标是每帧只检测到 1 只鸟和 1-2 对管道
"""
import cv2
import numpy as np

image = cv2.imread('game_screenshot.png')
if image is None:
    print("Error: Could not read game_screenshot.png")
    exit(1)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 更精细的参数测试
test_cases = [
    ("Test1", (15, 80, 80), (30, 200, 200), (0, 0, 80), (180, 30, 150)),
    ("Test2", (12, 60, 60), (35, 220, 220), (0, 0, 70), (180, 40, 170)),
    ("Test3", (18, 100, 100), (28, 180, 180), (0, 0, 90), (180, 25, 160)),
    ("Test4", (16, 90, 90), (32, 200, 200), (0, 0, 75), (180, 35, 165)),
]

best_score = float('inf')
best_params = None

for name, bird_lower, bird_upper, pipe_lower, pipe_upper in test_cases:
    # 黄色（鸟）掩膜
    bird_mask = cv2.inRange(hsv, bird_lower, bird_upper)
    bird_contours, _ = cv2.findContours(bird_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bird_valid = [c for c in bird_contours if cv2.contourArea(c) > 20]
    
    # 灰色（管道）掩膜
    pipe_mask = cv2.inRange(hsv, pipe_lower, pipe_upper)
    pipe_contours, _ = cv2.findContours(pipe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pipe_valid = [c for c in pipe_contours if cv2.contourArea(c) > 50]
    
    # 评分：越接近 1 只鸟和 2 对管道越好
    bird_score = abs(len(bird_valid) - 1)
    pipe_score = abs(len(pipe_valid) - 2) if len(pipe_valid) > 0 else 10
    total_score = bird_score + pipe_score
    
    print(f"{name}:")
    print(f"  鸟: {len(bird_valid)} (评分: {bird_score})")
    print(f"  管: {len(pipe_valid)} (评分: {pipe_score})")
    print(f"  总评分: {total_score}")
    print(f"  BIRD: {bird_lower} ~ {bird_upper}")
    print(f"  PIPE: {pipe_lower} ~ {pipe_upper}")
    print()
    
    if total_score < best_score:
        best_score = total_score
        best_params = (name, bird_lower, bird_upper, pipe_lower, pipe_upper)

print("=" * 50)
print(f"【最佳参数】{best_params[0]}")
print(f"BIRD_LOWER = {best_params[1]}")
print(f"BIRD_UPPER = {best_params[2]}")
print(f"PIPE_LOWER = {best_params[3]}")
print(f"PIPE_UPPER = {best_params[4]}")
