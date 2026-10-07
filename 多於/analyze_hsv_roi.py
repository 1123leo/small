"""
更精确的 HSV 范围分析 - 手动选择 ROI
"""
import cv2
import numpy as np

image = cv2.imread('game_screenshot.png')
if image is None:
    print("Error: Could not read game_screenshot.png")
    exit(1)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print("图片尺寸:", image.shape)
print()

# 手动指定感兴趣区域
# 根据 Flappy Bird 的典型布局：
# - 鸟通常在左上区域 (x: 30-100, y: 200-400)
# - 管道通常在中右区域

bird_roi = hsv[200:400, 30:100]  # y: 200-400, x: 30-100
pipe_roi = hsv[0:800, 200:500]   # 整个高度，x: 200-500

print("【鸟 ROI 分析】")
h_bird = bird_roi[:, :, 0]
s_bird = bird_roi[:, :, 1]
v_bird = bird_roi[:, :, 2]

# 找出明显的黄色（S > 50, V > 50）
yellow_mask = (s_bird > 50) & (v_bird > 50)
if yellow_mask.sum() > 0:
    h_yellow = h_bird[yellow_mask]
    s_yellow = s_bird[yellow_mask]
    v_yellow = v_bird[yellow_mask]
    print(f"黄色像素数: {len(h_yellow)}")
    print(f"H: {h_yellow.min()} - {h_yellow.max()} (mean: {h_yellow.mean():.0f})")
    print(f"S: {s_yellow.min()} - {s_yellow.max()} (mean: {s_yellow.mean():.0f})")
    print(f"V: {v_yellow.min()} - {v_yellow.max()} (mean: {v_yellow.mean():.0f})")
    
    h_min, h_max = max(0, h_yellow.min() - 3), min(180, h_yellow.max() + 3)
    s_min, s_max = max(0, s_yellow.min() - 10), min(255, s_yellow.max() + 10)
    v_min, v_max = max(0, v_yellow.min() - 10), min(255, v_yellow.max() + 10)
    print(f"推荐范围: ({h_min}, {s_min}, {v_min}) ~ ({h_max}, {s_max}, {v_max})")
else:
    print("鸟 ROI 中未找到黄色像素")

print()
print("【管道 ROI 分析】")
h_pipe = pipe_roi[:, :, 0]
s_pipe = pipe_roi[:, :, 1]
v_pipe = pipe_roi[:, :, 2]

# 找出灰色（S < 100, V > 50）
gray_mask = (s_pipe < 100) & (v_pipe > 50)
if gray_mask.sum() > 0:
    h_gray = h_pipe[gray_mask]
    s_gray = s_pipe[gray_mask]
    v_gray = v_pipe[gray_mask]
    print(f"灰色像素数: {len(h_gray)}")
    print(f"H: {h_gray.min()} - {h_gray.max()} (mean: {h_gray.mean():.0f})")
    print(f"S: {s_gray.min()} - {s_gray.max()} (mean: {s_gray.mean():.0f})")
    print(f"V: {v_gray.min()} - {v_gray.max()} (mean: {v_gray.mean():.0f})")
    
    h_min, h_max = 0, 180  # 灰色通常与 H 无关
    s_min, s_max = 0, min(100, s_gray.max() + 10)
    v_min, v_max = max(0, v_gray.min() - 10), min(255, v_gray.max() + 10)
    print(f"推荐范围: ({h_min}, {s_min}, {v_min}) ~ ({h_max}, {s_max}, {v_max})")
else:
    print("管道 ROI 中未找到灰色像素")
