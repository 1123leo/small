"""
自动分析游戏截图中的黄色（鸟）和灰色（管道）的 HSV 范围
"""
import cv2
import numpy as np

# 读取图片
image = cv2.imread('game_screenshot.png')
if image is None:
    print("Error: Could not read game_screenshot.png")
    exit(1)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print("=" * 60)
print("分析游戏截图中的颜色范围")
print("=" * 60)

# 手动定义粗略范围抓取样本
# 通常鸟应该在图像左上区域，管道在中间到右边

# 黄色范围（通常 H=15-30）
yellow_mask_loose = cv2.inRange(hsv, (10, 30, 30), (40, 255, 255))
yellow_pixels = hsv[yellow_mask_loose > 0]

# 灰色范围（通常 S<100）
gray_mask_loose = cv2.inRange(hsv, (0, 0, 50), (180, 100, 200))
gray_pixels = hsv[gray_mask_loose > 0]

print("\n【黄色（鸟）像素统计】")
if len(yellow_pixels) > 0:
    h_min = yellow_pixels[:, 0].min()
    h_max = yellow_pixels[:, 0].max()
    s_min = yellow_pixels[:, 1].min()
    s_max = yellow_pixels[:, 1].max()
    v_min = yellow_pixels[:, 2].min()
    v_max = yellow_pixels[:, 2].max()
    print(f"数量: {len(yellow_pixels)}")
    print(f"H 范围: {h_min} ~ {h_max}")
    print(f"S 范围: {s_min} ~ {s_max}")
    print(f"V 范围: {v_min} ~ {v_max}")
    # 扩展范围以增加容错性
    h_min = max(0, h_min - 5)
    h_max = min(180, h_max + 5)
    s_min = max(0, s_min - 10)
    v_min = max(0, v_min - 10)
    print(f"\n推荐范围（扩展）:")
    print(f"  lower: ({h_min}, {s_min}, {v_min})")
    print(f"  upper: ({h_max}, {s_max}, {v_max})")
else:
    print("未检测到黄色像素")

print("\n【灰色（管道）像素统计】")
if len(gray_pixels) > 0:
    h_min = gray_pixels[:, 0].min()
    h_max = gray_pixels[:, 0].max()
    s_min = gray_pixels[:, 1].min()
    s_max = gray_pixels[:, 1].max()
    v_min = gray_pixels[:, 2].min()
    v_max = gray_pixels[:, 2].max()
    print(f"数量: {len(gray_pixels)}")
    print(f"H 范围: {h_min} ~ {h_max}")
    print(f"S 范围: {s_min} ~ {s_max}")
    print(f"V 范围: {v_min} ~ {v_max}")
    # 灰色通常 S 很低
    s_max = min(100, s_max + 10)
    print(f"\n推荐范围（扩展）:")
    print(f"  lower: ({h_min}, 0, {v_min})")
    print(f"  upper: ({h_max}, {s_max}, {v_max})")
else:
    print("未检测到灰色像素")

print("\n" + "=" * 60)
