"""
分析管道区域的 S 值分布
"""
import cv2
import numpy as np

image = cv2.imread('game_screenshot.png')
if image is None:
    print("Error: game_screenshot.png not found")
    exit(1)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 游戏截图中管道通常在右边，且覆盖整个高度
# 鸟在左边，所以管道应该在 x > 150 的区域

# 粗略定义管道区域（右侧）
pipe_region = hsv[0:800, 150:500]

# 分析 S 值
s_values = pipe_region[:, :, 1].flatten()

# 找出非黑色像素（V > 50）
v_values = pipe_region[:, :, 2]
non_black = v_values > 50
s_non_black = s_values[non_black.flatten()]

print(f"管道区域的 S 值分析（V > 50的像素）")
print(f"总像素: {len(s_non_black)}")
if len(s_non_black) > 0:
    print(f"S 值分布:")
    print(f"  最小: {s_non_black.min()}")
    print(f"  最大: {s_non_black.max()}")
    print(f"  平均: {s_non_black.mean():.0f}")
    print(f"  中位数: {np.median(s_non_black):.0f}")
    
    # 统计分布
    for s_threshold in [10, 20, 30, 40, 50]:
        count = (s_non_black <= s_threshold).sum()
        percent = 100 * count / len(s_non_black)
        print(f"  S <= {s_threshold}: {count} 像素 ({percent:.1f}%)")
