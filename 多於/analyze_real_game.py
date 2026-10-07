"""
分析真实游戏截图的 HSV 颜色范围
"""
import cv2
import numpy as np
from pathlib import Path

# 检查真实截图
if not Path('real_game_screenshot.png').exists():
    print("❌ 需要先截图！请运行:")
    print("  python take_screenshot.py")
    print("\n操作步骤:")
    print("  1. 启动 Flappy Bird 游戏")
    print("  2. 运行 take_screenshot.py")
    print("  3. 再运行此脚本")
    exit(1)

# 读取真实游戏截图
img = cv2.imread('real_game_screenshot.png')
if img is None:
    print("❌ 无法读取 real_game_screenshot.png")
    exit(1)

print(f"✓ 读取图片: {img.shape[1]}x{img.shape[0]}")

# 转 HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 【方案 1】粗略检测，找出色彩分布
print("\n【粗略检测】所有非纯黑色像素的 HSV 分布:")
non_black = hsv[np.sum(img, axis=2) > 30]  # 排除纯黑色
if len(non_black) > 0:
    print(f"样本数: {len(non_black)}")
    print(f"H: {non_black[:, 0].min()}-{non_black[:, 0].max()} (平均 {non_black[:, 0].mean():.0f})")
    print(f"S: {non_black[:, 1].min()}-{non_black[:, 1].max()} (平均 {non_black[:, 1].mean():.0f})")
    print(f"V: {non_black[:, 2].min()}-{non_black[:, 2].max()} (平均 {non_black[:, 2].mean():.0f})")

# 【方案 2】尝试多个 H 值范围找目标颜色
print("\n【尝试不同的 H 值范围】:")
for h_center in [15, 20, 25, 30]:
    h_lower = max(0, h_center - 10)
    h_upper = min(180, h_center + 10)
    mask = cv2.inRange(hsv, (h_lower, 50, 50), (h_upper, 255, 255))
    count = np.sum(mask > 0)
    if count > 0:
        colors_in_range = hsv[mask > 0]
        s_avg = colors_in_range[:, 1].mean()
        v_avg = colors_in_range[:, 2].mean()
        print(f"  H={h_lower}-{h_upper}: {count:6d} 像素 (S平均{s_avg:.0f}, V平均{v_avg:.0f})")

# 【方案 3】检测高饱和度（可能是鸟）和低饱和度（可能是管道）
print("\n【按饱和度分类】:")
high_sat = hsv[hsv[:, 1] > 150]  # 高饱和度
low_sat = hsv[(hsv[:, 1] < 50) & (hsv[:, 2] > 50)]  # 低饱和度但明亮

if len(high_sat) > 0:
    print(f"高饱和度(S>150): {len(high_sat)} 像素")
    print(f"  H: {high_sat[:, 0].min()}-{high_sat[:, 0].max()} (平均 {high_sat[:, 0].mean():.0f})")
    print(f"  S: {high_sat[:, 1].min()}-{high_sat[:, 1].max()}")
    print(f"  V: {high_sat[:, 2].min()}-{high_sat[:, 2].max()}")

if len(low_sat) > 0:
    print(f"\n低饱和度(S<50): {len(low_sat)} 像素")
    print(f"  H: {low_sat[:, 0].min()}-{low_sat[:, 0].max()}")
    print(f"  S: {low_sat[:, 1].min()}-{low_sat[:, 1].max()} (平均 {low_sat[:, 1].mean():.0f})")
    print(f"  V: {low_sat[:, 2].min()}-{low_sat[:, 2].max()} (平均 {low_sat[:, 2].mean():.0f})")

print("\n💡 根据上面的分析，你可以调整 CNN test gym mss.py 中的:")
print("  BIRD_LOWER = (H_min, S_min, V_min)")
print("  BIRD_UPPER = (H_max, S_max, V_max)")
print("  PIPE_LOWER = (H_min, S_min, V_min)")
print("  PIPE_UPPER = (H_max, S_max, V_max)")
