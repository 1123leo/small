#!/usr/bin/env python3
"""
自动启动游戏、截屏、分析 HSV 范围的一体化脚本
"""
import subprocess
import time
import threading
import cv2
import numpy as np
from mss import mss
import os

def start_game():
    """在后台启动游戏"""
    print("[1] 启动游戏...")
    # 启动游戏进程（不阻塞）
    game_proc = subprocess.Popen(
        ["python", "-m", "flappy_bird_gymnasium"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return game_proc

def wait_for_game():
    """等待游戏窗口出现并稳定"""
    print("[2] 等待游戏启动...")
    time.sleep(5)  # 给游戏足够的启动时间
    print("    游戏应该已经启动了")

def capture_screenshot():
    """截取游戏窗口"""
    print("[3] 截屏...")
    
    with mss() as sct:
        # 尝试找到游戏窗口（通常在主屏幕）
        monitors = sct.monitors
        # 使用主显示器
        monitor = monitors[1] if len(monitors) > 1 else monitors[0]
        
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        output_path = "auto_game_screenshot.png"
        cv2.imwrite(output_path, frame)
        print(f"    截图已保存：{output_path}")
        return output_path

def analyze_hsv(image_path):
    """分析图片中的 HSV 范围"""
    print("[4] 分析图片的 HSV 范围...")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"    错误：无法读取图片 {image_path}")
        return
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 分析黄色区域（鸟）
    print("\n    === 黄色区域分析（鸟）===")
    yellow_mask = cv2.inRange(hsv, (10, 50, 50), (35, 255, 255))
    yellow_pixels = hsv[yellow_mask > 0]
    
    if len(yellow_pixels) > 0:
        print(f"    检测到 {len(yellow_pixels)} 个黄色像素")
        print(f"    H 范围：{yellow_pixels[:, 0].min()} ~ {yellow_pixels[:, 0].max()}")
        print(f"    S 范围：{yellow_pixels[:, 1].min()} ~ {yellow_pixels[:, 1].max()}")
        print(f"    V 范围：{yellow_pixels[:, 2].min()} ~ {yellow_pixels[:, 2].max()}")
        
        h_mean = yellow_pixels[:, 0].mean()
        s_mean = yellow_pixels[:, 1].mean()
        v_mean = yellow_pixels[:, 2].mean()
        print(f"    平均值：H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")
    else:
        print("    未检测到黄色像素，使用默认范围")
    
    # 分析灰色区域（管道）
    print("\n    === 灰色区域分析（管道）===")
    gray_mask = cv2.inRange(hsv, (0, 0, 50), (180, 50, 200))
    gray_pixels = hsv[gray_mask > 0]
    
    if len(gray_pixels) > 0:
        print(f"    检测到 {len(gray_pixels)} 个灰色像素")
        print(f"    H 范围：{gray_pixels[:, 0].min()} ~ {gray_pixels[:, 0].max()}")
        print(f"    S 范围：{gray_pixels[:, 1].min()} ~ {gray_pixels[:, 1].max()}")
        print(f"    V 范围：{gray_pixels[:, 2].min()} ~ {gray_pixels[:, 2].max()}")
        
        h_mean = gray_pixels[:, 0].mean()
        s_mean = gray_pixels[:, 1].mean()
        v_mean = gray_pixels[:, 2].mean()
        print(f"    平均值：H={h_mean:.1f}, S={s_mean:.1f}, V={v_mean:.1f}")
    else:
        print("    未检测到灰色像素，使用默认范围")
    
    # 推荐参数
    print("\n    === 推荐的 HSV 参数 ===")
    if len(yellow_pixels) > 0:
        h_min = max(0, yellow_pixels[:, 0].min() - 5)
        h_max = min(180, yellow_pixels[:, 0].max() + 5)
        s_min = max(0, yellow_pixels[:, 1].min() - 20)
        s_max = min(255, yellow_pixels[:, 1].max() + 20)
        v_min = max(0, yellow_pixels[:, 2].min() - 20)
        v_max = min(255, yellow_pixels[:, 2].max() + 20)
        print(f"    BIRD_LOWER = ({int(h_min)}, {int(s_min)}, {int(v_min)})")
        print(f"    BIRD_UPPER = ({int(h_max)}, {int(s_max)}, {int(v_max)})")
    
    if len(gray_pixels) > 0:
        h_min = max(0, gray_pixels[:, 0].min() - 5)
        h_max = min(180, gray_pixels[:, 0].max() + 5)
        s_min = max(0, gray_pixels[:, 1].min() - 10)
        s_max = min(255, gray_pixels[:, 1].max() + 10)
        v_min = max(0, gray_pixels[:, 2].min() - 20)
        v_max = min(255, gray_pixels[:, 2].max() + 20)
        print(f"    PIPE_LOWER = ({int(h_min)}, {int(s_min)}, {int(v_min)})")
        print(f"    PIPE_UPPER = ({int(h_max)}, {int(s_max)}, {int(v_max)})")

def main():
    print("=" * 50)
    print("自动游戏截屏 & HSV 分析工具")
    print("=" * 50)
    
    game_proc = None
    try:
        # 启动游戏
        game_proc = start_game()
        
        # 等待游戏启动
        wait_for_game()
        
        # 截屏
        image_path = capture_screenshot()
        
        # 分析 HSV
        analyze_hsv(image_path)
        
        print("\n" + "=" * 50)
        print("分析完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"错误：{e}")
    
    finally:
        # 关闭游戏
        if game_proc:
            print("\n关闭游戏...")
            game_proc.terminate()
            try:
                game_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                game_proc.kill()

if __name__ == "__main__":
    main()
