import cv2
import numpy as np
import os
from pathlib import Path
import glob

# HSV 颜色范围（需要根据游戏调整）
BIRD_LOWER = np.array([20, 100, 100])  # 黄色下界
BIRD_UPPER = np.array([40, 255, 255])  # 黄色上界
PIPE_LOWER = np.array([35, 50, 50])    # 灰色下界
PIPE_UPPER = np.array([85, 255, 255])  # 灰色上界

def detect_bird(frame):
    """检测鸟的位置"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BIRD_LOWER, BIRD_UPPER)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, mask
    
    # 找最大的轮廓（鸟）
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    
    if area < 100:  # 最小面积
        return None, mask
    
    # 获取矩形边界
    x, y, w, h = cv2.boundingRect(c)
    return {
        'x': x + w // 2,
        'y': y + h // 2,
        'area': area,
        'rect': (x, y, w, h)
    }, mask

def detect_pipes(frame):
    """检测管道位置"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, PIPE_LOWER, PIPE_UPPER)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    pipes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > 500:  # 最小面积
            x, y, w, h = cv2.boundingRect(c)
            pipes.append({
                'x': x + w // 2,
                'y': y + h // 2,
                'top': y,
                'bottom': y + h,
                'left': x,
                'right': x + w,
                'area': area,
                'rect': (x, y, w, h)
            })
    
    return pipes, mask

def process_image(image_path):
    """处理单张图片"""
    print(f"\n处理图片: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f"❌ 无法读取图片: {image_path}")
        return
    
    h, w = frame.shape[:2]
    print(f"图片尺寸: {w}x{h}")
    
    # 检测鸟和管道
    bird, bird_mask = detect_bird(frame)
    pipes, pipe_mask = detect_pipes(frame)
    
    print(f"\n🐦 鸟检测: ", end="")
    if bird:
        print(f"✓ x={bird['x']}, y={bird['y']}, area={bird['area']:.0f}")
    else:
        print("❌ 未检测到")
    
    print(f"🔲 管道检测: ✓ {len(pipes)} 个")
    for i, pipe in enumerate(pipes):
        print(f"   管道{i+1}: x={pipe['x']}, y={pipe['y']}, area={pipe['area']:.0f}")
    
    # 绘制结果
    result = frame.copy()
    
    if bird:
        x, y, w, h = bird['rect']
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)  # 绿色鸟框
        cv2.circle(result, (bird['x'], bird['y']), 3, (0, 255, 0), -1)
    
    for i, pipe in enumerate(pipes):
        x, y, w, h = pipe['rect']
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)  # 红色管道框
        cv2.putText(result, f"P{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 并排显示原图、掩膜、结果
    bird_mask_3ch = cv2.cvtColor(bird_mask, cv2.COLOR_GRAY2BGR)
    pipe_mask_3ch = cv2.cvtColor(pipe_mask, cv2.COLOR_GRAY2BGR)
    
    # 调整尺寸以便并排显示
    display_h = 240
    scale = display_h / h
    w_scaled = int(w * scale)
    
    frame_resized = cv2.resize(frame, (w_scaled, display_h))
    bird_mask_resized = cv2.resize(bird_mask_3ch, (w_scaled, display_h))
    pipe_mask_resized = cv2.resize(pipe_mask_3ch, (w_scaled, display_h))
    result_resized = cv2.resize(result, (w_scaled, display_h))
    
    display = np.hstack([frame_resized, bird_mask_resized, pipe_mask_resized, result_resized])
    
    cv2.imshow("原图 | 鸟掩膜 | 管道掩膜 | 检测结果", display)
    
    # 保存检测结果
    output_path = image_path.replace('.jpg', '_detected.jpg').replace('.png', '_detected.png')
    cv2.imwrite(output_path, result)
    print(f"✓ 检测结果已保存: {output_path}")
    
    return bird, pipes

def batch_process(image_dir="./game_images/"):
    """批量处理图片文件夹"""
    if not os.path.exists(image_dir):
        print(f"❌ 文件夹不存在: {image_dir}")
        print("创建示例用法: 将游戏截图放在 ./game_images/ 文件夹中")
        return
    
    image_files = glob.glob(os.path.join(image_dir, "*.jpg")) + \
                  glob.glob(os.path.join(image_dir, "*.png"))
    
    if not image_files:
        print(f"❌ 在 {image_dir} 中找不到图片文件")
        return
    
    print(f"找到 {len(image_files)} 张图片")
    
    for image_file in sorted(image_files):
        process_image(image_file)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # 处理指定的图片
        image_path = sys.argv[1]
        process_image(image_path)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # 批量处理 ./game_images/ 文件夹中的图片
        print("使用方式:")
        print("  1. 单张图片: python CNN_test_image.py <image_path>")
        print("  2. 批量处理: python CNN_test_image.py (需要 ./game_images/ 文件夹)")
        print("\n快捷键: q 退出")
        
        batch_process()
