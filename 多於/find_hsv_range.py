import cv2
import numpy as np

# 读取图片
image = cv2.imread('game_screenshot.png')
if image is None:
    print("Error: Could not read game_screenshot.png")
    exit(1)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

print("使用这个界面找到正确的 HSV 范围:")
print("- 调整滑块来筛选颜色")
print("- 在显示的掩膜中看到目标颜色时，记录滑块值")
print("- 关闭窗口后输出最终的 HSV 范围")
print()

def nothing(x):
    pass

# 创建窗口
cv2.namedWindow('图片')
cv2.namedWindow('掩膜 - 黄色检测')
cv2.namedWindow('掩膜 - 灰色检测')

# 黄色阈值滑块
cv2.createTrackbar('黄色 H_lower', '图片', 0, 180, nothing)
cv2.createTrackbar('黄色 H_upper', '图片', 30, 180, nothing)
cv2.createTrackbar('黄色 S_lower', '图片', 50, 255, nothing)
cv2.createTrackbar('黄色 V_lower', '图片', 50, 255, nothing)

# 灰色阈值滑块
cv2.createTrackbar('灰色 H_lower', '掩膜 - 灰色检测', 0, 180, nothing)
cv2.createTrackbar('灰色 H_upper', '掩膜 - 灰色检测', 180, 180, nothing)
cv2.createTrackbar('灰色 S_lower', '掩膜 - 灰色检测', 0, 255, nothing)
cv2.createTrackbar('灰色 S_upper', '掩膜 - 灰色检测', 50, 255, nothing)
cv2.createTrackbar('灰色 V_lower', '掩膜 - 灰色检测', 50, 255, nothing)

while True:
    # 获取黄色阈值
    h_lower_bird = cv2.getTrackbarPos('黄色 H_lower', '图片')
    h_upper_bird = cv2.getTrackbarPos('黄色 H_upper', '图片')
    s_lower_bird = cv2.getTrackbarPos('黄色 S_lower', '图片')
    v_lower_bird = cv2.getTrackbarPos('黄色 V_lower', '图片')
    
    bird_lower = np.array([h_lower_bird, s_lower_bird, v_lower_bird])
    bird_upper = np.array([h_upper_bird, 255, 255])
    
    # 获取灰色阈值
    h_lower_pipe = cv2.getTrackbarPos('灰色 H_lower', '掩膜 - 灰色检测')
    h_upper_pipe = cv2.getTrackbarPos('灰色 H_upper', '掩膜 - 灰色检测')
    s_lower_pipe = cv2.getTrackbarPos('灰色 S_lower', '掩膜 - 灰色检测')
    s_upper_pipe = cv2.getTrackbarPos('灰色 S_upper', '掩膜 - 灰色检测')
    v_lower_pipe = cv2.getTrackbarPos('灰色 V_lower', '掩膜 - 灰色检测')
    
    pipe_lower = np.array([h_lower_pipe, s_lower_pipe, v_lower_pipe])
    pipe_upper = np.array([h_upper_pipe, s_upper_pipe, 255])
    
    # 创建掩膜
    bird_mask = cv2.inRange(hsv, bird_lower, bird_upper)
    pipe_mask = cv2.inRange(hsv, pipe_lower, pipe_upper)
    
    # 显示
    cv2.imshow('图片', image)
    cv2.imshow('掩膜 - 黄色检测', bird_mask)
    cv2.imshow('掩膜 - 灰色检测', pipe_mask)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q 或 ESC
        break

cv2.destroyAllWindows()

# 输出最终结果
print(f"\n=== 最终 HSV 范围 ===")
print(f"BIRD_LOWER  = np.array([{h_lower_bird}, {s_lower_bird}, {v_lower_bird}])")
print(f"BIRD_UPPER  = np.array([{h_upper_bird}, 255, 255])")
print()
print(f"PIPE_LOWER  = np.array([{h_lower_pipe}, {s_lower_pipe}, {v_lower_pipe}])")
print(f"PIPE_UPPER  = np.array([{h_upper_pipe}, {s_upper_pipe}, 255])")
