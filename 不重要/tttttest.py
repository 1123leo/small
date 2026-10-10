import keyboard
import pydirectinput
import time
import pygetwindow as gw

def hardware_test():
    print("=== 2025 AI 實戰硬體訊號測試 ===")
    print("1. 請確保遊戲視窗已開啟。")
    print("2. 程式將在 5 秒後開始嘗試發送『跳躍』訊號。")
    print("3. 請立即將滑鼠點擊到遊戲視窗中央，使其成為焦點。")
    
    # 嘗試自動激活視窗
    try:
        win = gw.getWindowsWithTitle("Play Flappy Bird")[0]
        win.activate()
    except:
        print("⚠️ 無法自動定位視窗，請手動點擊遊戲！")

    time.sleep(5)

    # 測試三種不同的按鍵驅動方式
    test_methods = [
        ("Method A: pydirectinput (硬體掃描碼)", lambda: pydirectinput.press('space')),
        ("Method B: keyboard (系統底層驅動)", lambda: keyboard.press_and_release('space')),
        ("Method C: 長按模式 (模擬真人按壓)", lambda: (pydirectinput.keyDown('space'), time.sleep(0.1), pydirectinput.keyUp('space')))
    ]

    for name, method in test_methods:
        print(f"\n正在執行 {name}...")
        for i in range(3): # 每個方法試跳 3 次
            print(f"  -> 第 {i+1} 次嘗試...")
            method()
            time.sleep(1.5) # 間隔 1.5 秒方便你觀察

    print("\n=== 測試結束 ===")
    print("請檢查：哪一種方法讓鳥跳起來了？")
    print("如果都沒有動：")
    print("1. 請確認是否以『管理員身分』執行 VS Code/CMD。")
    print("2. 換一個 Flappy Bird 遊戲網頁測試（可能是遊戲鎖死了虛擬輸入）。")

if __name__ == "__main__":
    hardware_test()