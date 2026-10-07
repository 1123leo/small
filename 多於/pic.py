import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# --- 修正後的中文顯示設定 ---
def set_ch_font():
    # 1. 嘗試直接設定 Windows 常用的微軟正黑體
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI Symbol', 'SimSun', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示為方塊的問題

set_ch_font()

# --- 數據準備 (2005-2025) ---
years = [str(y) for y in range(2005, 2026)]

# 臺北市數據
tp_carbon = [1307.4, 1280.0, 1260.0, 1250.0, 1240.7, 1245.8, 1245.9, 1219.6, 1195.5, 1199.2, 1208.5, 1241.0, 1261.6, 1211.4, 1159.6, 1138.8, 1114.3, 1084.9, 1093.2, 1053.8, 980.5]
tp_temp = [23.3, 23.4, 23.5, 23.1, 23.3, 23.3, 23.2, 23.4, 23.5, 23.8, 24.1, 24.1, 23.9, 23.7, 23.8, 24.6, 24.2, 24.0, 24.3, 24.5, 24.1]

# 臺中市數據
tc_carbon = [3226.0, 3250.0, 3280.0, 3210.0, 3190.0, 3305.0, 3295.0, 3315.0, 3340.0, 3325.0, 3310.0, 3320.0, 3380.0, 3410.0, 3390.0, 3325.0, 3483.0, 3342.0, 3437.0, 3310.0, 2258.0]
tc_temp = [23.4, 23.5, 23.6, 23.2, 23.1, 23.4, 23.6, 23.5, 23.8, 24.0, 24.2, 24.2, 24.3, 24.2, 24.4, 24.7, 24.4, 24.1, 24.5, 24.6, 24.0]

# 臺南市數據
tn_carbon = [2642.0, 2655.0, 2670.0, 2610.0, 2580.0, 2687.0, 2341.0, 2320.0, 2355.0, 2330.0, 2310.0, 2295.0, 2315.0, 2229.0, 2233.0, 2140.0, 2312.0, 2270.0, 2205.0, 2150.0, 1849.4]
tn_temp = [24.2, 24.3, 24.5, 24.1, 24.0, 24.3, 24.4, 24.4, 24.6, 24.8, 25.1, 25.1, 25.0, 25.0, 25.2, 25.5, 25.3, 25.0, 25.4, 25.5, 25.1]

def plot_graph(city_name, carbon, temp, color_c, color_t):
    # 建立圖表，設定 dpi 讓文字更清晰
    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=200)
    
    # 左軸：碳排放
    ax1.set_xlabel('年份(Year)', fontsize=12)
    ax1.set_ylabel(f'碳排放量(萬公噸)', color=color_c, fontsize=12)
    ax1.plot(years, carbon, color=color_c, marker='o', linewidth=2.5, label='碳排放量')
    ax1.tick_params(axis='y', labelcolor=color_c)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # 右軸：氣溫
    ax2 = ax1.twinx()
    ax2.set_ylabel(f'年平均氣溫(°C)', color=color_t, fontsize=12)
    ax2.plot(years, temp, color=color_t, marker='x', linestyle='--', linewidth=2, label='年均溫')
    ax2.tick_params(axis='y', labelcolor=color_t)
    
    # 標題設定：確保無多餘空格
    plt.title(f'{city_name}_年度碳排放與氣溫對照分析(2005-2025)', fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.show()

# 繪製三張獨立圖表
plot_graph('臺北市', tp_carbon, tp_temp, '#1f77b4', '#d62728')
plot_graph('臺中市', tc_carbon, tc_temp, '#2ca02c', '#d62728')
plot_graph('臺南市', tn_carbon, tn_temp, '#ff7f0e', '#d62728')
