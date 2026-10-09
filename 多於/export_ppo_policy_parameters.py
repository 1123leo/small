import torch
from stable_baselines3 import PPO

# 載入模型
model = PPO.load("best_model")

# 模擬一個 36 維的輸入 (1 是 Batch Size, 36 是特徵長度)
dummy_input = torch.randn(1, 36)

# 導出模型 (我們只需要 policy 部分)
torch.onnx.export(
    model.policy, 
    dummy_input, 
    "bird_model.onnx", 
    opset_version=11,
    input_names=['input'], 
    output_names=['output']
)
print("ONNX 模型已導出為 bird_model.onnx")