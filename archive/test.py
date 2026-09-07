# test_best_model.py
# 测试 best_model.pth 是否与当前模型架构兼容

import torch
from Strategy_Network_2 import PolicyNetworkAlphaGoLee, NUM_CLASSES

# 1. 创建当前架构的模型（362 类，含 Pass 头）
model = PolicyNetworkAlphaGoLee(input_channels=2, num_actions=NUM_CLASSES)
print(f"模型类别数: {NUM_CLASSES}")

# 2. 加载 best_model.pth
checkpoint = torch.load('./checkpoints/best_model.pth', map_location='cpu')
state_dict = checkpoint['model_state_dict']

# 3. 检查 state_dict 的键和形状
print("\n=== 检查 state_dict ===")
for k, v in state_dict.items():
    if 'head' in k:
        print(f"  {k}: {v.shape}")

# 4. 尝试加载
try:
    model.load_state_dict(state_dict)
    print("\n✅ 模型加载成功！best_model.pth 与当前架构兼容")
except Exception as e:
    print(f"\n❌ 模型加载失败: {e}")

    # 5. 诊断：找出哪些层不匹配
    print("\n=== 不匹配的层 ===")
    new_state = model.state_dict()
    for k in new_state:
        if k not in state_dict:
            print(f"  缺失: {k} (形状 {new_state[k].shape})")
        elif new_state[k].shape != state_dict[k].shape:
            print(f"  形状不匹配: {k} (当前 {new_state[k].shape}, 检查点 {state_dict[k].shape})")