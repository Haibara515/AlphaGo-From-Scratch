import torch
import torch.nn.functional as F
from Strategy_Network import PolicyNetworkResNet  # 导入你的模型定义

# ==================== 配置 ====================
CHECKPOINT_PATH = "./checkpoints/latest_checkpoint(2) .pth"  # 或者用 checkpoint_epoch_2.pth
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================

def load_model(checkpoint_path, input_channels=2, num_actions=361):
    """加载检查点并返回模型"""
    # 1. 创建模型
    model = PolicyNetworkResNet(input_channels=input_channels, num_actions=num_actions)

    # 2. 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    # 3. 提取模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 模型加载成功！")
        print(f"   - 保存时的 epoch: {checkpoint.get('epoch', '未知')}")
        print(f"   - 保存时的 best_acc: {checkpoint.get('best_acc', '未知'):.4f}")
        print(f"   - 保存时的 avg_loss: {checkpoint.get('avg_loss', '未知'):.4f}")
    else:
        # 如果直接保存的是 state_dict
        model.load_state_dict(checkpoint)
        print("✅ 模型加载成功（直接加载 state_dict）")

    model.to(DEVICE)
    model.eval()
    return model


def test_forward(model):
    """测试前向传播"""
    print("\n🔍 测试前向传播...")

    # 模拟输入：batch_size=4, 2通道, 19x19棋盘
    dummy_input = torch.randn(4, 2, 19, 19).to(DEVICE)

    with torch.no_grad():
        logits = model(dummy_input)
        probs = F.softmax(logits, dim=1)

    print(f"   - 输入形状: {dummy_input.shape}")
    print(f"   - 输出 logits 形状: {logits.shape}")  # (4, 361)
    print(f"   - 输出概率形状: {probs.shape}")  # (4, 361)
    print(f"   - 每行概率之和: {probs.sum(dim=1)}")  # 应该全为 1
    print(f"   - 预测动作: {logits.argmax(dim=1)}")  # 每个样本的预测落子

    # 检查是否有 NaN
    if torch.isnan(logits).any():
        print("   ⚠️ 警告：输出包含 NaN 值！")
    else:
        print("   ✅ 输出正常，无 NaN 值")

    return logits, probs


def test_loaded_checkpoint():
    """完整测试流程"""
    print("=" * 60)
    print("围棋策略网络模型测试")
    print("=" * 60)

    # 1. 检查文件是否存在
    import os
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"❌ 检查点文件不存在: {CHECKPOINT_PATH}")
        print(f"   请确认路径是否正确")
        return

    file_size = os.path.getsize(CHECKPOINT_PATH) / 1e6
    print(f"📁 检查点文件: {CHECKPOINT_PATH} ({file_size:.1f} MB)")

    # 2. 加载模型
    model = load_model(CHECKPOINT_PATH)
    print(f"   - 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - 模型设备: {next(model.parameters()).device}")

    # 3. 测试前向传播
    test_forward(model)

    # 4. 统计信息
    print("\n📊 模型统计:")
    print(f"   - 总参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - 可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_loaded_checkpoint()