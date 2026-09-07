import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import time
import glob

# ============================================================
# 1. 数据集类
# ============================================================

class GoDataset(Dataset):
    """
    围棋数据集，支持从多个批次文件加载数据

    数据格式：
        states: (N, 2, 19, 19)  float16
        actions: (N,)           int64

    文件命名规范：
        states_batch_001.npy, states_batch_002.npy, ...
        actions_batch_001.npy, actions_batch_002.npy, ...
    """

    def __init__(self, data_dir="./processed_data"):
        """
        初始化数据集

        Args:
            data_dir: 数据文件夹路径，默认为 ./processed_data
        """
        self.data_dir = data_dir

        # 1. 获取所有批次文件
        self.states_files = sorted([f for f in os.listdir(data_dir)
                                    if f.startswith('states_batch_') and f.endswith('.npy')])
        self.actions_files = sorted([f for f in os.listdir(data_dir)
                                     if f.startswith('actions_batch_') and f.endswith('.npy')])

        # 2. 检查文件
        if not self.states_files:
            raise FileNotFoundError(
                f"在 {data_dir} 中没有找到任何批次文件！\n"
                f"请确认:\n"
                f"1. 目录路径是否正确\n"
                f"2. 文件名是否以 'states_batch_' 开头\n"
                f"3. 是否运行了数据处理程序生成了批次文件"
            )

        if len(self.states_files) != len(self.actions_files):
            print(f"⚠️ 警告: states 文件 {len(self.states_files)} 个，"
                  f"actions 文件 {len(self.actions_files)} 个，数量不匹配！")

        # 3. 计算每个批次的样本数和累计样本数
        self.batch_sizes = []
        self.cumulative_sizes = []
        total = 0

        print("\n📁 正在扫描批次文件...")
        for i, sf in enumerate(self.states_files):
            # 使用 mmap_mode 快速获取形状，不加载数据到内存
            file_path = os.path.join(data_dir, sf)
            shape = np.load(file_path, mmap_mode='r').shape
            size = shape[0]
            self.batch_sizes.append(size)
            total += size
            self.cumulative_sizes.append(total)

            # 打印前5个批次的信息
            if i < 5:
                size_mb = os.path.getsize(file_path) / 1e6
                print(f"   批次 {i + 1:03d}: {sf} ({size_mb:.1f}MB, {size:,} 个样本)")

        if len(self.states_files) > 5:
            print(f"   ... 共 {len(self.states_files)} 个批次")

        self.total_samples = total
        print(f"\n✅ 加载完成！共 {len(self.states_files)} 个批次，{total:,} 个样本")

        # 4. 输出数据信息
        print(f"\n📊 数据信息:")
        print(f"   - 每个样本形状: (2, 19, 19)")
        print(f"   - 存储类型: float16")
        print(f"   - 加载类型: float32 (自动转换)")
        print(f"   - 动作范围: 0 ~ 360")

    def __len__(self):
        """返回数据集的总样本数"""
        return self.total_samples

    def __getitem__(self, idx):
        """
        获取指定索引的样本

        Args:
            idx: 样本索引，范围 0 ~ total_samples-1

        Returns:
            state: (2, 19, 19) 的 float32 张量
            action: 0~360 的整数标签
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"索引 {idx} 超出范围 [0, {self.total_samples})")

        # 1. 找到 idx 对应的批次文件
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                batch_idx = i
                # 计算在该批次中的偏移量
                prev_size = self.cumulative_sizes[i - 1] if i > 0 else 0
                offset = idx - prev_size
                break

        # 2. 加载对应批次的数据（mmap模式，只读取需要的部分）
        states = np.load(os.path.join(self.data_dir, self.states_files[batch_idx]), mmap_mode='r')
        actions = np.load(os.path.join(self.data_dir, self.actions_files[batch_idx]), mmap_mode='r')

        # 3. 转换为 PyTorch 张量
        state = torch.tensor(states[offset], dtype=torch.float32)
        action = torch.tensor(actions[offset], dtype=torch.long)

        return state, action


def get_dataloaders(data_dir="./processed_data", batch_size=64, val_ratio=0.05, num_workers=0):
    """
    返回训练集和验证集的 DataLoader

    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        val_ratio: 验证集比例（默认 5%）
        num_workers: 数据加载进程数 (Windows下建议设为0)

    Returns:
        train_loader, val_loader
    """
    full_dataset = GoDataset(data_dir)
    total = len(full_dataset)
    val_size = int(total * val_ratio)
    train_size = total - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"\n📊 数据集划分:")
    print(f"   - 训练集: {train_size:,} 样本 ({100 - val_ratio*100:.0f}%)")
    print(f"   - 验证集: {val_size:,} 样本 ({val_ratio*100:.0f}%)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    return train_loader, val_loader


# ============================================================
# 2. 模型定义
# ============================================================

class ResidualBlock(nn.Module):
    """残差块（BasicBlock）"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class PolicyNetworkResNet(nn.Module):
    """
    基于 ResNet-18 改造的围棋策略网络

    输入：19×19×N 的特征图（N为特征通道数）
    输出：19×19 的落子概率分布（logits）
    """

    def __init__(self, input_channels=2, num_actions=361):
        super(PolicyNetworkResNet, self).__init__()

        # 初始卷积层
        self.conv_input = nn.Conv2d(
            input_channels, 64,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(64)

        # ResNet-18 主体（4个Stage，每个Stage有2个残差块）
        self.stage1 = self._make_layer(64, 64, 2, stride=1)
        self.stage2 = self._make_layer(64, 128, 2, stride=1)
        self.stage3 = self._make_layer(128, 256, 2, stride=1)
        self.stage4 = self._make_layer(256, 512, 2, stride=1)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 输出层
        self.fc_out = nn.Linear(512, num_actions)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        """构建一个Stage，包含多个残差块"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch_size, input_channels, 19, 19)
        x = F.relu(self.bn_input(self.conv_input(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.fc_out(x)
        return logits


# ============================================================
# 3. 断点续训工具函数
# ============================================================

def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, batch_idx,avg_loss, checkpoint_dir="./checkpoints"):
    """保存完整的检查点（含模型、优化器、调度器状态）"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,  # 新增
        'best_acc': best_acc,
        'avg_loss': avg_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }

    # 保存最新检查点
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    torch.save(checkpoint, latest_path)

    # 同时保存为 epoch 命名的检查点（方便回溯）
    epoch_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, epoch_path)

    print(f"💾 检查点已保存: epoch {epoch} (最新: {latest_path})")

    # 自动清理旧的 epoch 检查点（只保留最近5个）
    epoch_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")))
    if len(epoch_files) > 5:
        for f in epoch_files[:-5]:
            try:
                os.remove(f)
                print(f"   🗑️ 已清理旧检查点: {os.path.basename(f)}")
            except:
                pass


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    加载检查点，返回 (epoch, best_acc, avg_loss)

    Args:
        checkpoint_path: 检查点路径
        model: 模型实例
        optimizer: 优化器实例（可选）
        scheduler: 调度器实例（可选）

    Returns:
        (epoch, best_acc, avg_loss)
    """
    if not os.path.exists(checkpoint_path):
        print(f"ℹ️ 检查点不存在: {checkpoint_path}，从头开始训练")
        return 0, 0.0, 0.0

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # 加载模型
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 模型已加载")

    # 加载优化器
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ 优化器已加载")

    # 加载调度器
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"✅ 调度器已加载")

    epoch = checkpoint.get('epoch', 0)
    batch_idx = checkpoint.get('batch_idx', 0)  # 新增
    best_acc = checkpoint.get('best_acc', 0.0)
    avg_loss = checkpoint.get('avg_loss', 0.0)

    print(f"✅ 从检查点恢复: epoch {epoch}, best_acc {best_acc:.4f}, avg_loss {avg_loss:.4f}")
    return epoch,batch_idx,best_acc, avg_loss


def find_latest_checkpoint(checkpoint_dir="./checkpoints"):
    """自动查找最新的检查点文件"""
    # 优先查找 latest_checkpoint.pth
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pth")
    if os.path.exists(latest_path):
        return latest_path

    # 否则查找 epoch 命名的检查点，取最新的
    epoch_files = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth")))
    if epoch_files:
        return epoch_files[-1]

    return None


# ============================================================
# 4. 训练代码
# ============================================================

# ==================== 训练配置 ====================
BATCH_SIZE = 16          # 如果显存不够，改回 64
LEARNING_RATE = 0.01      # AlphaGo 的初始学习率（SGD 用 0.01）
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 30
VAL_RATIO = 0.05          # 验证集比例 5%
DATA_DIR = "./processed_data"
CHECKPOINT_DIR = "./checkpoints"
RESUME = True             # 是否自动从最新的检查点恢复
# =================================================


def evaluate(model, dataloader, device):
    """在验证集上评估准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for states, actions in dataloader:
            states = states.to(device)
            actions = actions.to(device)
            logits = model(states)
            pred = logits.argmax(dim=1)
            correct += (pred == actions).sum().item()
            total += actions.size(0)
    return correct / total


def train():
    """主训练函数（支持断点续训）"""
    # 1. 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    # 2. 加载数据
    print("\n📦 加载数据...")
    train_loader, val_loader = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
        num_workers=0
    )

    # 3. 创建模型
    model = PolicyNetworkResNet(input_channels=2, num_actions=361).to(device)

    # 4. 优化器（使用 SGD + 动量，更接近 AlphaGo）
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # 5. 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',           # 监控指标越大越好（准确率）
        factor=0.1,           # 衰减因子
        patience=3,           # 连续3轮没有提升就衰减
        min_lr=1e-6,
        verbose=True
    )

    # 6. 损失函数
    criterion = nn.CrossEntropyLoss()

    # 7. 创建保存目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 8. 统计信息
    print(f"\n📊 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 每轮批次数: {len(train_loader)}")
    print("=" * 60)

    # 9. 断点续训：自动恢复
    start_epoch = 0
    best_acc = 0.0

    if RESUME:
        checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)
        if checkpoint_path:
            start_epoch, start_batch, best_acc, _ = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler
            )
            print(f"🔄 从 epoch {start_epoch}, batch {start_batch} 继续训练...")
        else:
            start_epoch, start_batch = 0, 0
            print("ℹ️ 未找到检查点，从头开始训练")

    # 10. 训练循环
    for epoch in range(start_epoch, NUM_EPOCHS):
        # ---- 训练 ----
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for batch_idx,(states, actions) in enumerate(pbar):
            if epoch == start_epoch and batch_idx < start_batch:
                continue  # 跳过已处理的 batch
            states = states.to(device)
            actions = actions.to(device)

            optimizer.zero_grad()
            logits = model(states)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            # ========== 新增：每 10000 个 batch 保存一次检查点 ==========
            if (batch_idx + 1) % 1000 == 0:
                avg_loss_so_far = total_loss / (batch_idx + 1)
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    best_acc,
                    batch_idx + 1,  # ✅ 这里传 batch_idx
                    avg_loss_so_far,
                    CHECKPOINT_DIR
                )
                print("保存成功")
                print(f"\n   💾 已保存中间检查点 (batch {batch_idx + 1}, loss: {loss.item():.4f})")
            # ==========================================================

        avg_loss = total_loss / len(train_loader)
        print(f"\n✅ Epoch {epoch + 1} 训练完成! 平均 Loss: {avg_loss:.4f}")

        # ---- 验证 ----
        val_acc = evaluate(model, val_loader, device)
        print(f"📊 验证准确率: {val_acc:.4f}")

        # ---- 学习率调度 ----
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📊 当前学习率: {current_lr:.6f}")

        # ---- 保存最佳模型 ----
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'best_acc': best_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_path)
            print(f"💾 最佳模型已保存 (准确率: {best_acc:.4f})")

        # ---- 保存检查点（断点续训用） ----
        save_checkpoint(model, optimizer, scheduler, epoch + 1, best_acc,0, avg_loss, CHECKPOINT_DIR)

    # ---- 训练结束 ----
    print("\n" + "=" * 60)
    print(f"🎉 训练完成！最佳验证准确率: {best_acc:.4f}")
    print(f"📁 检查点保存在: {CHECKPOINT_DIR}/")
    print("=" * 60)


# ============================================================
# 5. 模型测试（可选）
# ============================================================

def test_model():
    """测试模型前向传播"""
    print("🔍 测试模型结构...")
    model = PolicyNetworkResNet(input_channels=2, num_actions=361)

    # 模拟输入：batch_size=4, 2通道, 19x19棋盘
    dummy_input = torch.randn(4, 2, 19, 19)
    output = model(dummy_input)

    print(f"   输入形状: {dummy_input.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 应用 Softmax 得到概率分布
    probs = F.softmax(output, dim=1)
    print(f"   概率分布形状: {probs.shape}")
    print(f"   每行概率之和: {probs.sum(dim=1)}")

    print("✅ 模型结构测试通过！\n")


# ============================================================
# 6. 主入口
# ============================================================

if __name__ == "__main__":
    # 如果你想先测试模型结构，取消下面这行的注释
    # test_model()

    # 开始训练
    train()