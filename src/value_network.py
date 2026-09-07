"""价值网络（Value Network）：13 层卷积（含 BatchNorm）+ 全局平均池化 + 全连接头。

本模块来自原项目 Value_Network.py，包含：
    - ValueNetworkAlphaGoLee：价值网络（输入 [B, 2, 19, 19]，输出 [B, 1] 线性标量）
    - GoDataset：mmap 内存映射数据集（读取 states_batch_*.npy / values_batch_*.npy）
    - train() / evaluate() / test_model()：MSE 回归训练循环

训练数据由 scripts/process_sgf_value.py（原 find_sgf_v3.py）生成：
标签为当前行棋方视角的 +1.0（胜）/ -1.0（负）/ 0.0（和）。
"""

import os
import glob
from itertools import islice
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


# ============================================================
# 全局常量
# ============================================================

BOARD_SIZE = 19  # 棋盘大小（价值网络训练数据为 19 路）


# ============================================================
# 1. 数据集类
# ============================================================

class GoDataset(Dataset):
    """
    围棋数据集，支持从多个批次文件加载数据（使用 mmap 内存映射，不占用大量物理内存）

    数据格式：
        states: (N, 2, 19, 19)  float16
        values:  (N,)           float32
                                -1.0 / 0.0 / +1.0（当前行棋方视角胜负标签）

    文件命名规范：
        states_batch_001.npy, states_batch_002.npy, ...
        values_batch_001.npy, values_batch_002.npy, ...
    """

    def __init__(self, data_dir="./processed_data_value"):
        """
        初始化数据集：为所有批次文件建立内存映射（mmap），不把数据加载进物理内存。

        说明：
        - mmap 句柄在 __init__ 中打开一次并缓存，__getitem__ 直接复用，
          避免每次取样本都重新打开文件；
        - 物理内存占用保持低位（按需从磁盘读取），不会因 np.concatenate 导致 OOM；
        - mmap 对象无法被 pickle 序列化，因此 num_workers 必须保持为 0
          （get_dataloaders 与 train() 均使用 num_workers=0）。

        Args:
            data_dir: 数据文件夹路径，默认为 ./processed_data_value
        """
        self.data_dir = data_dir

        # 1. 获取所有批次文件
        self.states_files = sorted([f for f in os.listdir(data_dir)
                                    if f.startswith('states_batch_') and f.endswith('.npy')])
        self.values_files = sorted([f for f in os.listdir(data_dir)
                                    if f.startswith('values_batch_') and f.endswith('.npy')])

        print(f"\n📁 找到 {len(self.states_files)} 个批次文件")
        print("🔄 正在建立内存映射（不加载数据到物理内存）...")

        # 2. 缓存所有 mmap 句柄（每个文件只打开一次）
        self.states_mmaps = []
        self.values_mmaps = []
        self.batch_sizes = []
        self.cumulative_sizes = []
        total = 0

        for i, (sf, vf) in enumerate(zip(self.states_files, self.values_files)):
            states_mmap = np.load(os.path.join(data_dir, sf), mmap_mode='r')
            values_mmap = np.load(os.path.join(data_dir, vf), mmap_mode='r')

            self.states_mmaps.append(states_mmap)
            self.values_mmaps.append(values_mmap)

            size = states_mmap.shape[0]
            self.batch_sizes.append(size)
            total += size
            self.cumulative_sizes.append(total)

            if (i + 1) % 500 == 0:
                print(f"   已映射 {i + 1}/{len(self.states_files)} 个文件")

        self.total_samples = total
        print(f"✅ 内存映射完成！")
        print(f"   总样本数: {total:,}")
        print(f"   物理内存占用: 低（按需从磁盘读取）")

    def __len__(self):
        """返回数据集的总样本数"""
        return self.total_samples

    def __getitem__(self, idx):
        """获取指定索引的样本（复用缓存的 mmap 句柄，不重新打开文件）"""
        import bisect

        # 用二分查找定位包含该索引的批次文件
        file_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        if file_idx > 0:
            offset = idx - self.cumulative_sizes[file_idx - 1]
        else:
            offset = idx

        # 直接访问缓存的 mmap 句柄（不再调用 np.load）
        state = torch.tensor(self.states_mmaps[file_idx][offset], dtype=torch.float32)
        value = torch.tensor(self.values_mmaps[file_idx][offset], dtype=torch.float32)

        return state, value


def get_dataloaders(data_dir="./processed_data_value", batch_size=64, val_ratio=0.05, num_workers=0):
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

class ValueNetworkAlphaGoLee(nn.Module):
    """价值网络：13 层卷积（每层后接 BatchNorm）+ 全局平均池化 + 2 层全连接。

    与策略网络的区别：
      - 输出不是 361/362 个落子 logit，而是 1 个标量胜率估计；
      - 13 层卷积每层后都接 BatchNorm，稳定梯度，避免深层网络训练不收敛；
      - 特征图经全局平均池化压成 192 维向量（替代 1×1 卷积 + 展平 361），
        再接 2 层全连接，输出为线性标量（不使用 Tanh，避免饱和导致梯度消失）。

    输入：
        x: [batch, input_channels, 19, 19]
           - 默认 input_channels=2（通道 0 = 当前方棋子，通道 1 = 对方棋子），
             与 find_sgf_v3.py 生成的数据一致。
    输出：
        value: [batch, 1] —— 当前行棋方最终获胜概率的估计值（线性输出）。
        配合 nn.MSELoss 训练：标签为 values_batch_*.npy（+1.0 / -1.0 / 0.0）。

    设备处理与权重初始化：
        与现有代码一致，类内部不处理设备（由外部 .to(device) 完成）。
        卷积层使用 PyTorch 默认初始化（Kaiming 均匀分布），BatchNorm 使用默认参数；
        全连接头 fc1 / fc2 使用 N(0, 0.1) + 零偏置；
        价值网络从头训练，不加载任何策略网络权重。
    """

    def __init__(self, input_channels: int = 2):
        super(ValueNetworkAlphaGoLee, self).__init__()

        # ---- 第 1 层：5×5 卷积 ----
        # padding=2 保持 19×19 空间尺寸
        self.conv1 = nn.Conv2d(
            input_channels, 192,
            kernel_size=5, stride=1, padding=2
        )
        self.bn1 = nn.BatchNorm2d(192)

        # ---- 第 2~12 层：3×3 卷积 × 11 ----
        # padding=1 保持 19×19 空间尺寸；每层卷积后都接 BatchNorm
        self.hidden_convs = nn.ModuleList([
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
            for _ in range(11)
        ])
        self.hidden_bns = nn.ModuleList([
            nn.BatchNorm2d(192)
            for _ in range(11)
        ])

        # ---- 全局平均池化：192 通道 -> 192 维向量 ----
        # 替代 1×1 卷积 + 展平 361 的旧方案：信息更完整且不依赖棋盘大小
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # ---- 全连接头：192 -> 256 -> 1 ----
        self.fc1 = nn.Linear(192, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 1)

        # ---- 价值头初始化：fc1 / fc2 使用 N(0, 0.1) + 零偏置 ----
        # 小标准差让输出保持合理尺度，避免随机初始化导致输出过大/过小，
        # 从而让 loss 能从 ~1.0 基线正常下降（从头训练，不使用预训练权重）。
        # fc1
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.fc1.bias)

        # fc2
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # x: [batch, input_channels, 19, 19]
        x = F.relu(self.bn1(self.conv1(x)))       # [B, 192, 19, 19]

        for conv, bn in zip(self.hidden_convs, self.hidden_bns):
            x = F.relu(bn(conv(x)))               # [B, 192, 19, 19]（全程不变）

        # 全局平均池化：保留每个通道的整体响应，展平为 192 维向量
        x = self.global_pool(x)                   # [B, 192, 1, 1]
        x = x.view(x.size(0), -1)                 # [B, 192]
        x = F.relu(self.bn_fc(self.fc1(x)))       # [B, 256]
        value = self.fc2(x)  # 不用 Tanh，避免饱和
        return value


# ============================================================
# 3. 断点续训工具函数
# ============================================================

def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, batch_idx, avg_loss, checkpoint_dir="./checkpoints_value"):
    """保存完整的检查点（含模型、优化器、调度器状态）。

    epoch 约定（0-based 的恢复起点）：
      - 轮中检查点：传入当前 epoch 索引 + 下一批未处理的 batch 索引（batch_idx）
      - 轮末检查点：传入下一轮 epoch 索引，batch_idx=0
    这样 load_checkpoint 返回的 (epoch, batch_idx) 可直接作为训练循环的恢复起点。
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'batch_idx': batch_idx,  # 恢复时的起点 batch（0-based，下一批未处理 batch 索引）
        'best_loss': best_loss,
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

    print(f"💾 检查点已保存: 恢复起点 epoch {epoch}（0-based）, batch {batch_idx} (最新: {latest_path})")

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
    加载检查点，返回 (epoch, batch_idx, best_loss, avg_loss)

    Args:
        checkpoint_path: 检查点路径
        model: 模型实例
        optimizer: 优化器实例（可选）
        scheduler: 调度器实例（可选）

    Returns:
        (epoch, batch_idx, best_loss, avg_loss)
    """
    if not os.path.exists(checkpoint_path):
        print(f"ℹ️ 检查点不存在: {checkpoint_path}，从头开始训练")
        return 0, 0, float('inf'), 0.0

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
    best_loss = checkpoint.get('best_loss', float('inf'))
    avg_loss = checkpoint.get('avg_loss', 0.0)

    print(f"✅ 从检查点恢复: epoch {epoch}（0-based）, batch {batch_idx}, "
          f"best_loss {best_loss:.4f}, avg_loss {avg_loss:.4f}")
    return epoch, batch_idx, best_loss, avg_loss


def find_latest_checkpoint(checkpoint_dir="./checkpoints_value"):
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
BATCH_SIZE = 256         # 更大批次：梯度更稳，价值头更容易收敛
# 诊断结论：SGD + 小 lr + batch=16 导致价值头几乎学不动（loss 停在 1.0）。
# 改用 AdamW + lr=1e-3 + batch=256，配合价值头 N(0, 0.1) 初始化与 LayerNorm 解决该问题。
LEARNING_RATE = 1e-3     # AdamW 常用学习率
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 30
VAL_RATIO = 0.05          # 验证集比例 5%
DATA_DIR = "./processed_data_value"    # 价值网络训练数据（find_sgf_v3.py 生成）
CHECKPOINT_DIR = "./checkpoints_value" # 与策略网络检查点分开存放
RESUME = True             # 是否自动从最新的检查点恢复
# =================================================


def evaluate(model, dataloader, device, criterion):
    """在验证集上评估平均 MSE（价值网络指标：越小越好）"""
    model.eval()
    total_loss = 0
    total_samples = 0
    with torch.no_grad():
        for states, values in dataloader:
            states = states.to(device)
            values = values.to(device)
            pred = model(states).squeeze(-1)   # [B]
            loss = criterion(pred, values)
            total_loss += loss.item() * len(values)
            total_samples += len(values)
    return total_loss / total_samples   # 返回平均 MSE


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
    model = ValueNetworkAlphaGoLee(input_channels=2).to(device)

    print("🔄 价值网络从头训练（不加载策略网络权重）")
    # 4. 优化器（AdamW：自适应学习率，能有效推动随机初始化的价值头，
    #    比 SGD + 小 lr 更适合让 loss 从 1.0 基线开始下降）
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # 5. 学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',           # 监控指标越小越好（损失）
        factor=0.1,           # 衰减因子
        patience=3,           # 连续3轮没有提升就衰减
        min_lr=1e-6,
        verbose=True
    )

    # 6. 损失函数
    criterion = nn.MSELoss()

    # 7. 创建保存目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 8. 统计信息
    print(f"\n📊 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 每轮批次数: {len(train_loader)}")
    print("=" * 60)

    # 9. 断点续训：自动恢复
    start_epoch = 0
    best_loss = float('inf')

    if RESUME:
        if os.path.exists(os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")):
            print(f"⚠️ 警告: 发现旧检查点，将从中恢复。"
                  f"如果这是之前失败的训练，请删除 {CHECKPOINT_DIR}/ 后重新训练")
        checkpoint_path = find_latest_checkpoint(CHECKPOINT_DIR)
        if checkpoint_path:
            start_epoch, start_batch, best_loss, _ = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler
            )
            print(f"🔄 从 epoch {start_epoch}（0-based）, batch {start_batch} 继续训练...")
        else:
            start_epoch, start_batch = 0, 0
            print("ℹ️ 未找到检查点，从头开始训练")

    # 10. 训练循环
    for epoch in range(start_epoch, NUM_EPOCHS):
        # ---- 训练 ----
        model.train()
        total_loss = 0
        processed_batches = 0   # 实际处理的 batch 数（恢复时会跳过已处理的 batch）

        # 跳过已处理的 batch，且不逐个加载它们（islice 直接从迭代器跳过）
        if epoch == start_epoch and start_batch > 0:
            train_iter = islice(train_loader, start_batch, None)
            batch_start = start_batch
        else:
            train_iter = train_loader
            batch_start = 0

        pbar = tqdm(train_iter,
                    desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
                    total=len(train_loader) - batch_start)

        for batch_idx, (states, values) in enumerate(pbar, start=batch_start):
            processed_batches += 1
            states = states.to(device)
            values = values.to(device)

            # ---- 训练数据校验（价值标签必须在 [-1, 1]）----
            assert values.min() >= -1.0 and values.max() <= 1.0, (
                f"标签越界: {values.min()}-{values.max()}（合法范围 -1.0 ~ +1.0）"
            )

            optimizer.zero_grad()
            pred = model(states).squeeze(-1)  # [B]
            assert pred.shape == values.shape, (
                f"预测形状 {pred.shape} 与标签形状 {values.shape} 不一致"
            )
            loss = criterion(pred, values)
            loss.backward()
            optimizer.step()

            # ========== 每 5000 个 batch 输出一次训练诊断 ==========
            # pred.std() ≈ 0 -> 输出卡在 0（梯度太小）
            # pred.std() ≈ 1 -> Tanh 饱和（梯度消失）
            if (batch_idx + 1) % 5000 == 0:
                pred_std = pred.std().item()
                grad_norm = sum(
                    p.grad.norm() ** 2 for p in model.parameters()
                    if p.grad is not None
                ) ** 0.5
                print(f"\n   📊 诊断 [batch {batch_idx + 1}]: "
                      f"pred.std()={pred_std:.4f}, grad_norm={grad_norm:.6f}")
                print(f"       pred 范围: [{pred.min().item():.3f}, {pred.max().item():.3f}]")
            # =======================================================

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            # ========== 每 1000 个 batch 保存一次中间检查点 ==========
            if (batch_idx + 1) % 1000 == 0:
                avg_loss_so_far = total_loss / processed_batches
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,          # 0-based：恢复后在同一轮内继续训练
                    best_loss,
                    batch_idx + 1,  # 下一批未处理的 batch 索引
                    avg_loss_so_far,
                    CHECKPOINT_DIR
                )
                print("保存成功")
                print(f"\n   💾 已保存中间检查点 (batch {batch_idx + 1}, loss: {loss.item():.4f})")
            # ==========================================================

        avg_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
        print(f"\n✅ Epoch {epoch + 1} 训练完成! 平均 Loss: {avg_loss:.4f}")

        # ---- 验证 ----
        val_loss = evaluate(model, val_loader, device, criterion)
        print(f"📊 验证平均 MSE: {val_loss:.4f}")

        # ---- 学习率调度 ----
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"📊 当前学习率: {current_lr:.6f}")

        # ---- 保存最佳模型 ----
        if val_loss < best_loss:
            best_loss = val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'best_loss': best_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, best_path)
            print(f"💾 最佳模型已保存 (验证 MSE: {best_loss:.4f})")

        # ---- 保存检查点（断点续训用） ----
        # 轮末检查点：epoch+1 为下一轮（0-based），batch 从 0 开始
        save_checkpoint(model, optimizer, scheduler, epoch + 1, best_loss, 0, avg_loss, CHECKPOINT_DIR)

    # ---- 训练结束 ----
    print("\n" + "=" * 60)
    print(f"🎉 训练完成！最佳验证 MSE: {best_loss:.4f}")
    print(f"📁 检查点保存在: {CHECKPOINT_DIR}/")
    print("=" * 60)


# ============================================================
# 5. 模型测试（可选）
# ============================================================

def test_model():
    """测试模型前向传播"""
    print("🔍 测试模型结构...")
    model = ValueNetworkAlphaGoLee(input_channels=2)

    # 模拟输入：batch_size=4, 2通道, 19x19棋盘
    dummy_input = torch.randn(4, 2, 19, 19)
    output = model(dummy_input)

    print(f"   输入形状: {dummy_input.shape}")
    print(f"   输出形状: {output.shape}")
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 价值网络输出 [B, 1]（线性输出，不使用 Tanh），无需 Softmax
    assert output.shape == (4, 1)
    print(f"   输出范围: {output.min().item():.4f} ~ {output.max().item():.4f}")

    print("✅ 模型结构测试通过！\n")



# ============================================================
# 6. 主入口
# ============================================================

if __name__ == "__main__":
    # 如果你想先测试模型结构，取消下面这行的注释
    # test_model()

    # 开始训练
    train()
