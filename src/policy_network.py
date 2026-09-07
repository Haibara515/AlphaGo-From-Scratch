"""策略网络（Policy Network）：AlphaGo Lee 13 层卷积架构。

本模块来自原项目 Strategy_Network_2.py，包含：
    - PolicyNetworkAlphaGoLee：策略网络（输入 [B, 2, 19, 19]，输出 [B, 362] logits，
      前 361 维对应棋盘交点，最后一维对应 Pass）
    - GoDataset：mmap 内存映射数据集（读取 states_batch_*.npy / actions_batch_*.npy）
    - train() / evaluate() / test_model()：监督学习训练循环（CrossEntropyLoss）

训练数据由 scripts/process_sgf_policy.py（原 find_sgf_v2.py）生成。
"""

import os
import time
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
# 全局常量与标签编码
# ============================================================

BOARD_SIZE = 19
NUM_CLASSES = BOARD_SIZE * BOARD_SIZE + 1  # 361 个棋盘交点 + 1 个 Pass = 362
PASS_ACTION = BOARD_SIZE * BOARD_SIZE      # 361：Pass 类别标签


def encode_label(move):
    """
    将 sgfmill 的落子格式转换为模型标签（与 find_sgf_v2.py 的编码保持一致）。

    Args:
        move: sgfmill 格式的落子
            - (row, col) 表示落子位置（0~18 的整数坐标）
            - None 表示 Pass

    Returns:
        0~360 的棋盘位置标签 (row * BOARD_SIZE + col)，或 361 的 Pass 标签
    """
    if move is None:
        return PASS_ACTION  # 361 = Pass

    row, col = move
    # 校验坐标，防止越界标签静默污染训练数据
    assert 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE, f"非法落子坐标: {move}"
    return row * BOARD_SIZE + col


# ============================================================
# 1. 数据集类
# ============================================================

class GoDataset(Dataset):
    """
    围棋数据集，支持从多个批次文件加载数据（使用 mmap 内存映射，不占用大量物理内存）

    数据格式：
        states: (N, 2, 19, 19)  float16
        actions: (N,)           int64
                                0~360 为棋盘位置，361 为 Pass

    文件命名规范：
        states_batch_001.npy, states_batch_002.npy, ...
        actions_batch_001.npy, actions_batch_002.npy, ...
    """

    def __init__(self, data_dir="./processed_data"):
        """
        初始化数据集：为所有批次文件建立内存映射（mmap），不把数据加载进物理内存。

        说明：
        - mmap 句柄在 __init__ 中打开一次并缓存，__getitem__ 直接复用，
          避免每次取样本都重新打开文件；
        - 物理内存占用保持低位（按需从磁盘读取），不会因 np.concatenate 导致 OOM；
        - mmap 对象无法被 pickle 序列化，因此 num_workers 必须保持为 0
          （get_dataloaders 与 train() 均使用 num_workers=0）。

        Args:
            data_dir: 数据文件夹路径，默认为 ./processed_data
        """
        self.data_dir = data_dir

        # 1. 获取所有批次文件
        self.states_files = sorted([f for f in os.listdir(data_dir)
                                    if f.startswith('states_batch_') and f.endswith('.npy')])
        self.actions_files = sorted([f for f in os.listdir(data_dir)
                                     if f.startswith('actions_batch_') and f.endswith('.npy')])

        print(f"\n📁 找到 {len(self.states_files)} 个批次文件")
        print("🔄 正在建立内存映射（不加载数据到物理内存）...")

        # 2. 缓存所有 mmap 句柄（每个文件只打开一次）
        self.states_mmaps = []
        self.actions_mmaps = []
        self.batch_sizes = []
        self.cumulative_sizes = []
        total = 0

        for i, (sf, af) in enumerate(zip(self.states_files, self.actions_files)):
            states_mmap = np.load(os.path.join(data_dir, sf), mmap_mode='r')
            actions_mmap = np.load(os.path.join(data_dir, af), mmap_mode='r')

            self.states_mmaps.append(states_mmap)
            self.actions_mmaps.append(actions_mmap)

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
        action = torch.tensor(self.actions_mmaps[file_idx][offset], dtype=torch.long)

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

class PolicyNetworkAlphaGoLee(nn.Module):
    """AlphaGo Lee 策略网络（2016 年 Nature 论文）：13 层纯卷积结构。

    与 Strategy_Network.py 中的 PolicyNetworkResNet 的关键区别：
      - 没有全局平均池化（AdaptiveAvgPool2d）：空间分辨率 19×19 全程保留；
      - 没有全连接层（fc_out）；
      - 没有在输出前 Flatten：棋盘部分由 1×1 卷积输出 [B, 1, 19, 19]，
        再 reshape 成 [B, 361]；Pass 部分由独立的 1×1 卷积 + 全局平均池化
        得到单个标量 logit，最后拼接成 [B, 362]。

    输入：
        x: [batch, input_channels, 19, 19]
           - 默认 input_channels=2，与 Strategy_Network.py 现有数据管线一致
             （通道 0 = 当前方棋子，通道 1 = 对方棋子）。
           - AlphaGo Lee 原始论文使用 48 个特征平面（8 步历史 × 双方棋色 +
             轮次/其他手工特征），可通过 PolicyNetworkAlphaGoLee(input_channels=48)
             使用完整特征集。
    输出：
        logits: [batch, 362] —— 原始 logits，未过 Softmax。
            前 361 个 logit 对应棋盘交点（索引 = row*19+col），
            最后一个 logit（索引 361）对应 Pass。
        与 Strategy_Network.py 的约定一致：直接配 nn.CrossEntropyLoss，
        （标签范围 0~361），Softmax 由损失函数内部处理。

    设备处理与权重初始化：
        与现有代码一致，类内部不处理设备（由外部 .to(device) 完成），
        权重使用 PyTorch 默认初始化（Kaiming 均匀分布），无自定义初始化。
    """

    def __init__(self, input_channels: int = 2, num_actions: int = NUM_CLASSES):
        super(PolicyNetworkAlphaGoLee, self).__init__()

        # 校验类别数：必须为 361（棋盘点）+ 1（Pass）= 362，
        # 否则棋盘 logit 与 Pass logit 拼接后的维度会与标签不一致。
        assert num_actions == NUM_CLASSES, (
            f"num_actions 必须为 {NUM_CLASSES}（{BOARD_SIZE * BOARD_SIZE} 个棋盘点 + 1 个 Pass），"
            f"实际传入 {num_actions}"
        )
        self.num_actions = num_actions

        # 注意：AlphaGo Lee 论文中策略网络输入是 48 个特征平面；
        # 本项目现有数据是 2 通道，因此默认 input_channels=2。
        # 若以后扩展为 48 平面特征，只需传 input_channels=48。

        # ---- 第 1 层：5×5 卷积 ----
        # padding=2 保持 19×19 空间尺寸（等价于论文中“补零到 23×23 再卷积”）
        self.conv1 = nn.Conv2d(
            input_channels, 192,
            kernel_size=5, stride=1, padding=2
        )

        # ---- 第 2~12 层：3×3 卷积 × 11 ----
        # padding=1 保持 19×19 空间尺寸
        self.hidden_convs = nn.ModuleList([
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
            for _ in range(11)
        ])

        # ---- 第 13 层（输出层）：1×1 卷积 ----
        # 192 通道 -> 1 通道，每个棋盘交点一个 logit；padding=0
        self.policy_head = nn.Conv2d(192, 1, kernel_size=1, stride=1, padding=0)

        # ---- Pass 输出头：1×1 卷积 + 全局平均池化 ----
        # Pass 是全局决策（与具体坐标无关），因此对特征图做全局平均池化
        # 得到单个标量 logit。注意：这里只池化 Pass 分支，
        # 棋盘部分的 361 个 logit 仍保持 19×19 空间结构，不受影响。
        self.pass_head = nn.Conv2d(192, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # x: [batch, input_channels, 19, 19]
        x = F.relu(self.conv1(x))                 # [B, 192, 19, 19]

        for conv in self.hidden_convs:
            x = F.relu(conv(x))                   # [B, 192, 19, 19]（全程不变）

        # 输出层不做 ReLU，输出原始 logits

        # 棋盘分支：1×1 卷积 -> [B, 1, 19, 19]，reshape 成 [B, 361]
        # 仅 reshape（不是 Flatten 进全连接层），保持“交点 -> logit”对应关系
        board_logits = self.policy_head(x)
        board_logits = board_logits.reshape(board_logits.size(0), -1)   # [B, 361]

        # Pass 分支：1×1 卷积 + 全局平均池化 -> [B, 1]
        pass_logit = self.pass_head(x).mean(dim=(2, 3))                 # [B, 1]

        # 拼接成 [B, 362]：索引 0~360 = 棋盘交点，索引 361 = Pass
        logits = torch.cat([board_logits, pass_logit], dim=1)           # [B, 362]
        return logits


# ============================================================
# 3. 断点续训工具函数
# ============================================================

def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, batch_idx, avg_loss, checkpoint_dir="./checkpoints"):
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
    加载检查点，返回 (epoch, batch_idx, best_acc, avg_loss)

    Args:
        checkpoint_path: 检查点路径
        model: 模型实例
        optimizer: 优化器实例（可选）
        scheduler: 调度器实例（可选）

    Returns:
        (epoch, batch_idx, best_acc, avg_loss)
    """
    if not os.path.exists(checkpoint_path):
        print(f"ℹ️ 检查点不存在: {checkpoint_path}，从头开始训练")
        return 0, 0, 0.0, 0.0

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

    print(f"✅ 从检查点恢复: epoch {epoch}（0-based）, batch {batch_idx}, "
          f"best_acc {best_acc:.4f}, avg_loss {avg_loss:.4f}")
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
# AlphaGo 原文使用 0.01，但那是大规模分布式异步 SGD + 大批量训练的配置；
# 单卡、batch_size=16 下 0.01 容易导致 loss 震荡。单卡训练建议：
#   - SGD + momentum: lr=0.001（当前默认）
#   - Adam: lr=3e-4（更稳定，但训练行为与 SGD 不同）
# 若未来改用多卡/大批量（如 batch_size 256+），可再调回 0.01。
LEARNING_RATE = 0.001
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
            assert logits.shape[1] == NUM_CLASSES, (
                f"输出维度应为 {NUM_CLASSES}，实际 {logits.shape[1]}"
            )
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
    model = PolicyNetworkAlphaGoLee(input_channels=2, num_actions=NUM_CLASSES).to(device)

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

        for batch_idx, (states, actions) in enumerate(pbar, start=batch_start):
            processed_batches += 1
            states = states.to(device)
            actions = actions.to(device)

            # ---- 训练数据校验（防止越界标签/输出维度导致静默错误）----
            assert actions.min() >= 0 and actions.max() <= PASS_ACTION, (
                f"标签越界: {actions.min()}-{actions.max()}（合法范围 0~{PASS_ACTION}）"
            )

            optimizer.zero_grad()
            logits = model(states)
            assert logits.shape[1] == NUM_CLASSES, (
                f"输出维度应为 {NUM_CLASSES}，实际 {logits.shape[1]}"
            )
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()

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
                    best_acc,
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
        # 轮末检查点：epoch+1 为下一轮（0-based），batch 从 0 开始
        save_checkpoint(model, optimizer, scheduler, epoch + 1, best_acc, 0, avg_loss, CHECKPOINT_DIR)

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
    model = PolicyNetworkAlphaGoLee(input_channels=2, num_actions=NUM_CLASSES)

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
