import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import time


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


def get_dataloader(data_dir="./processed_data", batch_size=64, shuffle=True, num_workers=0):
    """
    创建 DataLoader 的快捷函数

    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 数据加载进程数 (Windows下建议设为0)

    Returns:
        DataLoader 对象
    """
    dataset = GoDataset(data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return dataloader


def quick_check(data_dir="./processed_data"):
    """
    快速检查数据集的基本信息

    Returns:
        dict: 包含数据集的基本信息
    """
    print("=" * 60)
    print("数据集快速检查")
    print("=" * 60)

    try:
        dataset = GoDataset(data_dir)

        info = {
            "data_dir": data_dir,
            "num_batches": len(dataset.states_files),
            "total_samples": dataset.total_samples,
            "batch_sizes": dataset.batch_sizes[:5],  # 前5个批次的大小
            "sample_shape": (2, 19, 19),
            "action_range": (0, 360),
            "dtype": "float16 (存储) / float32 (加载)"
        }

        print("\n📋 数据集信息:")
        for key, value in info.items():
            if key == "batch_sizes":
                print(f"   - {key}: {value} ... (共 {len(dataset.batch_sizes)} 个批次)")
            else:
                print(f"   - {key}: {value}")

        # 测试加载第一个和最后一个样本
        print("\n🔍 测试样本加载...")
        state0, action0 = dataset[0]
        state_last, action_last = dataset[dataset.total_samples - 1]

        print(f"   - 第一个样本: state shape {state0.shape}, action {action0.item()}")
        print(f"   - 最后一个样本: state shape {state_last.shape}, action {action_last.item()}")
        print(f"   - state 值范围: {state0.min().item():.1f} ~ {state0.max().item():.1f}")

        # 测试 DataLoader
        print("\n🚀 测试 DataLoader (batch_size=64)...")
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

        start_time = time.time()
        for i, (states, actions) in enumerate(dataloader):
            if i == 0:
                print(f"   - 第一个 batch 形状: states {states.shape}, actions {actions.shape}")
                print(f"   - actions 范围: {actions.min().item()} ~ {actions.max().item()}")
                print(f"   - states 数据类型: {states.dtype}")
            if i >= 5:
                break
        elapsed = time.time() - start_time
        print(f"   - 加载 6 个 batch 耗时: {elapsed:.2f} 秒")
        print(f"   - 平均每个 batch: {elapsed / 6:.2f} 秒")

        print("\n✅ 数据集检查通过！可以开始训练。")
        print("=" * 60)

        return info

    except Exception as e:
        print(f"\n❌ 数据集检查失败: {e}")
        return None


# ==================== 测试代码 ====================
if __name__ == "__main__":
    # 运行快速检查
    quick_check()