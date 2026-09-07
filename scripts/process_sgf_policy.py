"""
find_sgf_v2.py - 围棋 SGF 棋谱数据处理脚本（修正版）

使用 sgfmill 库正确模拟围棋规则（提子、让子、Pass），
从 downloads/ 文件夹读取 .zip 压缩包，解压后解析 .sgf 棋谱，
生成训练数据供 Strategy_Network_2.py 使用。

数据格式：
    states: (N, 2, 19, 19)  float16
        通道 0 = 当前行棋方的棋子
        通道 1 = 对方的棋子
    actions: (N,)  int64
        0~360 表示落子位置（row * 19 + col）
        361 表示 Pass

用法：
    pip install sgfmill
    python find_sgf_v2.py
"""

import os
import glob
import zipfile
import numpy as np
from sgfmill import sgf
from sgfmill import sgf_moves

# ============================================================
# 配置
# ============================================================
DOWNLOAD_DIR = "./downloads"  # zip 压缩包所在目录
SGF_EXTRACT_DIR = "./sgf_data"  # 解压后的 SGF 文件目录（临时）
OUTPUT_DIR = "./processed_data"  # 训练数据输出目录
BOARD_SIZE = 19  # 棋盘大小
BATCH_SAVE_SIZE = 10000  # 每个 npy 批次保存的样本数
MIN_MOVES = 10  # 最少步数（过滤过短的棋谱）
MAX_MOVES = 400  # 最大步数（过滤异常长的棋谱）


# ============================================================
# 第 1 步：解压所有 zip 文件
# ============================================================

def extract_all_zips():
    """解压 downloads 文件夹里的所有 .zip 文件到 sgf_data/"""
    os.makedirs(SGF_EXTRACT_DIR, exist_ok=True)

    zip_files = sorted([f for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.zip')])
    if not zip_files:
        print(f"❌ 在 {DOWNLOAD_DIR} 中没有找到任何 .zip 文件")
        return False

    print(f"📦 找到 {len(zip_files)} 个 .zip 文件")
    print("=" * 60)

    for i, zip_name in enumerate(zip_files):
        zip_path = os.path.join(DOWNLOAD_DIR, zip_name)
        print(f"   [{i + 1}/{len(zip_files)}] 解压: {zip_name}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(SGF_EXTRACT_DIR)
        except Exception as e:
            print(f"   ⚠️ 解压失败: {e}")

    print("=" * 60)
    print(f"✅ 解压完成！SGF 文件在: {SGF_EXTRACT_DIR}/")
    return True


# ============================================================
# 第 2 步：解析单个 SGF 文件
# ============================================================

def sgf_to_training_data(sgf_path):
    """
    解析单个 SGF 文件，提取训练样本。

    使用 sgfmill 的 sgf_moves 自动处理：
    - 提子（captures）
    - 让子棋（setup stones）
    - Pass（move is None）
    - 分支变化（取主分支）

    Returns:
        states: (n, 2, 19, 19) float16 数组，或 None
        actions: (n,) int64 数组，或 None
    """
    try:
        # 1. 读取 SGF 文件
        with open(sgf_path, 'rb') as f:
            game = sgf.Sgf_game.from_bytes(f.read())

        # 2. 检查棋盘大小
        if game.get_size() != BOARD_SIZE:
            return None, None

        # 3. 用 sgf_moves 模拟对局（自动处理提子、让子、Pass）
        board, plays = sgf_moves.get_setup_and_moves(game)

        # 4. 过滤步数太少的棋谱
        if len(plays) < MIN_MOVES:
            return None, None

        # 5. 限制最大步数
        if len(plays) > MAX_MOVES:
            plays = plays[:MAX_MOVES]

        # 6. 遍历每一步，生成训练样本
        states_list = []
        actions_list = []

        for color, move in plays:
            # 提取当前棋盘状态（在落子之前）
            state = extract_board_state(board, color)

            # 确定动作标签
            if move is None:
                action = 361  # Pass
            else:
                row, col = move
                action = row * BOARD_SIZE + col

            states_list.append(state)
            actions_list.append(action)

            # 在棋盘上落子（sgfmill 自动处理提子）
            if move is not None:
                row, col = move
                board.play(row, col, color)

        # 7. 转换为 numpy 数组
        states = np.array(states_list, dtype=np.float16)
        actions = np.array(actions_list, dtype=np.int64)

        return states, actions

    except Exception:
        return None, None


def extract_board_state(board, current_player):
    """
    从 sgfmill.Board 提取 2 通道棋盘状态。

    Args:
        board: sgfmill.Board 对象
        current_player: 当前行棋方（'b' 或 'w'）

    Returns:
        state: (2, 19, 19) float16
            通道 0 = 当前行棋方的棋子
            通道 1 = 对方的棋子
    """
    state = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)

    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            stone = board.get(row, col)
            if stone is None:
                continue
            if stone == current_player:
                state[0, row, col] = 1
            else:
                state[1, row, col] = 1

    return state


# ============================================================
# 第 3 步：批量处理所有 SGF 文件
# ============================================================

def process_all_sgf_files():
    """遍历 sgf_data/ 下所有 .sgf 文件，解析并保存训练数据"""

    # 1. 查找所有 SGF 文件
    sgf_files = []
    for root, _, files in os.walk(SGF_EXTRACT_DIR):
        for f in files:
            if f.lower().endswith('.sgf'):
                sgf_files.append(os.path.join(root, f))

    sgf_files = sorted(sgf_files)
    print(f"\n📁 在 {SGF_EXTRACT_DIR}/ 中找到 {len(sgf_files)} 个 SGF 文件")

    if not sgf_files:
        print("❌ 没有找到任何 SGF 文件！")
        return

    # 2. 创建输出目录并清理旧数据
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    old_files = glob.glob(os.path.join(OUTPUT_DIR, "*.npy"))
    if old_files:
        print(f"🗑️ 清理旧训练数据 {len(old_files)} 个文件...")
        for f in old_files:
            os.remove(f)
    print(f"✅ 输出目录已就绪: {OUTPUT_DIR}/")

    # 3. 批量处理
    all_states = []
    all_actions = []
    batch_idx = 0
    total_samples = 0
    valid_games = 0
    skipped_games = 0

    print(f"\n🔄 开始解析 SGF 棋谱...")
    print("=" * 60)

    for i, sgf_path in enumerate(sgf_files):
        states, actions = sgf_to_training_data(sgf_path)

        if states is None:
            skipped_games += 1
            continue

        all_states.append(states)
        all_actions.append(actions)
        total_samples += len(states)
        valid_games += 1

        # 进度显示
        if (i + 1) % 1000 == 0:
            print(f"   进度: {i + 1}/{len(sgf_files)} | "
                  f"有效棋谱: {valid_games} | "
                  f"累计样本: {total_samples:,} | "
                  f"跳过: {skipped_games}")

        # 达到保存阈值时写入 npy 文件
        current_buffer_size = sum(len(s) for s in all_states)
        if current_buffer_size >= BATCH_SAVE_SIZE:
            batch_idx = save_batch(all_states, all_actions, batch_idx)
            all_states = []
            all_actions = []

    # 4. 保存剩余数据
    if all_states:
        save_batch(all_states, all_actions, batch_idx)

    # 5. 输出统计
    print("=" * 60)
    print(f"\n📊 处理完成！")
    print(f"   - SGF 文件总数: {len(sgf_files):,}")
    print(f"   - 有效棋谱: {valid_games:,}")
    print(f"   - 跳过棋谱: {skipped_games:,}")
    print(f"   - 总样本数: {total_samples:,}")
    print(f"   - 批次文件数: {batch_idx + 1}")
    print(f"   - 输出目录: {OUTPUT_DIR}/")
    print("\n✅ 训练数据已生成！")


def save_batch(states_list, actions_list, batch_idx):
    """保存一个批次的数据"""
    states = np.concatenate(states_list, axis=0).astype(np.float16)
    actions = np.concatenate(actions_list, axis=0).astype(np.int64)

    states_path = os.path.join(OUTPUT_DIR, f"states_batch_{batch_idx:03d}.npy")
    actions_path = os.path.join(OUTPUT_DIR, f"actions_batch_{batch_idx:03d}.npy")

    np.save(states_path, states)
    np.save(actions_path, actions)

    size_mb = os.path.getsize(states_path) / 1e6
    print(f"   💾 批次 {batch_idx:03d}: {len(states):,} 个样本 ({size_mb:.1f}MB)")

    return batch_idx + 1


# ============================================================
# 第 4 步：数据质量验证
# ============================================================

def validate_data():
    """验证生成的数据是否合理"""
    print("\n🔍 验证数据质量...")
    print("=" * 60)

    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "states_batch_*.npy")))
    if not files:
        print("❌ 没有找到数据文件！")
        return

    # 抽查第一个批次
    states = np.load(files[0], mmap_mode='r')
    actions = np.load(files[0].replace("states", "actions"), mmap_mode='r')

    print(f"   批次文件: {os.path.basename(files[0])}")
    print(f"   样本数: {len(states):,}")
    print(f"   状态形状: {states.shape}")
    print(f"   动作范围: {actions.min()} ~ {actions.max()}")

    # 检查前 3 个样本的棋盘状态
    for i in range(min(3, len(states))):
        s = states[i]
        current = s[0].sum()
        opponent = s[1].sum()
        total = current + opponent
        a = actions[i]
        action_str = "Pass" if a == 361 else f"({a // 19}, {a % 19})"
        print(f"   样本 {i}: 当前方 {current:.0f} 子, 对方 {opponent:.0f} 子, "
              f"总数 {total:.0f}, 标签 {a} ({action_str})")

    # Pass 统计
    pass_count = int((actions == 361).sum())
    print(f"\n   Pass 样本: {pass_count:,} ({pass_count / len(actions) * 100:.2f}%)")

    # 动作分布
    unique_actions = len(np.unique(actions))
    print(f"   唯一动作数: {unique_actions} / 362")

    print("✅ 数据验证完成！")
    print("=" * 60)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("围棋 SGF 棋谱处理工具（sgfmill 版）")
    print("=" * 60)

    # 检查 sgfmill
    try:
        import sgfmill
    except ImportError:
        print("❌ 请先安装 sgfmill:")
        print("   pip install sgfmill")
        exit(1)

    # 第 1 步：解压 zip
    if not extract_all_zips():
        exit(1)

    # 第 2~3 步：解析并保存
    process_all_sgf_files()

    # 第 4 步：验证数据
    validate_data()

    print("\n🎉 全部完成！接下来运行 Strategy_Network_2.py 开始训练")