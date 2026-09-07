import os
import zipfile
import sgf
import numpy as np
from tqdm import tqdm
import pickle
import shutil
import re

# ==================== 配置区 ====================
DOWNLOAD_DIR = "./downloads"
SGF_EXTRACT_DIR = "./sgf_data"
OUTPUT_DIR = "./processed_data"
BOARD_SIZE = 19
BATCH_SIZE = 500  # 每批处理多少个 .sgf 文件
SAVE_INTERVAL = 100000  # 每积累多少样本保存一次


# ===============================================

def extract_all_zips():
    """解压 downloads 文件夹里的所有 .zip 文件"""
    os.makedirs(SGF_EXTRACT_DIR, exist_ok=True)

    zip_files = [f for f in os.listdir(DOWNLOAD_DIR) if f.endswith('.zip')]
    if not zip_files:
        print(f"❌ 在 {DOWNLOAD_DIR} 中没有找到任何 .zip 文件")
        return False

    print(f"找到 {len(zip_files)} 个 .zip 文件")
    for zip_path in zip_files:
        full_path = os.path.join(DOWNLOAD_DIR, zip_path)
        print(f"  解压: {zip_path}")
        try:
            with zipfile.ZipFile(full_path, 'r') as zip_ref:
                zip_ref.extractall(SGF_EXTRACT_DIR)
        except Exception as e:
            print(f"    解压失败: {e}")
    return True


def parse_sgf_file(filepath):
    """使用正则表达式手动解析 SGF 文件，提取落子信息"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return [], []

    # 提取所有落子序列: ;B[ab] 或 ;W[cd]
    moves = re.findall(r';([BW])\[([a-s][a-s])\]', content)

    if not moves:
        return [], []

    states, actions = [], []
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    current_player = 1  # 1=黑, 2=白

    for color, coord in moves:
        row = ord(coord[0]) - ord('a')
        col = ord(coord[1]) - ord('a')

        if row < 0 or row >= BOARD_SIZE or col < 0 or col >= BOARD_SIZE:
            continue

        action = row * BOARD_SIZE + col

        # 构建状态（两个通道），使用 float16 存储
        state = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float16)
        state[0] = (board == current_player)
        state[1] = (board == 3 - current_player)

        states.append(state)
        actions.append(action)

        board[row, col] = current_player
        current_player = 3 - current_player

    return states, actions


def save_batch(states, actions, batch_num):
    """保存一批数据为 .npy 文件"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    states_array = np.array(states, dtype=np.float16)
    actions_array = np.array(actions, dtype=np.int64)

    # 使用三位数字编号，方便排序
    states_path = os.path.join(OUTPUT_DIR, f'states_batch_{batch_num:03d}.npy')
    actions_path = os.path.join(OUTPUT_DIR, f'actions_batch_{batch_num:03d}.npy')

    np.save(states_path, states_array)
    np.save(actions_path, actions_array)

    print(f"  ✅ 已保存批次 {batch_num:03d}: states {states_array.shape}, actions {actions_array.shape}")
    return states_path, actions_path


def process_all_sgf():
    """分批遍历 sgf_data 文件夹，解析所有 .sgf 文件"""
    sgf_files = []
    for root, _, files in os.walk(SGF_EXTRACT_DIR):
        for f in files:
            if f.endswith('.sgf'):
                sgf_files.append(os.path.join(root, f))

    if not sgf_files:
        print(f"❌ 在 {SGF_EXTRACT_DIR} 中没有找到任何 .sgf 文件")
        return

    total_files = len(sgf_files)
    print(f"找到 {total_files} 个 .sgf 文件")
    print(f"每批处理 {BATCH_SIZE} 个文件，共 {(total_files + BATCH_SIZE - 1) // BATCH_SIZE} 批")
    print(f"每 {SAVE_INTERVAL} 个样本保存一次")
    print("-" * 50)

    batch_num = 0
    total_games = 0
    total_samples = 0

    # 用于积累样本的缓冲区
    buffer_states = []
    buffer_actions = []
    buffer_sample_count = 0

    for i in tqdm(range(0, total_files, BATCH_SIZE), desc="总进度"):
        batch_files = sgf_files[i:i + BATCH_SIZE]
        batch_states, batch_actions = [], []
        batch_games = 0

        for filepath in batch_files:
            states, actions = parse_sgf_file(filepath)
            if states:
                batch_states.extend(states)
                batch_actions.extend(actions)
                batch_games += 1

        if batch_states:
            # 将当前批次的数据加入缓冲区
            buffer_states.extend(batch_states)
            buffer_actions.extend(batch_actions)
            buffer_sample_count += len(batch_states)
            total_games += batch_games
            total_samples += len(batch_states)

            # 如果缓冲区达到保存阈值，保存一批
            if buffer_sample_count >= SAVE_INTERVAL:
                batch_num += 1
                save_batch(buffer_states, buffer_actions, batch_num)

                # 清空缓冲区
                buffer_states = []
                buffer_actions = []
                buffer_sample_count = 0

        # 强制释放内存
        del batch_states, batch_actions

    # 保存剩余数据（如果有）
    if buffer_states:
        batch_num += 1
        save_batch(buffer_states, buffer_actions, batch_num)

    print("-" * 50)
    print(f"✅ 处理完成！")
    print(f"   - 共解析 {total_games} 局棋")
    print(f"   - 共提取 {total_samples} 个训练样本")
    print(f"   - 保存为 {batch_num} 个批次文件")
    print(f"   - 文件位置: {OUTPUT_DIR}/")
    print(f"   - 数据格式: float16")


def cleanup():
    """清理临时解压文件夹"""
    if os.path.exists(SGF_EXTRACT_DIR):
        shutil.rmtree(SGF_EXTRACT_DIR)
        print(f"已清理临时文件夹: {SGF_EXTRACT_DIR}")


def main():
    print("=" * 50)
    print("KGS 棋谱本地处理工具 (float16 分批保存)")
    print("=" * 50)

    if not extract_all_zips():
        return

    process_all_sgf()

    # 可选：清理临时文件（如果不需要保留 .sgf 文件，取消注释下一行）
    # cleanup()

    print("\n🎉 全部完成！")


if __name__ == "__main__":
    main()