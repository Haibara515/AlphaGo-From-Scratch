"""
find_sgf_v3.py - 围棋 SGF 棋谱数据处理脚本（价值网络训练数据版）

使用 sgfmill 库正确模拟围棋规则（提子、让子、Pass），
从 downloads/ 文件夹读取 .zip 压缩包，解压后解析 .sgf 棋谱，
为每个棋局位置生成「棋盘状态 + 最终胜负标签」训练数据，
供价值网络（Value Network）训练使用。

数据格式：
    states:  (N, 2, 19, 19)  float16
        通道 0 = 当前行棋方的棋子
        通道 1 = 对方的棋子
    values:  (N,)            float32
        +1.0 = 当前行棋方最终获胜
        -1.0 = 当前行棋方最终失败
         0.0 = 和棋
        （标签随行棋方视角变化：黑方回合用黑方视角，白方回合用白方视角）
    actions: (N,)            int64（可选）
        0~360 表示落子位置（row * 19 + col），361 表示 Pass

与 find_sgf_v2.py 的区别：
    - 输出目录为 ./processed_data_value（互不干扰）
    - 额外生成 values_batch_*.npy 胜负标签
    - 按棋局结果过滤：结果未知的棋谱直接跳过

用法：
    pip install sgfmill
    python find_sgf_v3.py
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
DOWNLOAD_DIR = "./downloads"          # zip 压缩包所在目录
SGF_EXTRACT_DIR = "./sgf_data"        # 解压后的 SGF 文件目录（临时）
OUTPUT_DIR = "./processed_data_value" # 价值网络训练数据输出目录（与 v2 不同）
BOARD_SIZE = 19                       # 棋盘大小
BATCH_SAVE_SIZE = 10000               # 每个 npy 批次保存的样本数
MIN_MOVES = 10                        # 最少步数（过滤过短的棋谱）
MAX_MOVES = 400                       # 最大步数（过滤异常长的棋谱）
SAVE_ACTIONS = True                   # 是否同时保存 actions_batch_*.npy（可选）

# 胜负标签取值（当前行棋方视角）
VALUE_WIN = 1.0    # 当前行棋方最终获胜
VALUE_LOSS = -1.0  # 当前行棋方最终失败
VALUE_DRAW = 0.0   # 和棋（罕见）


# ============================================================
# 第 1 步：解压所有 zip 文件
# ============================================================

def extract_all_zips():
    """解压 downloads 文件夹里的所有 .zip 文件到 sgf_data/（与 find_sgf_v2.py 一致）"""
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

def get_game_winner(game):
    """
    从 SGF 元数据中读取棋局最终胜者（不做视角转换）。

    Args:
        game: sgfmill.Sgf_game 对象

    Returns:
        'b' = 黑胜，'w' = 白胜，'draw' = 和棋，None = 结果未知（跳过）

    说明：sgfmill 的 get_winner() 对「和棋」和「结果未知」都返回 None，
    这里额外检查 RE 属性把真正的和棋（RE 为 0 / Draw / Jigo）区分出来；
    每手棋的 ±1 标签由调用方按「当前行棋方视角」转换。
    """
    winner = game.get_winner()
    if winner in ('b', 'w'):
        return winner

    # get_winner() 返回 None：可能是和棋，也可能是结果未知
    re_prop = game.get_root().get('RE')
    if re_prop is not None:
        re_text = str(re_prop).strip().lower()
        if re_text.startswith('0') or re_text.startswith('draw') or re_text.startswith('jigo'):
            return 'draw'
    return None


def extract_board_state(board, current_player):
    """
    从 sgfmill.Board 提取 2 通道棋盘状态（与 find_sgf_v2.py 一致）。

    Args:
        board: sgfmill.Board 对象
        current_player: 当前行棋方（'b' 或 'w'）

    Returns:
        state: (2, BOARD_SIZE, BOARD_SIZE) float16
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


def sgf_to_training_data(sgf_path):
    """
    解析单个 SGF 文件，提取价值网络训练样本。

    使用 sgfmill 的 sgf_moves 自动处理：
    - 提子（captures）
    - 让子棋（setup stones）
    - Pass（move is None）
    - 分支变化（取主分支）

    Returns:
        成功: (states, values, actions, None)
            states:  (n, 2, 19, 19) float16
            values:  (n,) float32，每手按当前行棋方视角：+1.0 胜 / -1.0 负 / 0.0 和
            actions: (n,) int64，0~360 落子位置，361 Pass
        失败: (None, None, None, 原因字符串)
    """
    try:
        # 1. 读取 SGF 文件
        with open(sgf_path, 'rb') as f:
            game = sgf.Sgf_game.from_bytes(f.read())

        # 2. 检查棋盘大小
        if game.get_size() != BOARD_SIZE:
            return None, None, None, '棋盘大小不符'

        # 3. 读取棋局胜者；结果未知的棋谱直接跳过
        #    （胜负/和棋的 ±1 标签在下面按每手棋的当前行棋方转换）
        winner = get_game_winner(game)
        if winner is None:
            return None, None, None, '结果未知'

        # 4. 用 sgf_moves 模拟对局（自动处理提子、让子、Pass）
        board, plays = sgf_moves.get_setup_and_moves(game)

        # 5. 过滤步数太少的棋谱
        if len(plays) < MIN_MOVES:
            return None, None, None, '步数过少'

        # 6. 限制最大步数
        if len(plays) > MAX_MOVES:
            plays = plays[:MAX_MOVES]

        # 7. 遍历每一步，生成训练样本
        states_list = []
        values_list = []
        actions_list = []

        for color, move in plays:
            # 提取当前棋盘状态（在落子之前）
            state = extract_board_state(board, color)

            # 胜负标签（当前行棋方视角）：
            #   当前行棋方 == 胜者 -> +1.0；否则 -> -1.0；和棋 -> 0.0
            if winner in ('b', 'w'):
                value = VALUE_WIN if color == winner else VALUE_LOSS
            else:
                value = VALUE_DRAW
            values_list.append(value)

            # 记录动作标签（供策略头或复盘使用；价值网络本身只用到 states/values）
            if move is None:
                action = BOARD_SIZE * BOARD_SIZE  # 361 = Pass
            else:
                row, col = move
                action = row * BOARD_SIZE + col
            actions_list.append(action)

            states_list.append(state)

            # 在棋盘上落子（sgfmill 自动处理提子）
            if move is not None:
                row, col = move
                board.play(row, col, color)

        # 8. 转换为 numpy 数组
        states = np.array(states_list, dtype=np.float16)
        values = np.array(values_list, dtype=np.float32)
        actions = np.array(actions_list, dtype=np.int64)

        return states, values, actions, None

    except Exception:
        return None, None, None, '解析失败'


# ============================================================
# 第 3 步：批量处理所有 SGF 文件
# ============================================================

def process_all_sgf_files():
    """遍历 sgf_data/ 下所有 .sgf 文件，解析并保存价值网络训练数据"""

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

    # 3. 批量处理（内存友好：按游戏累积，攒够 BATCH_SAVE_SIZE 再落盘）
    all_states = []
    all_values = []
    all_actions = []
    batch_idx = 0
    total_samples = 0
    valid_games = 0
    skipped_games = 0
    skip_reasons = {}          # 跳过原因统计
    current_player_win_samples = 0   # 当前行棋方获胜样本数（label = +1.0）
    current_player_loss_samples = 0  # 当前行棋方失败样本数（label = -1.0）
    draw_samples = 0                 # 和棋样本数（label = 0.0）

    print(f"\n🔄 开始解析 SGF 棋谱...")
    print("=" * 60)

    for i, sgf_path in enumerate(sgf_files):
        states, values, actions, reason = sgf_to_training_data(sgf_path)

        if states is None:
            skipped_games += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue

        all_states.append(states)
        all_values.append(values)
        all_actions.append(actions)
        total_samples += len(states)
        valid_games += 1

        # 累计标签分布（按样本统计，均为当前行棋方视角）
        current_player_win_samples += int((values == VALUE_WIN).sum())
        current_player_loss_samples += int((values == VALUE_LOSS).sum())
        draw_samples += int((values == VALUE_DRAW).sum())

        # 进度显示（每 500 个文件）
        if (i + 1) % 500 == 0:
            print(f"   进度: {i + 1}/{len(sgf_files)} | "
                  f"有效棋谱: {valid_games} | "
                  f"累计样本: {total_samples:,} | "
                  f"跳过: {skipped_games}")

        # 达到保存阈值时写入 npy 文件（整局不拆分）
        current_buffer_size = sum(len(s) for s in all_states)
        if current_buffer_size >= BATCH_SAVE_SIZE:
            batch_idx = save_batch(all_states, all_values, all_actions, batch_idx)
            all_states = []
            all_values = []
            all_actions = []

    # 4. 保存剩余数据
    if all_states:
        save_batch(all_states, all_values, all_actions, batch_idx)

    # 5. 输出统计
    print("=" * 60)
    print(f"\n📊 处理完成！")
    print(f"   - SGF 文件总数: {len(sgf_files):,}")
    print(f"   - 有效棋谱: {valid_games:,}")
    print(f"   - 跳过棋谱: {skipped_games:,}")
    for reason, count in sorted(skip_reasons.items(), key=lambda kv: -kv[1]):
        print(f"       * {reason}: {count:,}")
    print(f"   - 总样本数: {total_samples:,}")
    print(f"   - 批次文件数: {batch_idx + 1}")

    if total_samples > 0:
        print(f"\n   📈 标签分布（当前行棋方视角，按样本）:")
        print(f"       - 当前方获胜 (+1.0): {current_player_win_samples:,} "
              f"({current_player_win_samples / total_samples * 100:.2f}%)")
        print(f"       - 当前方失败 (-1.0): {current_player_loss_samples:,} "
              f"({current_player_loss_samples / total_samples * 100:.2f}%)")
        print(f"       - 和棋 (0.0): {draw_samples:,} "
              f"({draw_samples / total_samples * 100:.2f}%)")

    print(f"   - 输出目录: {OUTPUT_DIR}/")
    print("\n✅ 价值网络训练数据已生成！")


def save_batch(states_list, values_list, actions_list, batch_idx):
    """保存一个批次的数据（states / values / actions 三个文件）"""
    states = np.concatenate(states_list, axis=0).astype(np.float16)
    values = np.concatenate(values_list, axis=0).astype(np.float32)

    states_path = os.path.join(OUTPUT_DIR, f"states_batch_{batch_idx:03d}.npy")
    values_path = os.path.join(OUTPUT_DIR, f"values_batch_{batch_idx:03d}.npy")
    np.save(states_path, states)
    np.save(values_path, values)

    if SAVE_ACTIONS:
        actions = np.concatenate(actions_list, axis=0).astype(np.int64)
        actions_path = os.path.join(OUTPUT_DIR, f"actions_batch_{batch_idx:03d}.npy")
        np.save(actions_path, actions)

    size_mb = os.path.getsize(states_path) / 1e6
    print(f"   💾 批次 {batch_idx:03d}: {len(states):,} 个样本 ({size_mb:.1f}MB)")

    return batch_idx + 1


# ============================================================
# 第 4 步：数据质量验证
# ============================================================

def validate_data():
    """验证生成的数据是否合理（形状、取值范围、标签分布）"""
    print("\n🔍 验证数据质量...")
    print("=" * 60)

    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "states_batch_*.npy")))
    if not files:
        print("❌ 没有找到数据文件！")
        return

    # 抽查第一个批次
    states = np.load(files[0], mmap_mode='r')
    values = np.load(files[0].replace("states", "values"), mmap_mode='r')

    print(f"   批次文件: {os.path.basename(files[0])}")
    print(f"   样本数: {len(states):,}")
    print(f"   状态形状: {states.shape} | dtype: {states.dtype}")
    print(f"   标签形状: {values.shape} | dtype: {values.dtype}")
    print(f"   标签范围: {values.min()} ~ {values.max()}")

    # 校验标签取值范围：只能出现 {-1.0, 0.0, +1.0}
    unique_values = set(np.unique(values).tolist())
    if not unique_values.issubset({VALUE_LOSS, VALUE_DRAW, VALUE_WIN}):
        print(f"❌ 标签取值异常: {sorted(unique_values)}（应为 -1.0 / 0.0 / +1.0）")
    else:
        print(f"   标签取值: {sorted(unique_values)} ✓（仅含 -1.0 / 0.0 / +1.0）")

    # 检查前 3 个样本的棋盘状态与标签
    for i in range(min(3, len(states))):
        s = states[i]
        current = s[0].sum()
        opponent = s[1].sum()
        total = current + opponent
        v = values[i]
        if v == VALUE_WIN:
            result = "当前方胜"
        elif v == VALUE_LOSS:
            result = "当前方负"
        else:
            result = "和棋"
        print(f"   样本 {i}: 当前方 {current:.0f} 子, 对方 {opponent:.0f} 子, "
              f"总数 {total:.0f}, 标签 {v:.1f} ({result})")

    # 标签分布（第一个批次内，当前行棋方视角）
    win_count = int((values == VALUE_WIN).sum())
    loss_count = int((values == VALUE_LOSS).sum())
    draw_count = int((values == VALUE_DRAW).sum())
    n = len(values)
    print(f"\n   标签分布(本批次): 当前方胜 {win_count:,} ({win_count / n * 100:.2f}%), "
          f"当前方负 {loss_count:,} ({loss_count / n * 100:.2f}%), "
          f"和棋 {draw_count:,} ({draw_count / n * 100:.2f}%)")

    # 动作文件（可选）
    if SAVE_ACTIONS:
        actions_path = files[0].replace("states", "actions")
        if os.path.exists(actions_path):
            actions = np.load(actions_path, mmap_mode='r')
            print(f"   动作形状: {actions.shape} | dtype: {actions.dtype}")
            print(f"   动作范围: {actions.min()} ~ {actions.max()}")
            pass_count = int((actions == BOARD_SIZE * BOARD_SIZE).sum())
            print(f"   Pass 样本: {pass_count:,} ({pass_count / len(actions) * 100:.2f}%)")

    print("✅ 数据验证完成！")
    print("=" * 60)


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("围棋 SGF 棋谱处理工具（价值网络数据版）")
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

    print("\n🎉 全部完成！生成的数据可用于价值网络训练")
