"""
self_play.py - 围棋 AI 自我对弈训练数据生成脚本

用 MCTS（策略网络 + 价值网络）自我对弈，记录每一手的：
    states:     (N, 2, 19, 19) float32  当前行棋方视角棋盘
    mcts_probs: (N, 362)       float32  MCTS 访问次数概率（索引 361 = Pass）
    values:     (N,)           float32  终局结果（当前行棋方视角：+1 胜 / -1 负 / 0 和）

温度策略：前 30 手 temperature=1.0（充分探索），之后 0.1（偏向利用）。
劫规则：对局级维护局面指纹历史（与 GoEngine 的位置超劫一致），
        每一步搜索都基于完整历史做合法点过滤，保证不会走出非法打劫落子。

用法：
    python self_play.py
"""

import os
import glob
import numpy as np


# ============================================================
# 配置
# ============================================================
BOARD_SIZE = 19
N_BOARD = BOARD_SIZE * BOARD_SIZE  # 361
PASS_MOVE = N_BOARD                # 361
MAX_MOVES = 500                    # 单局最大手数上限
BATCH_SAVE_SIZE = 10000            # 每个 npy 批次保存的样本数
OUTPUT_DIR = "./self_play_data"    # 输出目录


# ============================================================
# 工具函数
# ============================================================

def sample_move(probs, temperature=1.0):
    """
    按概率分布采样动作。

    Args:
        probs: 362 维概率数组（索引 361 = Pass）
        temperature: 采样温度；<=0 表示贪心（取概率最大者）

    Returns:
        move: 0~360 或 361
    """
    if temperature <= 0:
        return int(np.argmax(probs))

    # 温度缩放：p^(1/T) 后归一化
    scores = probs ** (1.0 / temperature)
    total = scores.sum()
    if total <= 0:
        return int(np.random.choice(len(probs)))  # 数值异常兜底
    p = scores / total
    return int(np.random.choice(len(probs), p=p))


def load_models(policy_path, value_path, device):
    """
    加载策略网络与价值网络，并设为 eval 模式、移动到目标设备。

    Args:
        policy_path: 策略网络检查点路径
        value_path: 价值网络检查点路径
        device: 'cpu' 或 'cuda'

    Returns:
        (policy_model, value_model)
    """
    import torch
    from src.policy_network import PolicyNetworkAlphaGoLee
    from src.value_network import ValueNetworkAlphaGoLee

    def _load(model_class, path, name):
        model = model_class(input_channels=2)
        checkpoint = torch.load(path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.eval().to(device)
        print(f"✅ {name}已加载: {path}")
        return model

    policy_model = _load(PolicyNetworkAlphaGoLee, policy_path, "策略网络")
    value_model = _load(ValueNetworkAlphaGoLee, value_path, "价值网络")
    return policy_model, value_model


# ============================================================
# 自我对弈
# ============================================================

def self_play_one_game(policy_model, value_model, device,
                       num_simulations=200, mcts=None):
    """
    用 MCTS 自我对弈一局。

    Args:
        policy_model: 策略网络（eval 模式）
        value_model: 价值网络（eval 模式）
        device: 'cpu' 或 'cuda'
        num_simulations: 每步 MCTS 模拟次数
        mcts: 可选，复用的 MCTS 实例（避免每局重建）

    Returns:
        states: (N, 2, 19, 19) float32
        mcts_probs: (N, 362) float32（访问次数概率，索引 361 = Pass）
        values: (N,) float32（当前行棋方视角：+1 胜 / -1 负 / 0 和）
    """
    from src.mcts import (
        MCTS,
        PASS_MOVE,
        apply_move,
        board_to_state,
        get_position_fingerprint,
        is_game_over,
        score_game,
    )

    if mcts is None:
        mcts = MCTS(policy_model, value_model,
                    device=device, num_simulations=num_simulations)

    board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    current_player = 1
    passes = 0
    move_count = 0

    # 对局级劫历史（与 GoEngine 的位置超劫一致）：从空盘开始，
    # 每手棋（含提子）产生的新局面都加入；Pass 不改变棋盘，不新增指纹
    game_history = {get_position_fingerprint(board)}

    states = []
    probs_list = []
    players = []

    while not is_game_over(board, passes, move_count):
        # 温度策略：前 30 手充分探索，之后偏向利用
        temperature = 1.0 if move_count < 30 else 0.1

        # 用完整对局历史做 MCTS 搜索（保证合法点过滤与引擎一致）
        probs = mcts.search(board, current_player, game_history=game_history)

        states.append(board_to_state(board, current_player))
        probs_list.append(probs.astype(np.float32))
        players.append(current_player)

        move = sample_move(probs, temperature=temperature)
        if move == PASS_MOVE:
            passes += 1
        else:
            passes = 0
            row, col = divmod(move, BOARD_SIZE)
            board, _ = apply_move(board, row, col, current_player)
            game_history.add(get_position_fingerprint(board))  # 新局面加入历史

        move_count += 1
        current_player = 3 - current_player

    # 终局结果 -> 每手价值标签（当前行棋方视角）
    winner = score_game(board)
    values = np.array(
        [1.0 if winner == p else -1.0 if winner != 0 else 0.0 for p in players],
        dtype=np.float32,
    )

    return (
        np.stack(states).astype(np.float32),
        np.stack(probs_list).astype(np.float32),
        values,
    )


def _save_batch(states_list, probs_list, values_list, batch_idx, output_dir):
    """保存一个批次的数据（states / probs / values 三个文件）。"""
    states = np.concatenate(states_list, axis=0).astype(np.float32)
    probs = np.concatenate(probs_list, axis=0).astype(np.float32)
    values = np.concatenate(values_list, axis=0).astype(np.float32)

    np.save(os.path.join(output_dir, f"states_batch_{batch_idx:03d}.npy"), states)
    np.save(os.path.join(output_dir, f"probs_batch_{batch_idx:03d}.npy"), probs)
    np.save(os.path.join(output_dir, f"values_batch_{batch_idx:03d}.npy"), values)

    print(f"   💾 批次 {batch_idx:03d}: {len(states):,} 个样本")
    return batch_idx + 1


def run_self_play(num_games, policy_model, value_model, device,
                  num_simulations=200, output_dir=OUTPUT_DIR,
                  batch_save_size=BATCH_SAVE_SIZE):
    """
    运行多局自我对弈，并把训练数据分批保存到磁盘。

    Args:
        num_games: 对局数
        policy_model / value_model: 已加载的模型
        device: 'cpu' 或 'cuda'
        num_simulations: 每步 MCTS 模拟次数
        output_dir: 输出目录（默认 ./self_play_data）
        batch_save_size: 每批样本数上限（整局不拆分）
    """
    from src.mcts import MCTS

    os.makedirs(output_dir, exist_ok=True)
    # 清理旧数据
    old_files = glob.glob(os.path.join(output_dir, "*.npy"))
    if old_files:
        print(f"🗑️ 清理旧自我对弈数据 {len(old_files)} 个文件...")
        for f in old_files:
            os.remove(f)

    # MCTS 实例复用（模型已在内部移动到 device）
    mcts = MCTS(policy_model, value_model,
                device=device, num_simulations=num_simulations)

    all_states = []
    all_probs = []
    all_values = []
    batch_idx = 0
    total_samples = 0

    print(f"\n🔄 开始自我对弈: {num_games} 局 | 每步模拟 {num_simulations} 次")
    print("=" * 60)

    for game_idx in range(num_games):
        states, probs, values = self_play_one_game(
            policy_model, value_model, device,
            num_simulations=num_simulations, mcts=mcts,
        )
        all_states.append(states)
        all_probs.append(probs)
        all_values.append(values)
        total_samples += len(states)

        print(f"   🎮 第 {game_idx + 1}/{num_games} 局完成: "
              f"{len(states)} 手 | 累计样本: {total_samples:,}")

        # 达到保存阈值时写入 npy 文件（整局不拆分）
        if sum(len(s) for s in all_states) >= batch_save_size:
            batch_idx = _save_batch(all_states, all_probs, all_values, batch_idx, output_dir)
            all_states, all_probs, all_values = [], [], []

    # 保存剩余数据
    if all_states:
        batch_idx = _save_batch(all_states, all_probs, all_values, batch_idx, output_dir)

    print("=" * 60)
    print(f"✅ 自我对弈完成！共 {num_games} 局，{total_samples:,} 个样本，"
          f"保存到 {output_dir}/")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import torch

    # ==================== 配置（可按需修改） ====================
    POLICY_MODEL_PATH = "./checkpoints/best_model.pth"        # 策略网络
    VALUE_MODEL_PATH = "./checkpoints_value/best_model.pth"   # 价值网络
    NUM_GAMES = 10                                            # 对局数
    NUM_SIMULATIONS = 200    # CPU 可调小（如 50），GPU 可调大（如 400）
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # ============================================================

    print("=" * 60)
    print("围棋 AI 自我对弈训练数据生成")
    print(f"  设备: {DEVICE} | 每局模拟: {NUM_SIMULATIONS} | 局数: {NUM_GAMES}")
    print("=" * 60)

    try:
        policy_model, value_model = load_models(
            POLICY_MODEL_PATH, VALUE_MODEL_PATH, DEVICE
        )
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        exit(1)

    run_self_play(
        NUM_GAMES,
        policy_model,
        value_model,
        DEVICE,
        num_simulations=NUM_SIMULATIONS,
        output_dir=OUTPUT_DIR,
        batch_save_size=BATCH_SAVE_SIZE,
    )
