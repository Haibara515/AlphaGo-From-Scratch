"""
MCTS.py - 围棋 AI 蒙特卡洛树搜索（AlphaGo 风格）

使用策略网络（Policy Network）提供先验概率、价值网络（Value Network）提供局面评估，
通过 PUCT 公式完成选择（Selection）、扩展（Expansion）、评估（Evaluation）、
回传（Backpropagation）四个步骤，返回每个动作的访问次数概率分布。

模型约定：
    - 策略网络输出 [batch, 362] logits（0~360 棋盘交点，361 = Pass）
    - 价值网络输出 [batch, 1] 标量（线性输出时用 tanh 压缩到 [-1, +1]）
    - 棋盘表示：19×19，0=空，1=黑，2=白

用法：
    from src.mcts import MCTS, board_to_state, get_legal_moves

    mcts = MCTS(policy_model, value_model, device='cuda', num_simulations=800)
    move = mcts.select_move(board, current_player=1, temperature=0.0)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 常量
# ============================================================

BOARD_SIZE = 19                 # 棋盘大小
N_BOARD = BOARD_SIZE * BOARD_SIZE  # 361 个棋盘交点
PASS_MOVE = N_BOARD             # 361：Pass 动作
KOMI = 7.5                      # 白方贴目（中国规则）
MAX_MOVES = 500                 # 单局最大手数上限
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


# ============================================================
# 棋盘工具函数（纯规则，不依赖网络，可被 Go_game_2.py 复用）
# ============================================================

def board_to_state(board, current_player):
    """
    将 19×19 棋盘转换为模型输入格式（2 通道，当前行棋方视角）。

    Args:
        board: 19×19 棋盘（0=空, 1=黑, 2=白），列表或 numpy 数组
        current_player: 当前行棋方（1=黑, 2=白）

    Returns:
        state: (2, 19, 19) float32 数组
            通道 0 = 当前行棋方的棋子
            通道 1 = 对方的棋子
    """
    board = np.asarray(board)
    state = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    state[0] = (board == current_player)
    state[1] = (board == 3 - current_player)
    return state


def _collect_group(board, row, col):
    """返回 (row, col) 所在同色连通棋串的全部坐标（不含空格）。board 为 list of lists。"""
    color = board[row][col]
    if color not in (1, 2):
        return []
    group = []
    seen = set()
    stack = [(row, col)]
    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        group.append((r, c))
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE \
                    and board[nr][nc] == color and (nr, nc) not in seen:
                stack.append((nr, nc))
    return group


def _group_liberties(board, group):
    """返回棋串所有气（上下左右相邻空点）的集合。"""
    liberties = set()
    for r, c in group:
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE and board[nr][nc] == 0:
                liberties.add((nr, nc))
    return liberties


def _apply_move(board, row, col, player):
    """
    在 board 的副本上落子并提掉无气敌串。

    Args:
        board: 当前棋盘（list of lists，不会被修改）
        row, col: 落子坐标
        player: 落子方（1=黑, 2=白）

    Returns:
        (新棋盘, 提子数)
    """
    b = [list(r) for r in board]
    b[row][col] = player
    opponent = 3 - player
    captured = 0
    for dr, dc in DIRECTIONS:
        nr, nc = row + dr, col + dc
        if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
            continue
        if b[nr][nc] != opponent:
            continue
        group = _collect_group(b, nr, nc)
        if _group_liberties(b, group):
            continue  # 该敌串还有气，不构成提子
        for gr, gc in group:
            b[gr][gc] = 0
        captured += len(group)
    return b, captured


def apply_move(board, row, col, player):
    """公开接口：在 board 副本上落子并提掉无气敌串，返回 (新棋盘, 提子数)。"""
    return _apply_move(board, row, col, player)


def get_position_fingerprint(board):
    """
    生成棋盘局面指纹（用于位置超劫判重）。

    说明：指纹只包含棋盘（不含行棋方），与 Go_game_2.py 中 GoEngine 的
    位置超劫判定完全一致。不要改成 (board, current_player) 形式：那会让
    AI 的劫判定比对局引擎更宽松，AI 仍可能走出被引擎拒绝的“非法”落子。
    """
    return tuple(tuple(row) for row in board)


# 内部兼容别名
_position_key = get_position_fingerprint


def get_legal_moves(board, current_player, history=None):
    """
    返回当前行棋方的全部合法动作（0~360 落子点 + 361 Pass）。

    规则检查：
    - 已占用点非法；
    - 自杀非法（未提子且己方棋串无气）；
    - 打劫：若传入 history（本分支出现过的局面指纹集合，含当前局面），
      任何会复现历史局面的落子都会被禁止（位置超劫，比简单劫更严格）；
    - Pass（361）永远合法。

    Args:
        board: 当前棋盘（list of lists 或 numpy 数组）
        current_player: 当前行棋方（1=黑, 2=白）
        history: 可选，set 类型，本分支出现过的局面指纹集合

    Returns:
        List[int]：合法动作列表
    """
    board = [list(r) for r in board]
    moves = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] != 0:
                continue
            new_board, captured = _apply_move(board, row, col, current_player)

            # 自杀检查：未提子且己方无气
            if captured == 0:
                group = _collect_group(new_board, row, col)
                if not _group_liberties(new_board, group):
                    continue

            # 打劫检查（位置超劫）
            if history is not None:
                if _position_key(new_board) in history:
                    continue

            moves.append(row * BOARD_SIZE + col)

    # Pass 永远合法
    moves.append(PASS_MOVE)
    return moves


def is_game_over(board, consecutive_passes=0, move_count=0, max_moves=MAX_MOVES):
    """
    判断对局是否结束。

    Args:
        board: 当前棋盘
        consecutive_passes: 连续停手次数（0/1/2）
        move_count: 已下手数
        max_moves: 最大手数上限

    Returns:
        bool
    """
    if consecutive_passes >= 2:
        return True  # 双方连续停手
    if move_count >= max_moves:
        return True  # 达到最大手数
    # 棋盘填满（无空点）也视为结束
    for row in board:
        for stone in row:
            if stone == 0:
                return False
    return True


def score_game(board, komi=KOMI):
    """
    中国规则数子法计分：领地 + 棋盘上的活子，白方加贴目。

    Args:
        board: 终局棋盘
        komi: 白方贴目（默认 7.5）

    Returns:
        1 = 黑胜，2 = 白胜，0 = 平局
    """
    board = [list(r) for r in board]

    # 1) 数空：空点区域只被黑包围算黑地，只被白包围算白地，否则公空
    visited = [[False] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    black_territory = 0
    white_territory = 0

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r][c] != 0 or visited[r][c]:
                continue
            region = []
            stack = [(r, c)]
            visited[r][c] = True
            while stack:
                cr, cc = stack.pop()
                region.append((cr, cc))
                for dr, dc in DIRECTIONS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE \
                            and not visited[nr][nc] and board[nr][nc] == 0:
                        visited[nr][nc] = True
                        stack.append((nr, nc))

            has_black = False
            has_white = False
            for cr, cc in region:
                for dr, dc in DIRECTIONS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE:
                        if board[nr][nc] == 1:
                            has_black = True
                        elif board[nr][nc] == 2:
                            has_white = True

            if has_black and not has_white:
                black_territory += len(region)
            elif has_white and not has_black:
                white_territory += len(region)
            # 公空：双方都不计

    # 2) 数子：棋盘上的活子（简单计分，不人工标记死子）
    black_stones = sum(row.count(1) for row in board)
    white_stones = sum(row.count(2) for row in board)

    black = black_territory + black_stones
    white = white_territory + white_stones + komi

    if black > white:
        return 1
    if white > black:
        return 2
    return 0


# ============================================================
# MCTS 节点
# ============================================================

class Node:
    """MCTS 搜索树节点（字典式字段，便于直接访问）。"""

    __slots__ = (
        'parent', 'children', 'N', 'W', 'P',
        'board', 'current_player', 'pos_key', 'history',
        'consecutive_passes', 'move_count', 'last_move',
    )

    def __init__(self, parent=None, prior_prob=0.0, board=None, current_player=None,
                 pos_key=None, history=None, consecutive_passes=0, move_count=0,
                 last_move=None):
        self.parent = parent
        self.children = {}                       # move -> Node
        self.N = 0                               # 访问次数
        self.W = 0.0                             # 累计价值（当前节点行棋方视角）
        self.P = prior_prob                      # 先验概率（来自策略网络）
        self.board = board                       # 该节点的棋盘（list of lists）
        self.current_player = current_player     # 该节点轮到谁走
        self.pos_key = pos_key                   # 局面指纹
        self.history = history                   # 本分支出现过的局面指纹集合（含本节点）
        self.consecutive_passes = consecutive_passes  # 连续停手计数
        self.move_count = move_count             # 已下手数
        self.last_move = last_move               # 到达本节点所下的动作（361=Pass）


# ============================================================
# MCTS 类
# ============================================================

class MCTS:
    """
    AlphaGo 风格的蒙特卡洛树搜索。

    Args:
        policy_model: PolicyNetworkAlphaGoLee 实例（已加载、eval 模式）
        value_model: ValueNetworkAlphaGoLee 实例（已加载、eval 模式）
        device: 'cpu' 或 'cuda'（请求 cuda 但不可用时自动回退到 cpu）
        num_simulations: 每步搜索的模拟次数（AlphaGo 用 1600，单卡 400~800 即可）
        c_puct: 探索常数（AlphaGo 用 5.0，1.0~3.0 效果也不错）
    """

    def __init__(self, policy_model, value_model, device='cpu',
                 num_simulations=800, c_puct=1.0):
        # GPU 可用性回退：请求 cuda 但机器没有可用 GPU 时自动改用 CPU
        if isinstance(device, str) and device.startswith('cuda') \
                and not torch.cuda.is_available():
            print(f"⚠️ 警告: 请求的设备 {device} 不可用，回退到 CPU")
            device = 'cpu'

        self.policy_model = policy_model
        self.value_model = value_model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct

        # 确保模型处于评估模式，并显式移动到目标设备（CPU/GPU 均可）
        if self.policy_model is not None:
            self.policy_model.eval()
            self.policy_model.to(self.device)
        if self.value_model is not None:
            self.value_model.eval()
            self.value_model.to(self.device)

        print(f"MCTS 使用设备: {self.device}")

    # ---------------- 对外接口 ----------------

    def search(self, board, current_player, game_history=None):
        """
        从指定局面开始执行 MCTS 搜索。

        Args:
            board: 19×19 棋盘（0=空, 1=黑, 2=白）
            current_player: 当前行棋方（1=黑, 2=白）
            game_history: 可选，对局级局面指纹集合（含当前局面之前的所有局面）。
                传入后，搜索树会把它作为劫判定的基础历史，避免走出
                复现整局更早局面的“非法”落子（与 GoEngine 的位置超劫一致）。

        Returns:
            probs: 362 维 float32 数组，每个动作的访问次数占比（索引 361 = Pass）
        """
        board_list = [list(r) for r in board]
        root_key = _position_key(board_list)
        # 根节点历史 = 对局级历史（若有）∪ 当前局面本身
        root_history = set(game_history) if game_history is not None else set()
        root_history.add(root_key)
        root = Node(
            board=board_list,
            current_player=current_player,
            pos_key=root_key,
            history=root_history,
            consecutive_passes=0,
            move_count=0,
        )

        for _ in range(self.num_simulations):
            node = root

            # 1) 选择：沿 PUCT 分数最高的子节点下降，直到叶节点
            while node.children:
                move = self._select(node)
                node = node.children[move]

            # 2) 叶节点处理：终局用真实结果，否则扩展并评估
            if is_game_over(node.board, node.consecutive_passes, node.move_count):
                winner = score_game(node.board)
                if winner == 0:
                    value = 0.0
                else:
                    value = 1.0 if winner == node.current_player else -1.0
            else:
                self._expand(node)
                if node.children:
                    value = self._evaluate(node.board, node.current_player)
                else:
                    value = 0.0  # 防御性兜底（Pass 永远合法，正常不会走到）

            # 3) 回传：更新路径上所有节点的 N 和 W
            self._backpropagate(node, value)

        # 由根节点子节点的访问次数生成动作概率
        probs = np.zeros(N_BOARD + 1, dtype=np.float32)
        total_visits = sum(child.N for child in root.children.values())
        if total_visits > 0:
            for move, child in root.children.items():
                probs[move] = child.N / total_visits
        else:
            # 兜底：均匀分布到合法动作（例如根节点本身就是终局）
            legal = get_legal_moves(board_list, current_player, root.history)
            for move in legal:
                probs[move] = 1.0 / len(legal)
        return probs

    def select_move(self, board, current_player, temperature=0.0, game_history=None):
        """
        执行搜索并选择最终动作。

        Args:
            board: 19×19 棋盘
            current_player: 当前行棋方
            temperature: 0.0 = 贪心（选访问次数最多的动作）；
                         1.0 = 按访问次数分布采样（可加温度缩放）
            game_history: 可选，对局级局面指纹集合，透传给 search 用于劫判定

        Returns:
            move: 0~360（棋盘交点）或 361（Pass）
        """
        probs = self.search(board, current_player, game_history=game_history)

        if temperature <= 0:
            return int(np.argmax(probs))

        # 温度采样：p^(1/T) 后归一化
        scores = probs ** (1.0 / temperature)
        total = scores.sum()
        if total <= 0:
            legal = get_legal_moves(board, current_player)
            return int(np.random.choice(legal))
        p = scores / total
        return int(np.random.choice(N_BOARD + 1, p=p))

    # ---------------- 四个搜索步骤 ----------------

    def _select(self, node):
        """
        选择（Selection）：用 PUCT 公式选子。

        a* = argmax_a [ Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a)) ]

        约定：子节点的 W/N 是「对方视角」的价值，取负号才是本节点的价值。
        """
        best_move = None
        best_score = -float('inf')
        sqrt_parent = node.N ** 0.5 if node.N > 0 else 1.0

        for move, child in node.children.items():
            if child.N > 0:
                q = -child.W / child.N  # 对方视角 -> 本节点视角
            else:
                q = 0.0
            u = self.c_puct * child.P * sqrt_parent / (1 + child.N)
            score = q + u
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _expand(self, node):
        """
        扩展（Expansion）：用策略网络为叶节点生成子节点（合法动作 + 先验概率）。
        """
        legal_moves = get_legal_moves(node.board, node.current_player, node.history)

        # 策略网络推理（不计算梯度）
        state = torch.tensor(
            board_to_state(node.board, node.current_player),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy_model(state)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # [362]

        # 只保留合法动作，重新归一化（非法动作概率归零）
        legal_probs = {}
        for move in legal_moves:
            legal_probs[move] = float(probs[move])
        total = sum(legal_probs.values())
        if total <= 0:
            # 数值异常兜底：均匀分布
            for move in legal_moves:
                legal_probs[move] = 1.0 / len(legal_moves)
        else:
            for move in legal_moves:
                legal_probs[move] /= total

        # 为每个合法动作创建子节点
        for move, prior in legal_probs.items():
            if move == PASS_MOVE:
                child_board = [list(r) for r in node.board]
                child_passes = node.consecutive_passes + 1
            else:
                row, col = divmod(move, BOARD_SIZE)
                child_board, _ = _apply_move(node.board, row, col, node.current_player)
                child_passes = 0

            child_key = _position_key(child_board)
            child_history = set(node.history)
            child_history.add(child_key)

            child = Node(
                parent=node,
                prior_prob=prior,
                board=child_board,
                current_player=3 - node.current_player,  # 轮换行棋方
                pos_key=child_key,
                history=child_history,
                consecutive_passes=child_passes,
                move_count=node.move_count + 1,
                last_move=move,
            )
            node.children[move] = child

    def _evaluate(self, board, current_player):
        """
        评估（Evaluation）：用价值网络给局面打分（当前行棋方视角，[-1, +1]）。

        价值网络当前为线性输出（无 Tanh），这里用 tanh 压缩到 [-1, +1]，
        避免搜索中出现超出区间的极端值。
        """
        state = torch.tensor(
            board_to_state(board, current_player),
            dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = torch.tanh(self.value_model(state)).squeeze(-1).item()
        return float(value)

    def _backpropagate(self, node, value):
        """
        回传（Backpropagation）：沿父链更新访问次数与累计价值。

        每向上一层，视角取反（本节点的 +1 对父节点而言是 -1）。
        """
        sign = 1.0
        cur = node
        while cur is not None:
            cur.N += 1
            cur.W += sign * value
            sign = -sign
            cur = cur.parent

    # ---------------- 规则工具（委托给模块级函数，便于复用） ----------------

    def _get_legal_moves(self, board, current_player):
        """当前局面的全部合法动作（不含超劫历史检查，供外部快速查询）。"""
        return get_legal_moves(board, current_player)

    def _is_game_over(self, board, consecutive_passes=0, move_count=0):
        """判断对局是否结束。"""
        return is_game_over(board, consecutive_passes, move_count)

    def _score_game(self, board):
        """终局计分，返回 1=黑胜 / 2=白胜 / 0=平局。"""
        return score_game(board)


# ============================================================
# 自我对弈（可选，供未来强化学习使用）
# ============================================================

def self_play(mcts, num_games=10, temperature=1.0):
    """
    用给定 MCTS 进行简单的自我对弈，返回 (states, values)。

    Args:
        mcts: MCTS 实例
        num_games: 对局数
        temperature: 落子采样温度

    Returns:
        states: (N, 2, 19, 19) float32 数组
        values: (N,) float32 数组（当前行棋方视角：+1 胜 / -1 负 / 0 和）
    """
    all_states = []
    all_values = []

    for _ in range(num_games):
        board = [[0] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        current_player = 1
        passes = 0
        move_count = 0
        positions = []  # (state, player) 按行棋方视角记录
        game_history = {get_position_fingerprint(board)}  # 对局级劫历史（含当前局面）

        while not is_game_over(board, passes, move_count):
            positions.append((board_to_state(board, current_player), current_player))
            move = mcts.select_move(
                board, current_player,
                temperature=temperature, game_history=game_history,
            )
            if move == PASS_MOVE:
                passes += 1
            else:
                passes = 0
                row, col = divmod(move, BOARD_SIZE)
                board, _ = _apply_move(board, row, col, current_player)
                game_history.add(get_position_fingerprint(board))  # 新局面加入历史
            move_count += 1
            current_player = 3 - current_player

        winner = score_game(board)
        for state, player in positions:
            if winner == 0:
                value = 0.0
            else:
                value = 1.0 if player == winner else -1.0
            all_states.append(state)
            all_values.append(value)

    if not all_states:
        return np.zeros((0, 2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32), \
               np.zeros((0,), dtype=np.float32)
    return np.stack(all_states).astype(np.float32), np.array(all_values, dtype=np.float32)


# ============================================================
# 主入口：模型测试
# ============================================================

class _DummyPolicy(nn.Module):
    """占位策略网络：输出随机 logits，仅用于无真实模型时测试 MCTS 流程。"""

    def forward(self, x):
        batch = x.size(0)
        return torch.randn(batch, N_BOARD + 1, device=x.device) * 0.1


class _DummyValue(nn.Module):
    """占位价值网络：输出 0，仅用于无真实模型时测试 MCTS 流程。"""

    def forward(self, x):
        batch = x.size(0)
        return torch.zeros(batch, 1, device=x.device)


if __name__ == "__main__":
    print("=" * 60)
    print("MCTS 测试")
    print("=" * 60)

    # 1. 优先加载真实模型结构；失败则使用占位模型
    policy = None
    value = None
    try:
        from src.policy_network import PolicyNetworkAlphaGoLee
        from src.value_network import ValueNetworkAlphaGoLee

        policy = PolicyNetworkAlphaGoLee(input_channels=2)
        value = ValueNetworkAlphaGoLee(input_channels=2)
        print("✅ 已加载真实模型结构")
    except Exception as e:
        print(f"⚠️ 无法加载真实模型（{e}），使用占位模型")

    if policy is None:
        policy = _DummyPolicy()
    if value is None:
        value = _DummyValue()

    # 2. 创建 MCTS（测试用少量模拟）
    mcts = MCTS(policy, value, device='cpu', num_simulations=100, c_puct=1.0)

    # 3. 空棋盘上搜索并选择一手棋
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
    probs = mcts.search(board, current_player=1)
    move = mcts.select_move(board, current_player=1, temperature=0.0)

    print(f"\n选择动作: {move} ({'Pass' if move == PASS_MOVE else f'({move // BOARD_SIZE}, {move % BOARD_SIZE})'})")
    print(f"动作概率维度: {probs.shape}（应包含 362 个动作）")

    # 4. 打印概率最高的 5 个动作
    top_indices = np.argsort(probs)[::-1][:5]
    print("\nTop 5 动作:")
    for idx in top_indices:
        if idx == PASS_MOVE:
            label = "Pass"
        else:
            label = f"({idx // BOARD_SIZE}, {idx % BOARD_SIZE})"
        print(f"   move={idx:3d} ({label:10s}) 概率={probs[idx]:.4f}")

    # 5. 规则函数自检
    legal = get_legal_moves(board, current_player=1)
    assert PASS_MOVE in legal and len(legal) == N_BOARD + 1
    assert board_to_state(board, 1).shape == (2, BOARD_SIZE, BOARD_SIZE)
    print("\n✅ 规则函数自检通过：空盘 361 个落子点 + Pass 全部合法")
    print("=" * 60)
