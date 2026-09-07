import sys
from typing import List, Tuple, Optional

# 图形界面依赖改为惰性导入：缺少 pygame/torch 时，规则引擎仍可独立使用和测试
try:
    import pygame
except ImportError:
    pygame = None

try:
    import torch
except ImportError:
    torch = None


# ============================================================
#  1. 围棋规则引擎
# ============================================================

DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))

# 不同棋盘大小常用的贴目值（白方所得），可通过 GoEngine(komi=...) 覆盖
DEFAULT_KOMI = {19: 7.5, 13: 6.5, 9: 5.5}


def _collect_group(board, size: int, row: int, col: int) -> List[Tuple[int, int]]:
    """返回 (row, col) 所在同色连通棋串的全部坐标（不含空格）。"""
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
            if 0 <= nr < size and 0 <= nc < size \
                    and board[nr][nc] == color and (nr, nc) not in seen:
                stack.append((nr, nc))
    return group


def _group_liberties(board, size: int, group) -> set:
    """返回棋串所有气（上下左右相邻的空点）的集合。"""
    liberties = set()
    for r, c in group:
        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and board[nr][nc] == 0:
                liberties.add((nr, nc))
    return liberties


class GoEngine:
    """围棋核心规则引擎。

    规则要点：
    - 黑先、双方轮流落子；
    - 同色相连成串共气，无气即提；
    - 自杀（未提子且己方无气）非法；
    - 打劫采用位置超劫（Positional Superko）：任何一步都不能复现
      之前出现过的任意局面（含空盘），因此既挡单劫立即回提，
      也挡双劫、长生等循环；
    - 终局：双方连续停手或一方认输；
    - 计分：中国规则数子法（领地 + 活子，白方加贴目），死子先移除。
    """

    def __init__(self, size: int = 19, komi: Optional[float] = None):
        if not isinstance(size, int) or size < 2:
            raise ValueError("棋盘大小必须是不小于 2 的整数")
        self.size = size
        # 贴目：默认按棋盘大小取值，也可显式传入
        self.komi = float(komi) if komi is not None else float(DEFAULT_KOMI.get(size, 7.5))
        self.board = []
        self.captures = {1: 0, 2: 0}
        self.ko = None                # 简单劫的“劫争点”，仅作提示；真正判法见 position_history
        self.history = []             # 落子记录 [(row, col, color)]，停手记为 (-1, -1, color)
        self.position_history = set() # 出现过的全部局面，用于位置超劫
        self.dead_stones = set()      # 终局后双方一致认定的死子坐标
        self.passes = 0               # 连续停手计数
        self.game_over = False
        self.winner = None            # 认输时直接记录胜者
        self.current_player = 1       # 黑先
        self.last_move = None
        self.init_board()

    def init_board(self):
        """重置整局。"""
        self.board = [[0 for _ in range(self.size)] for _ in range(self.size)]
        self.captures = {1: 0, 2: 0}
        self.ko = None
        self.history = []
        # 空棋盘本身也计入历史局面：任何一步都不允许把局面还原成空盘
        self.position_history = {self._position_key(self.board)}
        self.dead_stones = set()
        self.passes = 0
        self.game_over = False
        self.winner = None
        self.current_player = 1
        self.last_move = None

    # ---------------- 基础工具 ----------------

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.size and 0 <= col < self.size

    def get_stone(self, row: int, col: int) -> int:
        if not self.in_bounds(row, col):
            return -1
        return self.board[row][col]

    @staticmethod
    def _position_key(board):
        """把棋盘转成可哈希的局面指纹（用于位置超劫判重）。"""
        return tuple(tuple(row) for row in board)

    def get_liberties(self, row: int, col: int) -> set:
        """返回 (row, col) 所在棋串的全部气；空格/越界返回空集。"""
        color = self.get_stone(row, col)
        if color not in (1, 2):
            return set()
        return _group_liberties(self.board, self.size,
                                _collect_group(self.board, self.size, row, col))

    def is_captured(self, row: int, col: int) -> bool:
        """某棋串是否无气（应当被提走）。"""
        color = self.get_stone(row, col)
        if color not in (1, 2):
            return False
        return not _group_liberties(self.board, self.size,
                                    _collect_group(self.board, self.size, row, col))

    def remove_group(self, row: int, col: int) -> int:
        """提走 (row, col) 所在的整串棋，返回实际提掉的棋子数。"""
        color = self.get_stone(row, col)
        if color not in (1, 2):
            return 0
        group = _collect_group(self.board, self.size, row, col)
        for r, c in group:
            self.board[r][c] = 0
        return len(group)

    # ---------------- 落子合法性 ----------------

    def _apply_move_to_board(self, board, row: int, col: int, color: int):
        """在 board（列表的列表，函数内会修改）上落子并提掉无气敌串。

        返回 (提子数, 劫争点)：
        - 劫争点：若本手恰好只提掉对方一颗子，劫争点就是那颗被提子
          的位置（对手若立即在此回提，会复现上一局面，属打劫）；
          否则为 None。
        """
        board[row][col] = color
        opponent = 3 - color
        captured_count = 0
        single_capture_point = None
        for dr, dc in DIRECTIONS:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < self.size and 0 <= nc < self.size):
                continue
            if board[nr][nc] != opponent:
                continue
            group = _collect_group(board, self.size, nr, nc)
            if _group_liberties(board, self.size, group):
                continue  # 该敌串还有气，不构成提子
            # 无气：整串立刻提掉（多子同提也在此处理）
            for gr, gc in group:
                board[gr][gc] = 0
            captured_count += len(group)
            if len(group) == 1:
                single_capture_point = (group[0][0], group[0][1])
        # 只有“恰好提一颗”才产生简单劫；提多颗时劫争点无效
        ko_point = single_capture_point if captured_count == 1 else None
        return captured_count, ko_point

    def _evaluate_move(self, row: int, col: int, color: int):
        """完整判定一手棋，返回 (是否合法, 原因, 落子后的棋盘, 提子数, 劫争点)。

        place_stone / get_legal_moves / can_place 共用这一个入口，
        避免各方法对规则的判定不一致。
        """
        if color not in (1, 2):
            return False, '无效颜色', None, 0, None
        if not self.in_bounds(row, col):
            return False, '坐标越界', None, 0, None
        if self.board[row][col] != 0:
            return False, '此处已有棋子', None, 0, None

        # 在副本上模拟：先落子，再提掉对方无气棋串
        board_copy = [list(row) for row in self.board]
        captured, ko_point = self._apply_move_to_board(board_copy, row, col, color)

        # 自杀判定：未提任何对方棋子且己方棋串无气 → 非法。
        # 若提了子，被提棋子的空点必然成为己方气，因此己方不可能无气；
        # 这里显式写 captured == 0 只是与“提子后可暂时无气”的规则表述一致。
        if captured == 0:
            group = _collect_group(board_copy, self.size, row, col)
            if not _group_liberties(board_copy, self.size, group):
                return False, '禁止自杀（落子后己方无气且未提子）', None, 0, None

        # 位置超劫：任何一步都不能复现之前出现过的任意局面。
        # 这比“单劫立即回提”更严格：双劫、长生等循环同样被禁止。
        if self._position_key(board_copy) in self.position_history:
            return False, '违反打劫规则：该手会复现先前局面', None, 0, None

        return True, 'ok', board_copy, captured, ko_point

    def is_legal_move(self, row: int, col: int, color: int) -> bool:
        """完整合法性（越界/占用/自杀/打劫均含）。"""
        ok, _, _, _, _ = self._evaluate_move(row, col, color)
        return ok

    def can_place(self, row: int, col: int, color: int) -> bool:
        """兼容旧接口：等价于 is_legal_move（含打劫判定）。"""
        return self.is_legal_move(row, col, color)

    def get_legal_moves(self, color: int) -> List[Tuple[int, int]]:
        """返回某方所有合法落子点（不含停手）。"""
        moves = []
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r][c] == 0 and self.is_legal_move(r, c, color):
                    moves.append((r, c))
        return moves

    # ---------------- 对局操作 ----------------

    def place_stone(self, row: int, col: int, color: int) -> dict:
        if self.game_over:
            return {'success': False, 'message': '对局已结束'}
        if color != self.current_player:
            return {'success': False, 'message': '不是你的回合'}

        ok, reason, board_after, captured, ko_point = self._evaluate_move(row, col, color)
        if not ok:
            return {'success': False, 'message': reason}

        # 直接采用模拟结果（提子已在模拟中完成），保证落子与提子次序正确
        self.board = board_after
        self.position_history.add(self._position_key(board_after))
        self.last_move = (row, col, color)
        self.ko = ko_point
        self.captures[color] = self.captures.get(color, 0) + captured
        self.history.append((row, col, color))
        self.passes = 0               # 落子打断连续停手
        self.current_player = 3 - color

        return {'success': True, 'captured': captured, 'game_over': False,
                'message': '落子成功'}

    def pass_move(self, color: int) -> dict:
        if self.game_over:
            return {'success': False, 'message': '对局已结束'}
        if color != self.current_player:
            return {'success': False, 'message': '不是你的回合'}
        self.passes += 1
        self.history.append((-1, -1, color))
        self.ko = None  # 停手解除“立即回提”的简单劫限制；局面重复仍由超劫判定
        self.current_player = 3 - color
        if self.passes >= 2:
            self.game_over = True
            return {'success': True, 'game_over': True, 'message': '双方连续停手，对局结束'}
        return {'success': True, 'game_over': False, 'message': '停一手'}

    def resign(self, color: int) -> dict:
        """认输：认输方失败，对手获胜，对局立即结束。"""
        if self.game_over:
            return {'success': False, 'message': '对局已结束'}
        if color not in (1, 2):
            return {'success': False, 'message': '无效颜色'}
        self.game_over = True
        self.winner = 3 - color
        return {'success': True, 'game_over': True, 'winner': self.winner,
                'message': f"{'黑' if color == 1 else '白'}棋认输，"
                           f"{'白' if color == 1 else '黑'}棋获胜"}

    def toggle_dead_stone(self, row: int, col: int) -> dict:
        """终局后标记/取消标记死子（中国规则数子前先确认死子）。"""
        if not self.game_over:
            return {'success': False, 'message': '对局结束后才能标记死子'}
        if not self.in_bounds(row, col):
            return {'success': False, 'message': '坐标越界'}
        key = (row, col)
        if self.board[row][col] == 0 and key not in self.dead_stones:
            return {'success': False, 'message': '该位置没有棋子'}
        if key in self.dead_stones:
            self.dead_stones.remove(key)
            return {'success': True, 'dead': False, 'message': '已取消死子标记'}
        self.dead_stones.add(key)
        return {'success': True, 'dead': True, 'message': '已标记为死子'}

    # ---------------- 终局计分（中国规则） ----------------

    def _compute_territory(self, board) -> dict:
        """对给定棋盘数空：空点区域只被黑包围算黑地，只被白包围算白地，否则公空。"""
        size = self.size
        visited = [[False] * size for _ in range(size)]
        black_territory = 0
        white_territory = 0
        dame = 0

        for r in range(size):
            for c in range(size):
                if board[r][c] != 0 or visited[r][c]:
                    continue
                # 收集一整块相连空点
                region = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    cr, cc = stack.pop()
                    region.append((cr, cc))
                    for dr, dc in DIRECTIONS:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < size and 0 <= nc < size \
                                and not visited[nr][nc] and board[nr][nc] == 0:
                            visited[nr][nc] = True
                            stack.append((nr, nc))

                # 判断该空区归属：看它紧邻哪些颜色的棋
                has_black = False
                has_white = False
                for cr, cc in region:
                    for dr, dc in DIRECTIONS:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < size and 0 <= nc < size:
                            if board[nr][nc] == 1:
                                has_black = True
                            elif board[nr][nc] == 2:
                                has_white = True

                if has_black and not has_white:
                    black_territory += len(region)
                elif has_white and not has_black:
                    white_territory += len(region)
                else:
                    dame += len(region)  # 公空，双方都不计

        return {'black': black_territory, 'white': white_territory, 'dame': dame}

    def get_territory(self) -> dict:
        """数空：死子先视为空点再数，返回 {'black', 'white', 'dame'}。"""
        board = [list(row) for row in self.board]
        for r, c in self.dead_stones:
            board[r][c] = 0
        return self._compute_territory(board)

    def get_score(self) -> dict:
        """中国规则数子法：领地 + 棋盘上活子；白方加贴目。"""
        board = [list(row) for row in self.board]
        # 1) 数子前先移除双方认定的死子；死子占的点按空点计入领地
        for r, c in self.dead_stones:
            board[r][c] = 0

        # 2) 领地
        territory = self._compute_territory(board)

        # 3) 棋盘上的活子
        black_stones = 0
        white_stones = 0
        for r in range(self.size):
            for c in range(self.size):
                if board[r][c] == 1:
                    black_stones += 1
                elif board[r][c] == 2:
                    white_stones += 1

        black = territory['black'] + black_stones
        white = territory['white'] + white_stones + self.komi
        return {'black': black, 'white': white, 'komi': self.komi}

    def get_winner(self) -> int:
        """返回 1=黑胜, 2=白胜, 0=平局（贴目为 .5 时不会平局）。"""
        if self.winner is not None:
            return self.winner
        scores = self.get_score()
        if scores['black'] > scores['white']:
            return 1
        if scores['white'] > scores['black']:
            return 2
        return 0


# ============================================================
#  2. Pygame 围棋游戏
# ============================================================

class GoPygame:
    """基于 Pygame 的围棋游戏"""

    def __init__(self, size: int = 19,
                 ai_black: bool = False,
                 ai_white: bool = True,
                 model_path: Optional[str] = None):

        if pygame is None:
            raise RuntimeError("未安装 pygame，无法启动图形界面（pip install pygame）")

        # 初始化 Pygame
        pygame.init()
        self.size = size
        self.engine = GoEngine(size)

        # 窗口尺寸
        self.window_size = 700
        self.margin = 30
        self.cell_size = (self.window_size - 2 * self.margin) / (size - 1)
        self.stone_radius = self.cell_size * 0.42

        # 创建窗口
        self.screen = pygame.display.set_mode((self.window_size + 200, self.window_size))
        pygame.display.set_caption(f"围棋 AI - {size}路")
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)

        # AI 设置
        self.ai_black = ai_black
        self.ai_white = ai_white
        self.model_path = model_path
        self.ai = None
        self.ai_think_time = 0.3

        # 状态
        self.running = True
        self.message = ""
        self.message_time = 0

        # 颜色
        self.colors = {
            'board': (220, 200, 160),
            'lines': (60, 40, 20),
            'black': (20, 20, 20),
            'white': (240, 240, 240),
            'text': (40, 30, 20),
            'highlight': (255, 0, 0, 100),
            'bg': (50, 45, 40),
        }

        # 鼠标悬停位置
        self.hover_pos = None

        # 加载AI
        if ai_black or ai_white:
            self._init_ai()

    def run_ai_vs_ai(self, record_moves: bool = True):
        """
        AI vs AI 专用运行模式
        record_moves: 是否记录每一步到控制台
        """
        print("\n" + "=" * 60)
        print("  AI vs AI - Battle Mode")
        print("  Black AI vs White AI")
        print("=" * 60)

        clock = pygame.time.Clock()

        # 如果AI先手，走第一步
        if self.ai_black:
            pygame.time.wait(300)
            self._do_ai_move()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_r:
                        self.engine.init_board()
                        self.show_message("New game")
                        self.render()
                        if self.ai_black:
                            pygame.time.wait(300)
                            self._do_ai_move()
                    elif event.key == pygame.K_SPACE:
                        # 按空格加速AI下棋
                        for _ in range(10):
                            if not self.engine.game_over:
                                self._do_ai_move()
                                self.render()
                                pygame.time.wait(50)

            # AI vs AI 自动下棋
            if not self.engine.game_over:
                color = self.engine.current_player
                if (color == 1 and self.ai_black) or (color == 2 and self.ai_white):
                    self._do_ai_move()
                    if record_moves:
                        move = self.engine.history[-1] if self.engine.history else None
                        if move and move[0] != -1:
                            coord = f"{chr(65 + move[1])}{self.size - move[0]}"
                            player = "Black" if move[2] == 1 else "White"
                            print(f"{player} -> {coord}  (Move {len(self.engine.history)})")
                        elif move:
                            print(f"{'Black' if move[2] == 1 else 'White'} -> PASS")

            self.render()
            clock.tick(60)

        pygame.quit()
        sys.exit()

    def _init_ai(self):
        """初始化 AI，加载模型；缺 torch 或模型时降级为占位逻辑。"""
        if torch is None:
            print("⚠️ 未安装 torch，AI 使用占位逻辑")
            self.ai = None
            return
        try:
            # 导入你的模型定义
            from Strategy_Network import PolicyNetworkResNet

            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.ai = PolicyNetworkResNet(input_channels=2, num_actions=self.size * self.size)

            # 加载检查点
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.ai.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.ai.load_state_dict(checkpoint)

            self.ai.to(self.device)
            self.ai.eval()
            print(f"✅ AI 模型已加载: {self.model_path}")
        except Exception as e:
            print(f"⚠️ AI 加载失败: {e}")
            self.ai = None

    def _build_state(self):
        """将当前棋盘状态转换为模型输入张量 (2, size, size)。"""
        import numpy as np

        board = self.engine.board
        size = self.engine.size

        state = np.zeros((2, size, size), dtype=np.float32)
        current = self.engine.current_player
        opponent = 3 - current

        for r in range(size):
            for c in range(size):
                stone = board[r][c]
                if stone == current:
                    state[0, r, c] = 1.0
                elif stone == opponent:
                    state[1, r, c] = 1.0

        return torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _ai_predict(self, color: int) -> Optional[Tuple[int, int]]:
        """AI 预测落子位置（只从合法点里选）。"""
        legal = self.engine.get_legal_moves(color)
        if not legal:
            return None

        # 如果模型未加载，使用占位逻辑
        if self.ai is None:
            center = (self.engine.size // 2, self.engine.size // 2)
            best = min(legal, key=lambda pos: (pos[0] - center[0]) ** 2 + (pos[1] - center[1]) ** 2)
            return best

        # ========== 接入真实模型 ==========
        try:
            # 1. 构建输入张量
            state = self._build_state()

            # 2. 模型推理
            with torch.no_grad():
                logits = self.ai(state)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            # 3. 在合法位置中选择概率最高的
            best_move = None
            best_prob = -1
            for r, c in legal:
                idx = r * self.engine.size + c   # 修正：索引随棋盘大小变化
                prob = probs[idx]
                if prob > best_prob:
                    best_prob = prob
                    best_move = (r, c)

            return best_move
        except Exception as e:
            print(f"AI 推理失败: {e}")
            # 降级到占位逻辑
            center = (self.engine.size // 2, self.engine.size // 2)
            return min(legal, key=lambda pos: (pos[0] - center[0]) ** 2 + (pos[1] - center[1]) ** 2)

    def _get_board_pos(self, mouse_x: int, mouse_y: int) -> Optional[Tuple[int, int]]:
        """将鼠标坐标转换为最近的棋盘交叉点（交点落子）。"""
        for r in range(self.size):
            for c in range(self.size):
                x = self.margin + c * self.cell_size
                y = self.margin + r * self.cell_size
                dx = mouse_x - x
                dy = mouse_y - y
                if dx * dx + dy * dy < (self.cell_size * 0.5) ** 2:
                    return (r, c)
        return None

    def render(self):
        """渲染棋盘"""
        self.screen.fill(self.colors['bg'])

        # 棋盘背景
        board_w = self.window_size - 2 * self.margin
        board_rect = (self.margin - 10, self.margin - 10,
                      board_w + 20, board_w + 20)
        pygame.draw.rect(self.screen, self.colors['board'], board_rect, border_radius=8)
        pygame.draw.rect(self.screen, (160, 140, 100), board_rect, 2, border_radius=8)

        # 画网格线
        for i in range(self.size):
            x = self.margin + i * self.cell_size
            y = self.margin + i * self.cell_size
            pygame.draw.line(self.screen, self.colors['lines'],
                             (x, self.margin), (x, self.window_size - self.margin), 1)
            pygame.draw.line(self.screen, self.colors['lines'],
                             (self.margin, y), (self.window_size - self.margin, y), 1)

        # 画星位（按棋盘大小取常见星位）
        star_points = {19: [3, 9, 15], 13: [3, 6, 9], 9: [2, 4, 6]}.get(self.size, [])
        for r in star_points:
            for c in star_points:
                x = self.margin + c * self.cell_size
                y = self.margin + r * self.cell_size
                pygame.draw.circle(self.screen, self.colors['lines'], (x, y), 5)

        # 画棋子
        for r in range(self.size):
            for c in range(self.size):
                stone = self.engine.get_stone(r, c)
                if stone != 0:
                    x = self.margin + c * self.cell_size
                    y = self.margin + r * self.cell_size
                    if stone == 1:
                        pygame.draw.circle(self.screen, self.colors['black'], (x, y), self.stone_radius)
                        pygame.draw.circle(self.screen, (60, 60, 60), (x - 2, y - 2), self.stone_radius * 0.3)
                    else:
                        pygame.draw.circle(self.screen, self.colors['white'], (x, y), self.stone_radius)
                        pygame.draw.circle(self.screen, (220, 220, 220), (x - 2, y - 2), self.stone_radius * 0.3)
                    pygame.draw.circle(self.screen, (80, 80, 80), (x, y), self.stone_radius, 1)

        # 画死子标记（红圈 + 叉）
        if self.engine.dead_stones:
            for (r, c) in self.engine.dead_stones:
                x = self.margin + c * self.cell_size
                y = self.margin + r * self.cell_size
                pygame.draw.circle(self.screen, (220, 30, 30), (x, y), self.stone_radius + 2, 2)
                s = self.stone_radius * 0.45
                pygame.draw.line(self.screen, (220, 30, 30), (x - s, y - s), (x + s, y + s), 2)
                pygame.draw.line(self.screen, (220, 30, 30), (x - s, y + s), (x + s, y - s), 2)

        # 画最后落子标记
        if self.engine.last_move:
            r, c, color = self.engine.last_move
            x = self.margin + c * self.cell_size
            y = self.margin + r * self.cell_size
            mark_color = (255, 255, 255) if color == 1 else (0, 0, 0)
            pygame.draw.circle(self.screen, mark_color, (x, y), self.cell_size * 0.1)

        # 画鼠标悬停（pygame.draw 不支持透明度，用带 alpha 的临时表面）
        if self.hover_pos and not self.engine.game_over:
            r, c = self.hover_pos
            if self.engine.get_stone(r, c) == 0:
                x = self.margin + c * self.cell_size
                y = self.margin + r * self.cell_size
                dia = int(self.stone_radius * 2) + 2
                hover = pygame.Surface((dia, dia), pygame.SRCALPHA)
                if self.engine.current_player == 1:
                    hover_color = (0, 0, 0, 90)
                else:
                    hover_color = (255, 255, 255, 120)
                pygame.draw.circle(hover, hover_color, (dia // 2, dia // 2), int(self.stone_radius))
                self.screen.blit(hover, (int(x - self.stone_radius - 1), int(y - self.stone_radius - 1)))

        # 右侧信息面板
        info_x = self.window_size + 10
        title = self.large_font.render("对局信息", True, self.colors['text'])
        self.screen.blit(title, (info_x, 20))

        player = self.engine.current_player
        player_text = "● 黑棋" if player == 1 else "○ 白棋"
        p_label = self.font.render(f"走棋: {player_text}", True, self.colors['text'])
        self.screen.blit(p_label, (info_x, 60))

        moves = self.font.render(f"手数: {len(self.engine.history)}", True, self.colors['text'])
        self.screen.blit(moves, (info_x, 90))

        black_cap = self.engine.captures.get(1, 0)
        white_cap = self.engine.captures.get(2, 0)
        cap_text = self.font.render(f"提子: 黑 {black_cap}  白 {white_cap}", True, self.colors['text'])
        self.screen.blit(cap_text, (info_x, 120))

        territory = self.engine.get_territory()
        terr_text = self.font.render(f"领地: 黑 {territory['black']}  白 {territory['white']}", True,
                                     self.colors['text'])
        self.screen.blit(terr_text, (info_x, 150))

        ai_status = "🤖 AI 执白" if self.ai_white else "🤖 AI 执黑" if self.ai_black else "双人对战"
        ai_label = self.font.render(ai_status, True, self.colors['text'])
        self.screen.blit(ai_label, (info_x, 180))

        komi_text = self.font.render(f"贴目: {self.engine.komi:g}", True, self.colors['text'])
        self.screen.blit(komi_text, (info_x, 210))

        if self.engine.game_over:
            scores = self.engine.get_score()
            score_text = self.font.render(
                f"总分(含贴目): 黑 {scores['black']:g}  白 {scores['white']:g}",
                True, self.colors['text'])
            self.screen.blit(score_text, (info_x, 240))

            winner = self.engine.get_winner()
            if winner == 1:
                win_text = self.large_font.render("🏆 黑胜", True, (200, 50, 50))
            elif winner == 2:
                win_text = self.large_font.render("🏆 白胜", True, (50, 50, 200))
            else:
                win_text = self.large_font.render("🤝 平局", True, (100, 100, 100))
            self.screen.blit(win_text, (info_x, 280))

            dead_hint = self.font.render("点击棋子标记死子", True, self.colors['text'])
            self.screen.blit(dead_hint, (info_x, 330))
            reset_hint = self.font.render("R: 重新开局  ESC: 退出", True, self.colors['text'])
            self.screen.blit(reset_hint, (info_x, 360))
        else:
            tips = [
                "点击棋盘落子",
                "P: 停手",
                "X: 认输",
                "R: 重新开局",
                "ESC: 退出",
            ]
            for i, tip in enumerate(tips):
                tip_text = self.font.render(tip, True, self.colors['text'])
                self.screen.blit(tip_text, (info_x, 240 + i * 30))

        # 消息
        if self.message and pygame.time.get_ticks() - self.message_time < 2000:
            msg_text = self.font.render(self.message, True, (200, 50, 50))
            self.screen.blit(msg_text, (self.margin, self.window_size - 30))

        pygame.display.flip()

    def show_message(self, msg: str):
        self.message = msg
        self.message_time = pygame.time.get_ticks()

    def handle_click(self, row: int, col: int):
        """处理点击落子；终局后点击用于标记/取消死子。"""
        if self.engine.game_over:
            result = self.engine.toggle_dead_stone(row, col)
            self.show_message(result['message'])
            self.render()
            return

        color = self.engine.current_player

        # 检查是否为AI回合
        if (color == 1 and self.ai_black) or (color == 2 and self.ai_white):
            self.show_message("请等待 AI 思考...")
            return

        # 执行落子
        result = self.engine.place_stone(row, col, color)
        if not result['success']:
            self.show_message(result['message'])
            return

        self.show_message(f"落子 ({row}, {col})")
        self.render()

        # AI 走棋
        self._do_ai_move()

    def _do_ai_move(self):
        """AI 走棋（无合法点时自动停手）。"""
        if self.engine.game_over:
            return

        color = self.engine.current_player
        if (color == 1 and not self.ai_black) or (color == 2 and not self.ai_white):
            return

        # 使用 pygame.time 延迟，让用户看到落子
        pygame.time.wait(int(self.ai_think_time * 1000))

        move = self._ai_predict(color)
        if move is None:
            result = self.engine.pass_move(color)
            self.show_message("AI 停手")
            self.render()
            if result.get('game_over', False):
                self.show_message("对局结束！")
                self.render()
            return

        row, col = move
        result = self.engine.place_stone(row, col, color)
        if result['success']:
            self.show_message(f"AI 落子 ({row}, {col})")
            self.render()
            if result.get('game_over', False):
                self.show_message("对局结束！")
                self.render()

    def _handle_resign_key(self):
        """X 键认输：双人模式认当前回合方；人机模式只允许人类方认输。"""
        if self.engine.game_over:
            return
        color = self.engine.current_player
        if self.ai_black != self.ai_white:  # 人机模式
            human_color = 2 if self.ai_black else 1
            if color != human_color:
                self.show_message("AI 回合无法认输")
                return
        result = self.engine.resign(color)
        self.show_message(result['message'])
        self.render()

    def run(self):
        """主循环"""
        clock = pygame.time.Clock()

        # 如果AI先手，走第一步
        if self.ai_black:
            pygame.time.wait(500)
            self._do_ai_move()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_r:
                        # 重新开局
                        self.engine.init_board()
                        self.show_message("新开局")
                        self.render()
                        if self.ai_black:
                            pygame.time.wait(300)
                            self._do_ai_move()
                    elif event.key == pygame.K_p:
                        # 停手
                        if not self.engine.game_over:
                            color = self.engine.current_player
                            result = self.engine.pass_move(color)
                            self.show_message(result.get('message', '停手'))
                            self.render()
                            if result.get('game_over', False):
                                self.show_message("对局结束！")
                                self.render()
                            else:
                                self._do_ai_move()
                    elif event.key == pygame.K_x:
                        # 认输
                        self._handle_resign_key()

                elif event.type == pygame.MOUSEMOTION:
                    # 更新悬停位置
                    mouse_x, mouse_y = event.pos
                    if mouse_x < self.window_size - self.margin and mouse_y < self.window_size - self.margin:
                        self.hover_pos = self._get_board_pos(mouse_x, mouse_y)
                    else:
                        self.hover_pos = None

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # 左键
                        mouse_x, mouse_y = event.pos
                        if mouse_x < self.window_size - self.margin and mouse_y < self.window_size - self.margin:
                            pos = self._get_board_pos(mouse_x, mouse_y)
                            if pos:
                                self.handle_click(pos[0], pos[1])

            # AI 回合自动走棋
            if not self.engine.game_over:
                color = self.engine.current_player
                is_ai_turn = (color == 1 and self.ai_black) or (color == 2 and self.ai_white)
                if is_ai_turn:
                    self._do_ai_move()
                    self.render()
                    pygame.time.wait(int(self.ai_think_time * 100))

            self.render()
            clock.tick(60)

            if self.engine.game_over:
                # 显示结束信息
                winner = self.engine.get_winner()
                if winner == 1:
                    end_msg = "🏆 黑棋胜！"
                elif winner == 2:
                    end_msg = "🏆 白棋胜！"
                else:
                    end_msg = "🤝 平局！"
                self.show_message(f"对局结束！{end_msg} 点击棋子标记死子，R 重新开局")

        pygame.quit()
        sys.exit()


# ============================================================
#  3. 主入口
# ============================================================

def main():
    print("=" * 60)
    print("  围棋 AI - Pygame 版")
    print("=" * 60)
    print("  操作说明:")
    print("    - 鼠标点击棋盘落子")
    print("    - P 键: 停一手")
    print("    - X 键: 认输")
    print("    - R 键: 重新开局")
    print("    - ESC: 退出")
    print("=" * 60)

    # 选择模式
    print("\n请选择模式:")
    print("  1. 双人对战")
    print("  2. 人机对战 (AI 执白)")
    print("  3. 人机对战 (AI 执黑)")
    print("  4. AI vs AI (正常速度)")
    print("  5. AI vs AI (快速)")

    choice = input("请输入 (1-5): ").strip()
    # 模型路径（根据你的实际路径修改）
    model_path = "./checkpoints/latest_checkpoint(2) .pth"  # 或者用 best_model.pth

    if choice == '1':
        game = GoPygame(19, ai_black=False, ai_white=False, model_path=model_path)
        game.run()
    elif choice == '2':
        game = GoPygame(19, ai_black=False, ai_white=True, model_path=model_path)
        game.run()
    elif choice == '3':
        game = GoPygame(19, ai_black=True, ai_white=False, model_path=model_path)
        game.run()
    elif choice == '4':
        # AI vs AI (normal speed)
        game = GoPygame(19, ai_black=True, ai_white=True, model_path=model_path)
        game.ai_think_time = 0.5
        game.run()
    elif choice == '5':
        # AI vs AI (fast mode)
        game = GoPygame(19, ai_black=True, ai_white=True, model_path=model_path)
        game.ai_think_time = 0.05
        game.run()
    else:
        print("Invalid choice, starting Human vs AI...")
        game = GoPygame(19, ai_black=False, ai_white=True, model_path=model_path)
        game.run()


if __name__ == "__main__":
    main()
