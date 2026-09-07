"""MCTS 模块单元测试：合法点过滤（自杀/打劫）、棋盘编码、计分。"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.mcts import (  # noqa: E402
    PASS_MOVE,
    apply_move,
    board_to_state,
    get_legal_moves,
    get_position_fingerprint,
    is_game_over,
    score_game,
)


def _empty():
    return [[0] * 19 for _ in range(19)]


def test_empty_board_all_moves_legal():
    legal = get_legal_moves(_empty(), 1)
    assert len(legal) == 362 and PASS_MOVE in legal


def test_suicide_excluded():
    board = _empty()
    board[0][1] = 2
    board[1][0] = 2
    board[1][1] = 2
    legal = get_legal_moves(board, 1)
    assert 0 not in legal  # (0,0) 自杀


def test_ko_via_history():
    """history 含落子后局面指纹时，该手被屏蔽（位置超劫）。"""
    board = _empty()
    new_board, _ = apply_move(board, 9, 9, 1)
    history = {get_position_fingerprint(new_board)}
    legal = get_legal_moves(board, 1, history)
    assert 9 * 19 + 9 not in legal
    assert PASS_MOVE in legal


def test_board_to_state_channels():
    board = _empty()
    board[3][4] = 1
    board[5][6] = 2
    state = board_to_state(board, 1)
    assert state.shape == (2, 19, 19) and state.dtype == np.float32
    assert state[0, 3, 4] == 1 and state[1, 5, 6] == 1


def test_is_game_over_and_score():
    empty = _empty()
    assert is_game_over(empty, consecutive_passes=2) is True
    assert score_game([[1] * 19 for _ in range(19)]) == 1
    assert score_game(empty) == 2  # 贴目导致白胜
