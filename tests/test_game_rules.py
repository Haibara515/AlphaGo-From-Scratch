"""规则引擎（src/play.py 中的 GoEngine）单元测试：合法点 / 自杀 / 打劫 / 计分。"""

import pytest

from src.play import GoEngine


def _empty():
    return [[0] * 19 for _ in range(19)]


def test_empty_board_legal_moves():
    """空盘黑方第一步应可在任意交点落子。"""
    engine = GoEngine(19)
    assert engine.is_legal_move(9, 9, 1) is True
    assert engine.is_legal_move(0, 0, 1) is True


def test_occupied_point_illegal():
    """已占用点不能再落子。"""
    engine = GoEngine(19)
    engine.board[0][0] = 1
    assert engine.is_legal_move(0, 0, 2) is False


def test_suicide_illegal():
    """自杀（无气且不提子）非法。"""
    engine = GoEngine(19)
    engine.board[0][1] = 2
    engine.board[1][0] = 2
    engine.board[1][1] = 2
    assert engine.is_legal_move(0, 0, 1) is False


def test_ko_recapture_banned():
    """位置超劫：白立即回提会复现前一局面，必须被禁止。"""
    engine = GoEngine(19)
    board = _empty()
    for r, c in [(0, 1), (1, 0), (2, 1)]:
        board[r][c] = 1
    for r, c in [(0, 2), (1, 3), (2, 2)]:
        board[r][c] = 2
    board[1][1] = 2  # 白劫子
    engine.board = board
    engine.position_history = {
        engine._position_key(_empty()),
        engine._position_key(board),
    }
    engine.current_player = 1
    engine.game_over = False

    # 黑提劫合法
    assert engine.is_legal_move(1, 2, 1) is True
    result = engine.place_stone(1, 2, 1)
    assert result["success"] and result["captured"] == 1
    # 白立即回提非法（复现前一局面）
    assert engine.is_legal_move(1, 1, 2) is False


def test_scoring():
    """中国规则数子：满盘黑胜；空盘因贴目白胜。"""
    engine = GoEngine(19)
    engine.board = [[1] * 19 for _ in range(19)]
    assert engine.get_winner() == 1
    engine.board = _empty()
    assert engine.get_winner() == 2  # 白方贴目 7.5
