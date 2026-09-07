"""自我对弈工具函数单元测试（sample_move，不依赖 torch）。"""

import numpy as np

from src.self_play import sample_move


def test_sample_move_greedy():
    probs = np.zeros(362, dtype=np.float32)
    probs[5] = 0.7
    probs[361] = 0.3
    assert sample_move(probs, temperature=0.0) == 5


def test_sample_move_temperature_covers_positive():
    probs = np.zeros(362, dtype=np.float32)
    probs[5] = 0.7
    probs[361] = 0.3
    moves = {sample_move(probs, temperature=1.0) for _ in range(2000)}
    assert 5 in moves and 361 in moves


def test_sample_move_zero_fallback():
    probs = np.zeros(362, dtype=np.float32)
    assert 0 <= sample_move(probs, temperature=1.0) < 362
