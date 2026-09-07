#!/usr/bin/env bash
# 一键实验脚本：价值网络数据 -> 价值网络训练 -> 自我对弈。
# 请在仓库根目录运行：bash scripts/run_experiment.sh
set -euo pipefail

echo "[1/3] 生成价值网络训练数据（需要 downloads/*.zip 与 sgfmill）"
python scripts/process_sgf_value.py || echo "（提示：暂无 SGF 数据，跳过数据生成）"

echo "[2/3] 训练价值网络"
python -m src.value_network

echo "[3/3] 自我对弈，生成强化学习训练数据"
python -m src.self_play

echo "完成！结果见 self_play_data/ 与 checkpoints_value/"
