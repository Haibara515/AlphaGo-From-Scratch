#!/usr/bin/env bash
# 下载预训练权重示例脚本。
# 权重体积大，建议用 Git LFS 或网盘分发；请把 <url> 替换为真实下载地址。
set -euo pipefail

mkdir -p checkpoints checkpoints_value

echo "=== 请将训练好的权重放到以下位置 ==="
echo "  策略网络: checkpoints/best_model.pth"
echo "  价值网络: checkpoints_value/best_model.pth"
echo
echo "示例（替换 <url>）："
echo "  wget -O checkpoints/best_model.pth       <policy-url>"
echo "  wget -O checkpoints_value/best_model.pth <value-url>"
