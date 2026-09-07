"""工具函数：配置加载与通用小工具。

说明：
- load_config()：读取 config/config.yaml（缺 pyyaml 时返回空 dict 并提示）；
- label_to_coord / coord_str：19 路动作索引（0~360）与棋盘坐标互转，361 表示 Pass。

本模块不依赖 torch/pygame，可被测试与脚本安全导入。
"""

import os
from typing import Optional, Tuple

try:
    import yaml
except ImportError:
    yaml = None


BOARD_SIZE = 19
PASS_MOVE = BOARD_SIZE * BOARD_SIZE  # 361


def load_config(path: Optional[str] = None) -> dict:
    """
    加载 config/config.yaml 配置。

    Args:
        path: 配置文件路径；默认取仓库根目录下的 config/config.yaml

    Returns:
        dict：解析后的配置；找不到文件或缺少 pyyaml 时返回空 dict
    """
    if path is None:
        # src/utils.py -> 仓库根/config/config.yaml
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, "config", "config.yaml")
    if yaml is None:
        print("⚠️ 未安装 pyyaml，跳过 config.yaml 加载（使用代码内默认值）")
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def label_to_coord(action: int) -> Optional[Tuple[int, int]]:
    """
    动作索引 -> 棋盘坐标 (row, col)；Pass（361）返回 None。

    Args:
        action: 0~360（row * 19 + col）或 361（Pass）

    Returns:
        (row, col) 或 None
    """
    if action == PASS_MOVE:
        return None
    return divmod(int(action), BOARD_SIZE)


def coord_str(action: int) -> str:
    """
    动作索引 -> 棋盘坐标字符串（如 "K7" / "Pass"），用于界面与日志显示。

    Args:
        action: 0~360 或 361（Pass）

    Returns:
        str
    """
    if action == PASS_MOVE:
        return "Pass"
    row, col = label_to_coord(action)
    return f"{chr(ord('A') + col)}{BOARD_SIZE - row}"
