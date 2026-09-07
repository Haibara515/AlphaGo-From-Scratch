"""setup.py - 可选安装脚本（以 src 为包根目录）。

安装后可全局使用 `import src.xxx`，但本项目推荐直接在仓库根目录运行：
    python -m src.self_play
    python -m pytest tests/ -v
"""

from setuptools import find_packages, setup


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="alphago-from-scratch",
    version="0.1.0",
    description="AlphaGo 从零复刻：策略网络 + 价值网络 + MCTS 自我对弈（PyTorch）",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "pygame>=2.5",
        "tqdm>=4.65",
        "sgfmill>=1.1",
    ],
)
