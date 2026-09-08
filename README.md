# AlphaGo-From-Scratch

> 基于 DeepMind 论文，用 PyTorch 从零复刻 AlphaGo（Lee 版本）：
> **策略网络 + 价值网络 + MCTS 蒙特卡洛树搜索**，并带完整数据管线、自我对弈与 Pygame 对战界面。

## 项目背景

我是一名**机械工程背景（机器视觉方向）**的开发者。这个项目的初衷是：用一套真正
「从论文到可运行程序」的完整 AI 系统，证明自己不仅会调包，更理解深度强化学习的
**建模、训练、搜索与工程化**全链路。

围棋是 AI 领域的“登月工程”：棋盘状态空间巨大、没有标准奖励信号、需要长程规划。
从零实现 AlphaGo，等于把「深度学习 + 强化学习 + 博弈树搜索」三大主题一次性跑通。

## 跨界亮点：用物理直觉理解 AI

机械与物理的训练让我对 AlphaGo 有天然的亲切感，常用以下类比帮助自己建模：

- **策略网络 ≈ 力场 / 速度场**：给当前局面一个“该往哪走”的倾向分布；
- **价值网络 ≈ 势能函数**：给局面一个标量“能量高低”，谁占优一目了然；
- **MCTS 的 PUCT 探索 ≈ 多体系统的试探平衡**：`Q` 是“已测到的能量”，
  `P · √N/(1+N)` 是“未勘探区域的梯度”，二者权衡就像在能量景观中做多尺度采样；
- **提子与气 ≈ 连通域的受力分析**：气（liberty）就是棋串的“自由度”，
  自由度归零即“失稳”，被系统移除——这与机构自由度分析高度同构。

这种“把 AI 当物理系统去调试”的视角，帮助我在价值网络 loss 卡在 1.0 时，快速定位到
**Tanh 饱和 / 初始化不当 / 优化器过弱**这类“能量景观卡死”问题，并完成修复。

## 核心架构

```text
SGF 棋谱 ──► 数据管线 ──► 策略网络监督学习 ──┐
                             价值网络监督学习 ──┼──► MCTS(PUCT) ──► 自我对弈 / 对战
                                              └───────────────┘
```

| 模块 | 文件 | 说明 |
|---|---|---|
| 策略网络 | `src/policy_network.py` | AlphaGo Lee 13 层卷积，输出 362 logits（含 Pass） |
| 价值网络 | `src/value_network.py` | 13 层卷积 + BatchNorm + 全局池化 + FC，输出标量 |
| MCTS | `src/mcts.py` | 选择/扩展/评估/回传，位置超劫，CPU/GPU 兼容 |
| 自我对弈 | `src/self_play.py` | 生成 `states + probs + values` 训练数据 |
| 对战/规则引擎 | `src/play.py` | GoEngine 完整规则 + Pygame 人机/AI 对战 |
| 数据管线 | `scripts/process_sgf_*.py` | SGF → `.npy`（策略/价值数据） |

详细设计见 [docs/architecture.md](docs/architecture.md)。

## 当前成果

- [x] 围棋规则引擎（提子 / 自杀 / 位置超劫 / 中国规则计分）
- [x] SGF 棋谱解析与训练数据生成（策略网络 + 价值网络）
- [x] 策略网络：AlphaGo Lee 架构，362 类监督学习
- [x] 价值网络：±1 标签（当前行棋方视角）MSE 回归
- [x] MCTS：PUCT 搜索 + 合法动作过滤 + 对局级打劫历史
- [x] 自我对弈数据生成（温度退火：前 30 手 1.0，之后 0.1）
- [x] Pygame 对战：双人 / 人机 / AI vs AI / 纯策略 vs MCTS
- [x] CPU/GPU 双端支持与设备自动回退
- [ ] 48 平面完整特征（8 步历史 + 轮次）
- [ ] AlphaGo Zero 式强化学习（自我对弈 + 联合训练）

## 目录结构

```text
AlphaGo-From-Scratch/
├── config/config.yaml      # 集中式参数配置
├── src/                    # 核心源码（Python 包）
│   ├── policy_network.py   # 策略网络
│   ├── value_network.py    # 价值网络
│   ├── mcts.py             # MCTS
│   ├── self_play.py        # 自我对弈
│   ├── play.py             # 规则引擎 + Pygame 对战
│   └── utils.py            # 工具函数
├── tests/                  # 单元测试
├── docs/                   # 架构说明 / 训练日志
├── scripts/                # 数据管线与实验脚本
├── models/                 # 权重说明（大文件不提交）
└── archive/                # 历史版本归档
```

## 技术栈

- Python 3.9+
- PyTorch 2.x（CUDA 可选）
- NumPy / Pygame / tqdm / sgfmill / PyYAML
- 测试：pytest

完整依赖见 [requirements.txt](requirements.txt)。

## 安装与运行

```bash
# 1) 克隆并安装依赖
git clone <your-repo-url>
cd AlphaGo-From-Scratch
pip install -r requirements.txt

# 2) （可选）准备 SGF 数据：把 .zip 放进 downloads/ 后执行
python scripts/process_sgf_policy.py   # 生成策略网络训练数据 processed_data/
python scripts/process_sgf_value.py    # 生成价值网络训练数据 processed_data_value/

# 3) 训练（均从仓库根目录执行）
python -m src.policy_network           # 训练策略网络 -> checkpoints/
python -m src.value_network            # 训练价值网络 -> checkpoints_value/

# 4) 自我对弈，生成强化学习数据
python -m src.self_play                # -> self_play_data/

# 5) 对战（Pygame 菜单：人机 / AI vs AI / 纯策略 vs MCTS）
python -m src.play

# 6) 运行单元测试
python -m pytest tests/ -v
```

> 注意：代码内默认数据/权重路径均为仓库根目录下的相对路径
> （`./checkpoints/`、`./processed_data/` 等），请按上述方式从根目录运行；
> 各入口的超参数与路径统一记录在 [config/config.yaml](config/config.yaml)。

## 模型权重

训练好的 `.pth` 体积较大（策略网络约 29 MB），可用提供的脚本自行训练，后续可能会传入模型权重参考 [models/README.md](models/README.md) 与
[scripts/download_weights.sh](scripts/download_weights.sh)。

## 训练日志

实验记录与调参历史见 [docs/training_log.md](docs/training_log.md)。

## 致谢与引用

- Silver, D., et al. *Mastering the game of Go with deep neural networks and tree
  search.* Nature, 2016.（AlphaGo Lee）
- [Leela Zero](https://github.com/leela-zero/leela-zero)：开源围棋 AI，本项目自对弈
  与数据格式深受其启发
- [sgfmill](https://mjw.woodcraft.me.uk/sgfmill/)：SGF 解析库
- PyTorch / Pygame 开源社区

## License

[MIT](LICENSE)
