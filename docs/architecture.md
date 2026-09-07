# 架构说明（Architecture）

本项目按 DeepMind 论文《Mastering the game of Go with deep neural networks and tree
search》（Silver et al., 2016）复刻 AlphaGo Lee：**策略网络 + 价值网络 + MCTS**。

## 1. 整体流水线

```text
   SGF 棋谱(.zip)
        │
        ▼
  scripts/process_sgf_policy.py ──► processed_data/（states + actions）
  scripts/process_sgf_value.py  ──► processed_data_value/（states + values ±1）
        │
        ▼
  src/policy_network.py ──监督学习──► checkpoints/best_model.pth（策略 π）
  src/value_network.py  ──监督学习──► checkpoints_value/best_model.pth（价值 v）
        │
        ▼
  src/mcts.py（PUCT 搜索，结合 π 与 v）
        │
        ├── src/self_play.py  ──► self_play_data/（states + probs + values，供强化学习）
        └── src/play.py       ──► Pygame 人机 / AI 对战
```

## 2. 策略网络（`src/policy_network.py`）

- 输入：`[B, 2, 19, 19]`（通道 0 = 当前方棋子，通道 1 = 对方棋子）
- 13 层纯卷积：第 1 层 5×5（padding=2），第 2~12 层 3×3（padding=1），
  全程保持 19×19 空间分辨率
- 输出头：棋盘分支 1×1 卷积 → `[B, 361]`；Pass 分支（1×1 卷积 + 全局平均池化）→ `[B, 1]`
- 输出：`[B, 362]` logits（0~360 交点，361 = Pass），配合 `CrossEntropyLoss`

```text
[B,2,19,19] ─► conv 5x5 ─► BN? ─► ... ─► 1x1 conv ─► reshape [B,361]  ┐
                                     └─ pass 头 ─► [B,1] ──────────────┴─► [B,362]
```

## 3. 价值网络（`src/value_network.py`）

- 13 层卷积（每层后接 BatchNorm）+ **全局平均池化** + 全连接头
- 输出：`[B, 1]` 线性标量（训练时与 ±1 标签做 MSE）
- 推理时可用 `tanh` 压缩到 [-1, +1]（见 `src/mcts.py` 的评估逻辑）

```text
[B,2,19,19] ─► conv+BN ×13 ─► AdaptiveAvgPool2d(1) ─► [B,192]
             ─► FC(192→256)+BN+ReLU ─► FC(256→1) ─► [B,1]
```

## 4. MCTS（`src/mcts.py`）

每步模拟四阶段：

```text
① 选择     a* = argmax_a [ Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a)) ]
② 扩展     叶节点：策略网络给出合法动作先验概率（非法动作屏蔽并重归一化）
③ 评估     终局用真实胜负；否则价值网络打分（当前行棋方视角）
④ 回传     沿路径更新 N、W，每上一层价值取反（视角切换）
```

- 动作概率 = 根节点子节点访问次数占比（`search()` 返回 362 维）
- 落子 = `select_move(temperature=0.0)` 取访问次数最多的动作
- 合法点过滤由 `get_legal_moves()` 完成：占用 / 自杀 / **位置超劫**
  （对局级历史指纹与 `GoEngine` 完全一致，保证 AI 不会走出引擎拒绝的“非法”落子）

## 5. 规则引擎（`src/play.py` 内 `GoEngine`）

- 黑先、双方轮流；同色连串共气、无气提子；自杀非法
- 打劫：位置超劫（禁止复现任意历史局面，含空盘）
- 终局：双方连续停手 / 认输 / 最大手数
- 计分：中国规则数子法（领地 + 活子，白方加贴目 7.5）

## 6. 设计取舍

- 输入只使用 2 个特征平面（当前方 / 对方），与论文 48 平面的差距是后续改进方向
- 价值标签采用「当前行棋方视角」±1，与棋盘通道语义一致，降低学习难度
- 自我对弈数据同时保存策略分布，可直接扩展为 AlphaGo Zero 式强化学习
