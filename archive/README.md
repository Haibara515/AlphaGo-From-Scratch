# archive/ - 历史版本与旧代码归档

此目录**原样保留**开发过程中较早版本的代码，供对照参考；新开发请使用 `src/` 与
`scripts/` 中的版本。归档文件没有改写内部 import，因此不能在仓库新结构下直接运行。

| 归档文件 | 说明 | 被取代为 |
|---|---|---|
| `Go_game.py` | 较早的 Pygame 围棋界面（含规则引擎） | `src/play.py` |
| `Strategy_Network.py` | ResNet 策略网络版本 | `src/policy_network.py`（AlphaGo Lee 版） |
| `find_sgf.py` | 第一版 SGF 解析（非 sgfmill） | `scripts/process_sgf_policy.py` |
| `diagnose_sgf.py` | 数据诊断脚本（旧数据集） | `scripts/process_sgf_value.py` |
| `train.py` | 旧版 ResNet 检查点加载脚本 | `src/policy_network.py` / `src/value_network.py` |
| `test.py` | best_model.pth 兼容性检查脚本 | `tests/test_policy_network.py` |

> 原始文件仍完整保留在开发目录（`D:\pythonProject`），归档仅为仓库展示用途。
