# models/ - 模型权重

本目录**不存放**训练好的权重文件（`.pth` 体积大，不适合直接提交到 Git）。

## 权重文件清单（训练完成后放置）

| 文件 | 来源 | 建议路径 |
|---|---|---|
| 策略网络 `best_model.pth` | `python -m src.policy_network` | `checkpoints/best_model.pth` |
| 价值网络 `best_model.pth` | `python -m src.value_network` | `checkpoints_value/best_model.pth` |

## 分发方式

- 推荐使用 **Git LFS** 管理权重：`git lfs track "*.pth"`
- 或者上传到网盘（百度网盘 / Google Drive），在 Release 页面提供下载链接
- 参考 `scripts/download_weights.sh` 中的示例命令

## 体积参考（本地实测）

- 策略网络检查点约 **29 MB**
- 价值网络检查点约 **4~30 MB**（视是否包含优化器状态）
