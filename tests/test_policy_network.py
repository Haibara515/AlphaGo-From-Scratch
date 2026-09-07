"""策略网络单元测试：输入输出形状。"""

import pytest

torch = pytest.importorskip("torch")

from src.policy_network import PolicyNetworkAlphaGoLee  # noqa: E402


def test_forward_shape():
    model = PolicyNetworkAlphaGoLee(input_channels=2)
    model.eval()
    x = torch.randn(4, 2, 19, 19)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (4, 362)
