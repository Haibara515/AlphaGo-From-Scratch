"""价值网络单元测试：输入输出形状。"""

import pytest

torch = pytest.importorskip("torch")

from src.value_network import ValueNetworkAlphaGoLee  # noqa: E402


def test_forward_shape():
    model = ValueNetworkAlphaGoLee(input_channels=2)
    model.eval()
    x = torch.randn(4, 2, 19, 19)
    with torch.no_grad():
        value = model(x)
    assert value.shape == (4, 1)
