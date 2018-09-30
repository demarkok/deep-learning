import torch

from resnext import resnext34


def test_resnext34():
    net = resnext34()
    data = torch.rand(10, 3, 224, 224)
    result = net(data)
    assert result.shape == torch.Size([10, 1000])
