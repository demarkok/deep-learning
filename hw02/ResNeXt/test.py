import torch
from torch.utils.data import TensorDataset

from ResNeXt.resnext import resnext34
from ResNeXt.trainer import Trainer


def test_resnext34():
    net = resnext34()
    data = torch.rand(10, 3, 224, 224)
    result = net(data)
    assert result.shape == torch.Size([10, 1000])


def test_trainer():
    net = resnext34()
    train_dataset = TensorDataset(torch.rand(10, 3, 224, 224), torch.zeros(10, dtype=torch.long))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_function = torch.nn.CrossEntropyLoss()
    trainer = Trainer(net, optimizer, loss_function)
    trainer.train(n_epochs=10, batch_size=1, train_data=train_dataset)


test_trainer()