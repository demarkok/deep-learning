from torch import nn
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torchvision
import numpy as np

dataset_root = "./FashionMNIST"

batch_size = 100

train_data = FashionMNIST(dataset_root, train=True, download=True, transform=torchvision.transforms.ToTensor())
test_data = FashionMNIST(dataset_root, train=False, download=True, transform=torchvision.transforms.ToTensor())

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


def visualize_data():
    batch, labels = iter(test_loader).next()
    grid = torchvision.utils.make_grid(batch, nrow=10)
    plt.axis('off')
    plt.imshow(np.transpose(grid, (1, 2, 0)))

visualize_data()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        kernel_size = 5
        padding = (kernel_size - 1) // 2
        channels1 = 8
        channels2 = 16
        layer3 = channels2 * 7 * 7 # (considering the initial size to be 28x28, and two 2x2 maxPoolings)
        layer4 = 128

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.channels1, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=self.channels1, out_channels=channels2, kernel_size=kernel_size, padding=padding)
        self.squeeze = lambda x: x.view(-1, layer3)
        self.linear1 = nn.Linear(layer3, layer4)
        self.linear2 = nn.Linear(layer4, 10)

        self.activation = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.squeeze(x)

        x = self.linear1(x)
        x = self.activation(x)

        x = self.linear2(x)
        return self.softmax(x)


plt.show()


