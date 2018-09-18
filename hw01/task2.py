import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torchvision
import numpy as np

dataset_root = "./FashionMNIST"

torch.manual_seed(42)
np.random.seed(239)
torch.cuda.manual_seed(179)

batch_size = 100
n_classes = 10


class Flip(object):
    def __call__(self, img):
        return torchvision.transforms.functional.hflip(img)


transform = torchvision.transforms.Compose([Flip(), torchvision.transforms.ToTensor()])

train_data = FashionMNIST(dataset_root, train=True, download=True, transform=transform)
test_data = FashionMNIST(dataset_root, train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


def visualize_data():
    batch, _ = iter(test_loader).next()
    grid = torchvision.utils.make_grid(batch, nrow=10)
    plt.subplots(10, 10)
    plt.axis('off')
    plt.imshow(np.transpose(grid, (1, 2, 0)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        kernel_size = 5
        padding = (kernel_size - 1) // 2
        channels1 = 2
        channels2 = 4
        layer3 = channels2 * 7 * 7  # (considering the initial size to be 28x28, and two 2x2 maxPoolings)
        layer4 = 64

        self.conv1 = nn.Conv2d(1, channels1, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(channels1, channels2, kernel_size, padding=padding)
        self.squeeze = lambda x: x.view(-1, layer3)
        self.linear1 = nn.Linear(layer3, layer4)
        self.linear2 = nn.Linear(layer4, n_classes)

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


def train(net, n_epochs):
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()

    def total_loss(loader):
        with torch.no_grad():
            losses = [loss_function(net(batch_input), batch_labels).item() for batch_input, batch_labels in loader]
            return np.average(losses)

    train_losses = []
    test_losses = []

    for i in range(n_epochs):
        for (batch_input, batch_labels) in train_loader:
            optimizer.zero_grad()
            loss = loss_function(net(batch_input), batch_labels)
            loss.backward()
            optimizer.step()
        tl = total_loss(train_loader)
        train_losses.append(tl)
        test_losses.append(total_loss(test_loader))
        print("train loss on the epoch {}: {}".format(i + 1, tl))
    plt.plot(train_losses, label="Train loss")
    plt.plot(test_losses, label="Test loss")
    plt.show()


net = Net()
train(net, 30)
with torch.no_grad():
    output = torch.cat([net(batch) for batch, _ in test_loader]).max(dim=1)[1]
    labels = torch.cat([batch_labels for _, batch_labels in test_loader])

for i in range(n_classes):
    true_positive = ((output == i) & (labels == i)).sum().double().item()
    i_class = (output == i).sum().double().item()
    precision = true_positive / i_class
    print("Precision for the class {}: {}".format(i, precision))
