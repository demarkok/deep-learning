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
plt.show()

