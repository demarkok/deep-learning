import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, model, optimizer, loss_function):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

    def train(self, n_epochs, batch_size, train_data, test_data=None):

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        if test_data is not None:
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        def total_loss(loader):
            with torch.no_grad():
                losses = [self.loss_function(self.model(batch_input), batch_labels).item()
                          for batch_input, batch_labels in loader]
                return np.average(losses)

        writer = SummaryWriter()

        for i in range(n_epochs):
            for (batch_input, batch_labels) in train_loader:
                self.optimizer.zero_grad()
                loss = self.loss_function(self.model(batch_input), batch_labels)
                loss.backward()
                self.optimizer.step()
            writer.add_scalar('train loss', total_loss(train_loader))
            if test_data is not None:
                writer.add_scalars('test loss', total_loss(test_loader))

        writer.close()
