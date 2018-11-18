import argparse
import logging
import os

import torch
import torchvision.datasets as datasets
from torch.optim import Adam
from torchvision import transforms

from vae.trainer import Trainer
from vae.vae import VAE, loss_function


def get_config():
    parser = argparse.ArgumentParser(description='Training VAE on CIFAR10')

    parser.add_argument('--log-root', type=str, default='../logs')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--log-name', type=str, default='train_vae.log')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train ')
    parser.add_argument('--image-size', type=int, default=32,
                        help='size of images to generate')
    parser.add_argument('--image_channels', type=int, default=3, help='number of image channels')
    parser.add_argument('--n_show_samples', type=int, default=8)
    parser.add_argument('--show_img_every', type=int, default=10)
    parser.add_argument('--log_metrics_every', type=int, default=100)
    parser.add_argument('--latent_size', type=int, default=100, help='dimensionality of the latent space')
    config = parser.parse_args()
    config.cuda = not config.no_cuda and torch.cuda.is_available()
    config.device = 'cuda' if config.cuda else 'cpu'

    return config


def main():
    config = get_config()

    print(config.device)

    if not os.path.exists(config.log_root):
        os.makedirs(config.log_root)

    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_root,
                                             config.log_name)),
            logging.StreamHandler()],
        level=logging.INFO)

    transform = transforms.Compose([transforms.Scale(config.image_size), transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(root=config.data_root, download=True, transform=transform, train=True)
    test_dataset = datasets.CIFAR10(root=config.data_root, download=True, transform=transform, train=False)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                                   num_workers=4, pin_memory=True,)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                                                  num_workers=4, pin_memory=True, )

    vae = VAE(image_size=config.image_size, image_channels=config.image_channels, latent_size=config.latent_size)
    trainer = Trainer(vae, train_loader=train_dataloader, test_loader=test_dataloader,
                      optimizer=Adam(vae.parameters(), lr=0.0002, betas=(0.5, 0.999)), device=config.device,
                      loss_function=loss_function)

    for epoch in range(config.epochs):
        trainer.train(epoch, log_interval=config.log_metrics_every)
        trainer.test(epoch, batch_size=config.batch_size, log_interval=config.log_metrics_every,
                     show_img_every=config.show_img_every, n_show_samples=config.n_show_samples)


if __name__ == '__main__':
    main()
