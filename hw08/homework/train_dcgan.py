import argparse
import logging
import os

import torch
import torchvision.datasets as datasets
from torch.optim import Adam
from torchvision import transforms

from homework.dcgan.dcgan import DCGenerator, DCDiscriminator
from homework.dcgan.trainer import DCGANTrainer



def get_config():
    parser = argparse.ArgumentParser(description='Training DCGAN on CIFAR10')

    parser.add_argument('--log-root', type=str, default='../logs')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--log-name', type=str, default='train_dcgan.log')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train ')
    parser.add_argument('--image-size', type=int, default=32,
                        help='size of images to generate')
    parser.add_argument('--n_show_samples', type=int, default=8)
    parser.add_argument('--show_img_every', type=int, default=10)
    parser.add_argument('--log_metrics_every', type=int, default=100)
    config = parser.parse_args()
    config.cuda = not config.no_cuda and torch.cuda.is_available()

    return config


def get_mnist_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=256, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--image_size', type=int, default=32, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')

    parser.add_argument('--log-root', type=str, default='../logs')
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--log-name', type=str, default='train_dcgan.log')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train ')
    parser.add_argument('--n_show_samples', type=int, default=8)
    parser.add_argument('--show_img_every', type=int, default=10)
    parser.add_argument('--log_metrics_every', type=int, default=100)
    parser.add_argument('--image_channels', type=int, default=1)


    config = parser.parse_args()
    config.cuda = not config.no_cuda and torch.cuda.is_available()

    return config


# latent_dim = 100
# image_channels = 3

def main():

    config = get_mnist_config()

    if not os.path.exists(config.log_root):
        os.makedirs(config.log_root)

    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(config.log_root,
                                             config.log_name)),
            logging.StreamHandler()],
        level=logging.INFO)


    transform = transforms.Compose([transforms.Scale(config.image_size), transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = datasets.CIFAR10(root=config.data_root, download=True,
                               transform=transform)

    dataset_mnist = datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(config.image_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
    #                                          num_workers=4, pin_memory=True)

    dataloader = torch.utils.data.DataLoader(dataset_mnist,
        batch_size=config.batch_size, shuffle=True)

    discriminator, generator = DCDiscriminator(config.image_size, config.image_channels), DCGenerator(config.image_size, config.latent_dim, config.image_channels)

    trainer = DCGANTrainer(generator=generator, discriminator=discriminator,
                           optimizer_d=Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
                           optimizer_g=Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999)),
                           metrics_dir='metrics', device='cpu', latent_size=config.latent_dim)

    trainer.train(dataloader, config.epochs, config.n_show_samples, config.show_img_every, config.log_metrics_every)


if __name__ == '__main__':
    main()
    # config = get_config()
    # transform = transforms.Compose([transforms.Scale(config.image_size), transforms.ToTensor(),
    #                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #
    # dataset = datasets.CIFAR10(root=config.data_root, download=True,
    #                            transform=transform)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
    #                                          num_workers=4, pin_memory=True)
    #
    # for i, data in enumerate(dataloader):
    #     real, _ = data
    #     print(real.size())
    #     break
