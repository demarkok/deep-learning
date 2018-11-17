import torch.nn as nn


class DCGenerator(nn.Module):

    def __init__(self, latent_dim, image_channels):
        super(DCGenerator, self).__init__()

        init_layers = 512

        self.new_blocks = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, init_layers, kernel_size=4, stride=1),
            nn.BatchNorm2d(init_layers),

            nn.ConvTranspose2d(init_layers, init_layers // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_layers // 2),

            nn.ConvTranspose2d(init_layers // 2, init_layers // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(init_layers // 4),

            nn.ConvTranspose2d(init_layers // 4, image_channels, kernel_size=4, stride=2, padding=1),

            nn.Tanh()

        )

    def forward(self, z):

        z = self.new_blocks(z)
        return z

        # z = z.view(z.shape[0], -1)
        # z = self.l1(z)
        # z = z.view(z.shape[0], self.init_layers, self.init_size, self.init_size)
        # img = self.conv_blocks(z)
        # return img


class DCDiscriminator(nn.Module):

    def __init__(self, image_channels):
        super(DCDiscriminator, self).__init__()

        init_layers = 64

        self.new_blocks = nn.Sequential(
            # 32x32

            nn.Conv2d(image_channels, init_layers, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(init_layers),
            # 16x16

            nn.Conv2d(init_layers, init_layers * 2, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(init_layers * 2),

            # 8x8
            nn.Conv2d(init_layers * 2, init_layers * 4, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(init_layers * 4),
            # 4x4

            nn.Conv2d(init_layers * 4, 1, kernel_size=4),
            nn.Sigmoid()
        )

    def forward(self, img):
        img = self.new_blocks(img)
        return img
