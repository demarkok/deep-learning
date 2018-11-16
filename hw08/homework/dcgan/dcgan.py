import torch.nn as nn


class DCGenerator(nn.Module):

    def __init__(self, image_size, latent_dim, image_channels):
        super(DCGenerator, self).__init__()
        self.init_size = image_size // 4

        self.init_layers = 128

        self.l1 = nn.Sequential(nn.Linear(latent_dim, self.init_layers * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(self.init_layers),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.init_layers, self.init_layers, 3, stride=1, padding=1),

            nn.BatchNorm2d(self.init_layers, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.init_layers, self.init_layers // 2, 3, stride=1, padding=1),


            nn.BatchNorm2d(self.init_layers // 2, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.init_layers // 2, image_channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.shape[0], -1)
        z = self.l1(z)
        z = z.view(z.shape[0], self.init_layers, self.init_size, self.init_size)
        img = self.conv_blocks(z)
        return img


class DCDiscriminator(nn.Module):

    def __init__(self, image_size, image_channels):
        super(DCDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                        nn.LeakyReLU(0.2, inplace=True),
                        nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(image_channels, 32, bn=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),

        )

        # The height and width of downsampled image
        ds_size = image_size // 2**4
        self.adv_layer = nn.Sequential( nn.Linear(256*ds_size**2, 1),
                                        nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
