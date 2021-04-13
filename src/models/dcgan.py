import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_dim, image_shape, feature_size):
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.feature_size = feature_size

        super(Generator, self).__init__()

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.feature_size * 4,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.feature_size * 4),
            nn.ReLU(True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(self.feature_size * 4, self.feature_size * 2,
                               kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_size * 2),
            nn.ReLU(True)
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(self.feature_size * 2, self.feature_size,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_size),
            nn.ReLU(True),
        )

        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(self.feature_size, self.image_shape[2],
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        out = self.deconv1(z)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, image_shape, feature_size):
        self.image_shape = image_shape
        self.feature_size = feature_size
        super(Discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.image_shape[2], self.feature_size,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.feature_size, self.feature_size * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.feature_size * 2, self.feature_size * 4,
                      kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.feature_size * 4, self.image_shape[2],
                      kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img):
        out = self.conv1(img)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out.view(-1, self.image_shape[2])


def test():
    z = torch.randn(32, 100)
    G = Generator(latent_dim=100, image_shape=(28, 28, 1), feature_size=128)
    # img = torch.randn(32, 28, 28, 1)
    D = Discriminator(image_shape=(28, 28, 1), feature_size=128)
    print(G(z).shape)
    print(D(G(z)).shape)


if __name__ == '__main__':
    test()
