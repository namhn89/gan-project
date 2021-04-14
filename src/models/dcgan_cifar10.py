import torch
from torch import nn

class NetG_CIFAR10(nn.Module):
    def __init__(self, latent_dim, image_shape, feature_size):
        super(NetG_CIFAR10, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape
        self.feature_size = feature_size

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.latent_dim, out_channels=self.feature_size * 4,
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.feature_size * 4),
            nn.ReLU(True),
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.feature_size * 4, out_channels=self.feature_size * 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_size * 2),
            nn.ReLU(True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.feature_size * 2, out_channels=self.feature_size,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_size),
            nn.ReLU(True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.feature_size, out_channels=self.image_shape[0],
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


class NetD_CIFAR10(nn.Module):
    def __init__(self, image_shape, feature_size):
        super(NetD_CIFAR10, self).__init__()
        self.image_shape = image_shape
        self.feature_size = feature_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=self.image_shape[0], out_channels=self.feature_size,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_size, out_channels=self.feature_size * 2,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_size * 2, out_channels=self.feature_size * 4,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=self.feature_size * 4, out_channels=1,
                      kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        out = self.conv1(img)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return out.view(-1, 1)


def test_CIFAR10():
    z = torch.randn(128, 100)
    G = NetG_CIFAR10(latent_dim=100, image_shape=(3, 32, 32), feature_size=128)
    # img = torch.randn(128, 3, 32, 32)
    D = NetD_CIFAR10(image_shape=(3, 32, 32), feature_size=128)
    print(G(z).shape)
    print(D(G(z)).shape)


if __name__ == '__main__':
    test_CIFAR10()
