import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class NetG(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(NetG, self).__init__()
        self.latent_dim = latent_dim
        self.image_shape = image_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.image_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.image_shape)
        return img


class NetD(nn.Module):
    def __init__(self, image_shape, loss_function="mse"):
        super(NetD, self).__init__()
        self.image_shape = image_shape
        self.loss_function = loss_function

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.image_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            # nn.Sigmoid(),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        if self.loss_function == "bce":
            validity = self.sigmoid(validity)

        return validity


def test():
    z = torch.randn(32, 100)
    G = NetG(latent_dim=100, image_shape=(1, 28, 28))
    # img = torch.randn(32, 28, 28, 1)
    D = NetD(image_shape=(1, 28, 28), loss_function="bce")
    print(G(z).shape)
    print(D(G(z)).shape)


if __name__ == '__main__':
    test()
