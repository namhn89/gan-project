import argparse
import os
import numpy as np
import math

from torch.utils.data import DataLoader
import torch

import torchvision
import torch.utils.data
from torchvision import datasets
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torchvision.utils import save_image

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

import config

cuda = True if torch.cuda.is_available() else False


def generate_dataloader(name_dataset, img_size, batch_size):
    # Configure data loader

    # MNIST
    # Image size: (1, 28, 28)
    os.makedirs(config.DATA_MNIST, exist_ok=True)

    dataloader_mnist = torch.utils.data.DataLoader(
        datasets.MNIST(
            config.DATA_MNIST,
            train=True,
            download=False,
            transform=transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    # CIFAR10
    # Image size: (3, 32, 32)
    os.makedirs(config.DATA_CIFAR10, exist_ok=True)
    dataloader_cifar10 = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            config.DATA_CIFAR10,
            train=True,
            download=False,
            transform=transforms.Compose(
                [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    # CELEBA
    # Image size: (3, 64, 64)
    dataset = torchvision.datasets.ImageFolder(root=config.DATA_CELEBA,
                                               transform=transforms.Compose([
                                                   transforms.Resize(img_size),
                                                   transforms.CenterCrop(img_size),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                               ]))
    # Create the dataloader
    dataloader_celeba = torch.utils.data.DataLoader(dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True)
    if name_dataset == 'mnist':
        return dataloader_mnist
    elif name_dataset == 'cifar10':
        return dataloader_cifar10
    elif name_dataset == 'celeba':
        return dataloader_celeba

