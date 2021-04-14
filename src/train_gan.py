import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm

import config
import preprocess_data

from models.vanila_gan import Discriminator, Generator
from utils.general import weights_init

cuda = True if torch.cuda.is_available() else False


def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=10000, help="interval between image samples")
    parser.add_argument("--path_images", type=str, default="images_vanilla_gan", help="Saving generated images")
    parser.add_argument("--gpu", type=str, )
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    # Save generated images
    os.makedirs(args.path_images, exist_ok=True)

    # Configure data loader
    dataloader = preprocess_data.generate_dataloader(name_dataset='mnist',
                                                     img_size=args.img_size,
                                                     batch_size=args.batch_size)

    # Loss functions
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(latent_dim=args.latent_dim,
                          image_shape=(args.channels, args.img_size, args.img_size))
    discriminator = Discriminator(image_shape=(args.channels, args.img_size, args.img_size))

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # ----------
    #  Training
    # ----------

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [Discriminator loss: %f] [Generator loss: %f]"
                % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            steps = epoch * len(dataloader) + i
            if steps % args.sample_interval == 0:
                save_image(gen_imgs.data[:25], args.path_images + "/%d.png" % steps, nrow=5, normalize=True)


if __name__ == '__main__':
    main()
