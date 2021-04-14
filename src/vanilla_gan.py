import argparse
import os
import numpy as np
import math
import datetime
import logging
from pathlib import Path
import shutil
from distutils.dir_util import copy_tree

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm

import preprocess_data

from models.vanila_gan import Discriminator, Generator
from utils.general import weights_init

cuda = True if torch.cuda.is_available() else False


def get_noise(n_samples, z_dim, device='cpu'):
    return torch.randn(n_samples, z_dim, device=device)


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def show_tensor_images(image_tensor, writer, type_image, step, num_images=25, size=(1, 28, 28)):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    # show images
    # matplotlib_imshow(image_grid, one_channel=True)

    # add tensorboard
    writer.add_image(type_image, image_grid, global_step=step)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")

    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")

    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    # parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    # parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "celeba", "cifar10"])
    parser.add_argument("--display_step", type=int, default=10000, help="interval between image samples")
    parser.add_argument("--gpu", type=str, default='0', help='Specify')
    parser.add_argument('--log_dir', type=str, default="vanilla_gan",
                        help='experiment root')
    return parser.parse_args()


def main():
    global img_size, channels

    def log_string(string):
        logger.info(string)
        print(string)

    args = parse_args()
    log_model = args.log_dir + "_n_epochs" + str(args.n_epochs)
    log_model = log_model + "_batch_size" + str(args.batch_size)
    log_model = log_model + "_" + args.dataset

    if args.dataset == 'mnist':
        img_size = 28
        channels = 1
    elif args.dataset == 'cifar10':
        img_size = 32
        channels = 3
    elif args.dataset == 'celeba':
        img_size = 64
        channels = 3

    '''CREATE DIR'''
    time_str = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))

    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(time_str)
    else:
        experiment_dir = experiment_dir.joinpath(log_model)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)

    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, "log"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    log_string(args)
    log_string(log_model)

    '''TENSORBROAD'''
    log_string('Creating Tensorboard ...')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensor_dir = experiment_dir.joinpath('tensorboard/')
    tensor_dir.mkdir(exist_ok=True)
    summary_writer = SummaryWriter(os.path.join(tensor_dir))

    # GPU Indicator
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Save generated images
    saved_path = experiment_dir.joinpath('images/')
    os.makedirs(saved_path, exist_ok=True)

    # Configure data loader
    dataloader = preprocess_data.generate_dataloader(name_dataset=args.dataset,
                                                     img_size=img_size,
                                                     batch_size=args.batch_size)

    # Loss functions
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(latent_dim=args.latent_dim,
                          image_shape=(channels, img_size, img_size))
    discriminator = Discriminator(image_shape=(channels, img_size, img_size))

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
    log_string("Starting Training Loop...")

    G_losses = []
    D_losses = []

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

            log_string(
                "[Epoch %d/%d] [Batch %d/%d] [Discriminator loss: %f] [Generator loss: %f]"
                % (epoch, args.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())

            steps = epoch * len(dataloader) + i
            summary_writer.add_scalar('Discriminator Loss', d_loss.item(), steps)
            summary_writer.add_scalar('Generator Loss', g_loss.item(), steps)

            if steps % args.display_step == 0:
                with torch.no_grad():
                    # save_image(gen_imgs.data[:25], args.path_images + "/%d.png" % steps, nrow=5, normalize=True)
                    fake = generator(z)

                    save_image(real_imgs.data[:25], saved_path.joinpath("_real_%d.png" % steps), nrow=5, normalize=True)
                    save_image(fake.data[:25], saved_path.joinpath("_fake_%d.png" % steps), nrow=5, normalize=True)

                    show_tensor_images(fake, summary_writer, "Fake Image", steps)
                    show_tensor_images(real_imgs, summary_writer, "Real Image", steps)

                # do checkpointing
                torch.save(generator.state_dict(),
                           checkpoints_dir.joinpath(f"{args.log_dir}_G_iter_{steps}.pth"))
                torch.save(discriminator.state_dict(),
                           checkpoints_dir.joinpath(f"{args.log_dir}_D_iter_{steps}.pth"))


if __name__ == '__main__':
    main()
