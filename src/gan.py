import argparse
import os
import numpy as np
import datetime
import logging
from pathlib import Path
import shutil
import random
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

from torchvision.utils import save_image
from torchvision.utils import make_grid

import preprocess_data

from models.vanilla_gan import NetG, NetD
from models.dcgan_mnist import NetD_MNIST, NetG_MNIST
from models.dcgan_cifar10 import NetD_CIFAR10, NetG_CIFAR10
from models.dcgan_celeba import NetD_CelebA, NetG_CelebA
from utils.general import weights_init

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_noise(n_samples, z_dim):
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


def show_tensor_images(image_tensor, writer, type_image, step, num_images=25):
    """
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    """
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5, normalize=True)
    # show images
    # matplotlib_imshow(image_grid, one_channel=True)
    # add tensorboard
    writer.add_image(type_image, image_grid, global_step=step)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=200,
                        help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="size of the batches")

    parser.add_argument("--lr", type=float, default=0.0002,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--loss_function", type=str, default="mse",
                        help="Loss Function", choices=["mse", "bce"])

    parser.add_argument("--latent_dim", type=int, default=100,
                        help="dimensionality of the latent space")
    parser.add_argument("--feature_size", type=int, default=64,
                        help="dimensionality of the feature")

    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "celeba", "cifar10"])
    parser.add_argument("--display_step", type=int, default=1000,
                        help="interval between image samples")
    parser.add_argument("--save_checkpoint_step", type=int, default=10000,
                        help="Saving checkpoint after step")

    parser.add_argument("--gpu", type=str, default='0', help='Specify GPU ')
    parser.add_argument('--log_dir', type=str, default="gan",
                        help='experiment root', choices=["gan", "dcgan"])
    return parser.parse_args()


def main():
    global img_size, channels, adversarial_loss, generator, discriminator

    def log_string(string):
        logger.info(string)
        print(string)

    args = parse_args()
    log_model = args.log_dir + "_n_epochs_" + str(args.n_epochs)
    log_model = log_model + "_batch_size_" + str(args.batch_size)
    log_model = log_model + "_loss_" + str(args.loss_function)
    log_model = log_model + "_display_step_" + str(args.display_step)
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
    if tensor_dir.exists():
        shutil.rmtree(tensor_dir)
    tensor_dir.mkdir(exist_ok=True)
    summary_writer = SummaryWriter(os.path.join(tensor_dir))

    # GPU Indicator
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Save generated images
    saved_path = experiment_dir.joinpath('images/')
    os.makedirs(saved_path, exist_ok=True)

    # Configure data loader
    dataloader = preprocess_data.generate_dataloader(
        name_dataset=args.dataset,
        img_size=img_size,
        batch_size=args.batch_size
    )

    # Loss functions
    if args.loss_function == "bce":
        adversarial_loss = torch.nn.BCELoss()
    elif args.loss_function == "mse":
        adversarial_loss = torch.nn.MSELoss()

    # Initialize generator and discriminator
    if args.log_dir == "gan":
        generator = NetG(
            latent_dim=args.latent_dim,
            image_shape=(channels, img_size, img_size)
        )
        discriminator = NetD(
            image_shape=(channels, img_size, img_size),
            loss_function=args.loss_function,
        )
    elif args.log_dir == "dcgan":
        if args.dataset == 'mnist':
            generator = NetG_MNIST(
                latent_dim=args.latent_dim,
                image_shape=(channels, img_size, img_size),
                feature_size=args.feature_size
            )
            discriminator = NetD_MNIST(
                image_shape=(channels, img_size, img_size),
                feature_size=args.feature_size,
                loss_function=args.loss_function
            )
        elif args.dataset == 'cifar10':
            generator = NetG_CIFAR10(
                latent_dim=args.latent_dim,
                image_shape=(channels, img_size, img_size),
                feature_size=args.feature_size
            )
            discriminator = NetD_CIFAR10(
                image_shape=(channels, img_size, img_size),
                feature_size=args.feature_size,
                loss_function=args.loss_function
            )
        elif args.dataset == 'celeba':
            generator = NetG_CelebA(
                latent_dim=args.latent_dim,
                image_shape=(channels, img_size, img_size),
                feature_size=args.feature_size
            )
            discriminator = NetD_CelebA(
                image_shape=(channels, img_size, img_size),
                feature_size=args.feature_size,
                loss_function=args.loss_function
            )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

    # Assign device for model, criterion
    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # ----------
    #  Training
    # ----------
    log_string("Starting Training Loop...")

    G_losses = []
    D_losses = []

    fixed_noise = torch.randn(args.batch_size, args.latent_dim, device=device)

    for epoch in range(args.n_epochs):
        for i, (images, _) in enumerate(dataloader):

            # Adversarial ground truths
            real_label = 1.
            fake_label = 0.

            # Configure input
            label = torch.full((images.size(0), 1), real_label, dtype=torch.float, device=device)
            real_images = images.to(device)

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            # Train with all-real batch

            discriminator.zero_grad()
            output = discriminator(real_images)

            errD_real = adversarial_loss(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(images.size(0), args.latent_dim, device=device)

            gen_images = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(gen_images.detach())
            # Calculate D's loss on the all-fake batch
            errD_fake = adversarial_loss(output, label)
            errD_fake.backward()
            # Add the gradients from the all-real and all-fake batches
            errD = (errD_real + errD_fake) / 2
            D_G_z1 = output.mean().item()
            # Update D
            optimizer_D.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################

            generator.zero_grad()
            # Since we just updated D, perform another forward pass of all-fake batch through D
            label.fill_(real_label)
            output = discriminator(gen_images)
            # Calculate G's loss based on this output
            errG = adversarial_loss(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizer_G.step()

            if i % 50 == 0:
                log_string(
                    "[Epoch %d/%d] [Batch %d/%d]"
                    "\t[Loss_D: %.4f]\t[Loss_G: %.4f]\t[D(x): %.4f]\t[D(G(z)): %.4f / %.4f]"
                    % (epoch, args.n_epochs, i, len(dataloader),
                       errD.item(), errG.item(), D_x, D_G_z1, D_G_z2)
                )

            D_losses.append(errD.item())
            G_losses.append(errG.item())

            steps = epoch * len(dataloader) + i
            summary_writer.add_scalars(
                'Loss',
                {
                    'D': errD.item(),
                    'G': errG.item()
                },
                steps
            )
            summary_writer.add_scalar('D(x)', D_x, steps)
            summary_writer.add_scalar('D(G(z1))', D_G_z1, steps)
            summary_writer.add_scalar('D(G(z2))', D_G_z2, steps)

            if steps % args.display_step == 0:
                with torch.no_grad():
                    fake = generator(fixed_noise)

                    save_image(real_images.data[:25], saved_path.joinpath("real_%d.png" % steps),
                               nrow=5, normalize=True)
                    save_image(fake.data[:25], saved_path.joinpath("fake_%d.png" % steps),
                               nrow=5, normalize=True)

                    show_tensor_images(fake, summary_writer, "Fake Image", steps)
                    show_tensor_images(real_images, summary_writer, "Real Image", steps)

            # do checkpointing
            if steps % args.save_checkpoint_step == 0:
                torch.save(generator.state_dict(),
                           checkpoints_dir.joinpath(f"{args.log_dir}_G_iter_{steps}.pth"))
                torch.save(discriminator.state_dict(),
                           checkpoints_dir.joinpath(f"{args.log_dir}_D_iter_{steps}.pth"))

    # Plot lossy graph
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator")
    plt.plot(D_losses, label="Discriminator")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    plt.savefig(experiment_dir.joinpath('graph.png'))
    summary_writer.add_figure("Graph Loss", plt.gcf())
    summary_writer.close()


if __name__ == '__main__':
    main()
