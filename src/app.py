import streamlit as st
import time
import torch

from models.vanilla_gan import NetG, NetD
from models.dcgan_mnist import NetD_MNIST, NetG_MNIST
from models.dcgan_cifar10 import NetD_CIFAR10, NetG_CIFAR10
from models.dcgan_celeba import NetD_CelebA, NetG_CelebA
from torchvision.utils import save_image
from PIL import Image


def get_noise(n_samples, z_dim, device):
    return torch.randn(n_samples, z_dim, device=device)


def main():
    global channels, img_size, generator
    app_formal_name = "Generative Adversarial Networks Visualization"
    st.title(app_formal_name)

    MODELS = ["dcgan", "gan"]
    DATASETS = ["celeba", "mnist", "cifar10"]

    add_models = st.sidebar.selectbox(
        "Please choose a model",
        MODELS
    )

    add_datasets = st.sidebar.selectbox(
        "Please choose a dataset",
        DATASETS
    )

    loss_function = st.sidebar.selectbox(
        "Please a choose loss function",
        ["bce", "mse"]
    )

    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    latent_dim = 100
    feature_size = 64

    if add_datasets == "celeba" and add_models == "dcgan" and loss_function == "bce":
        latent_dim = 64

    if add_datasets == 'mnist':
        img_size = 28
        channels = 1
    elif add_datasets == 'cifar10':
        img_size = 32
        channels = 3
    elif add_datasets == 'celeba':
        img_size = 64
        channels = 3

    if add_models == "gan":
        generator = NetG(
            latent_dim=latent_dim,
            image_shape=(channels, img_size, img_size)
        )
        discriminator = NetD(
            image_shape=(channels, img_size, img_size),
            loss_function=loss_function,
        )
    elif add_models == "dcgan":
        if add_datasets == 'mnist':
            generator = NetG_MNIST(
                latent_dim=latent_dim,
                image_shape=(channels, img_size, img_size),
                feature_size=feature_size
            )
            discriminator = NetD_MNIST(
                image_shape=(channels, img_size, img_size),
                feature_size=feature_size,
                loss_function=loss_function
            )
        elif add_datasets == 'cifar10':
            generator = NetG_CIFAR10(
                latent_dim=latent_dim,
                image_shape=(channels, img_size, img_size),
                feature_size=feature_size
            )
            discriminator = NetD_CIFAR10(
                image_shape=(channels, img_size, img_size),
                feature_size=feature_size,
                loss_function=loss_function
            )
        elif add_datasets == 'celeba':
            generator = NetG_CelebA(
                latent_dim=latent_dim,
                image_shape=(channels, img_size, img_size),
                feature_size=feature_size
            )
            discriminator = NetD_CelebA(
                image_shape=(channels, img_size, img_size),
                feature_size=feature_size,
                loss_function=loss_function
            )

    NUM_SAMPLES = 8
    NUM_ROWS = 8
    z = get_noise(NUM_ROWS * NUM_SAMPLES, latent_dim, device=device)
    PATH = '../weights/' + add_models + "_" + add_datasets + "_" + loss_function + ".pth"
    generator.load_state_dict(torch.load(PATH, map_location=device))
    generator.eval()

    play = st.sidebar.button("Play")
    if play:
        with torch.no_grad():
            fakes = generator(z)
            # print(fakes.shape)
            save_image(fakes, "out.png",
                       nrow=NUM_ROWS, normalize=True)
        image = Image.open("out.png")
        new_width = 600
        new_height = 600
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
        st.image(resized_image, caption="Generated Image")


if __name__ == '__main__':
    main()
