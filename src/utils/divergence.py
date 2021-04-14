import torch
import numpy as np
import torch.nn.functional as F


def rand_projections(embedding_dim, num_samples=50):
    projections = [w / np.sqrt((w ** 2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).float()


def sliced_wasserstein_distance(encoded_samples, distribution_samples, num_projections=1000, p=2, device='cuda'):
    embedding_dim = distribution_samples.size(1)
    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections).to(device)
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    return wasserstein_distance.mean()


# https://github.com/napsternxg/pytorch-practice/blob/master/Pytorch%20-%20MMD%20VAE.ipynb
def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1)  # (x_size, 1, dim)
    y = y.unsqueeze(0)  # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
    return torch.exp(-kernel_input)  # (x_size, y_size)


def maximum_mean_discrepancy(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()
    return mmd


# https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
def kl_between_gaussians(mu1, sigma1, mu2, sigma2, dim):
    return 1 / 2 * (torch.log(torch.prod(sigma2) / torch.prod(sigma1)) - dim + torch.sum(sigma1 / sigma2) + (
            mu2 - mu1) ** 2 / sigma2)


# http://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/
def gaussian_wasserstein_distance(mu1, sigma1, mu2, sigma2, squared=False):
    distance = torch.sum((mu1 - mu2) ** 2, dim=-1) + torch.sum((torch.sqrt(sigma1) - torch.sqrt(sigma2)) ** 2, dim=-1)

    if squared:
        return torch.mean(distance)
    else:
        return torch.mean(torch.sqrt(distance))


def kullback_leibler_gaussians(mean, log_variance):
    return -0.5 * torch.mean(1 + log_variance - mean.pow(2) - log_variance.exp())


def kullback_leibler_tensor(x, y):
    return F.kl_div(input=torch.log(x), target=y, reduction="mean")


def kullback_leibler(p, q):
    return F.kl_div(input=torch.log(p), target=q, reduction="batchmean")


def jensen_shannon(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kullback_leibler(p, m) + 0.5 * kullback_leibler(q, m)
