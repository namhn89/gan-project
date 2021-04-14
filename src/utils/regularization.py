import torch
import numpy as np


# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
def compute_gradient_penalty(discriminator, real_samples: torch.Tensor, fake_samples: torch.Tensor):
    Tensor = torch.cuda.FloatTensor

    batch_size = real_samples.shape[0]

    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    if len(real_samples.shape) == 4:
        alpha = torch.rand((batch_size, 1, 1, 1)).cuda()
    elif len(real_samples.shape) == 2:
        alpha = torch.rand((batch_size, 1)).cuda()
    else:
        raise NotImplementedError

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1.0 - alpha) * fake_samples)).requires_grad_(True)

    d_interpolates = discriminator(interpolates).view(batch_size, 1)
    fake = torch.ones((batch_size, 1), requires_grad=False).cuda()
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
