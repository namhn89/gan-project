from torch.nn import MSELoss, BCELoss


def bce_loss(x, y, reduction="mean"):
    return BCELoss(reduction=reduction)(x, y)


def mse_loss(x, y, reduction="mean"):
    return MSELoss(reduction=reduction)(x, y)