import numpy as np
import torch


def normalization(x):
    return x / torch.max(x)


def cov_torch(m, rowvar=False):
    """
    Estimate a covariance matrix given data.
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.
    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def get_inner_distances(u, metric="euclidean"):
    if metric == "euclidean":
        return euclidean_distance(u, u)
    elif metric == "squared_euclidean":
        return euclidean_distance(u, u, squared=True)
    elif metric == "covariance":
        n, _ = u.shape
        u_m = torch.mean(u, dim=-1, keepdim=True)
        return 1 / (n - 1) * torch.matmul(u - u_m, torch.transpose(u - u_m, dim0=0, dim1=1))
    else:
        raise ValueError('metric not implemented yet')


def euclidean_distance(x, y, squared=True):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, 2).sum(2)

    # replace NaNs by 0
    dist[torch.isnan(dist)] = 0
    # add small value to avoid numerical issues
    dist = dist + 1e-16
    if squared:
        return dist
    else:
        return torch.sqrt(dist)


def rbf_kernel(x, y, metric="euclidean", sigma=1.0):
    if metric == "euclidean":
        D = euclidean_distance(x, y, squared=True)
        D = torch.exp(- D / (2 * sigma * sigma))  # n x n

        return D
    else:
        raise NotImplementedError


def cosine_similarity(x, y):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)

    x = x / torch.norm(x, p=2, dim=1, keepdim=True)
    y = y / torch.norm(y, p=2, dim=1, keepdim=True)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = (x * y).sum(2)

    # replace NaNs by 0
    dist[torch.isnan(dist)] = 0
    # add small value to avoid numerical issues
    dist = dist + 1e-16

    return dist


def compute_toy_metric(data, x, y):
    n = x.shape[0]
    m = y.shape[0]

    if data in ["scurve", "swissroll"]:
        if data == "scurve":
            x = torch.clamp(x, min=-1, max=1)
            y = torch.clamp(y, min=-1, max=1)
            tx = torch.asin(x[:, 0])
            ty = torch.asin(y[:, 0])
        elif data == "swissroll":
            tx = torch.sqrt(x[:, 0] ** 2 + x[:, 2] ** 2)
            ty = torch.sqrt(y[:, 0] ** 2 + y[:, 2] ** 2)
        else:
            raise NotImplementedError

        x_coor = torch.cat((x[:, 1].reshape(-1, 1), tx.reshape(-1, 1)), dim=1)
        y_coor = torch.cat((y[:, 1].reshape(-1, 1), ty.reshape(-1, 1)), dim=1)

        d = x_coor.shape[1]

        x_coor = x_coor.unsqueeze(1).expand(n, m, d)
        y_coor = y_coor.unsqueeze(0).expand(n, m, d)

        dist = torch.sqrt(torch.pow(x_coor - y_coor, 2).sum(2))

        return dist

    elif data == "plane2d":
        return euclidean_distance(x, y, squared=False)
    else:
        raise NotImplementedError


# Reference: https://pythonot.github.io/_modules/ot/utils.html#euclidean_distances
def euclidean_distance_numpy(X, Y, squared=False):
    """
    Considering the rows of X (and Y=X) as vectors, compute the
    distance matrix between each pair of vectors.
    Parameters
    ----------
    X : {array-like}, shape (n_samples_1, n_features)
    Y : {array-like}, shape (n_samples_2, n_features)
    squared : boolean, optional
        Return squared Euclidean distances.
    Returns
    -------
    distances : {array}, shape (n_samples_1, n_samples_2)
    """
    XX = np.einsum('ij,ij->i', X, X)[:, np.newaxis]
    YY = np.einsum('ij,ij->i', Y, Y)[np.newaxis, :]
    distances = np.dot(X, Y.T)
    distances *= -2
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)
    if X is Y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0
    return distances if squared else np.sqrt(distances, out=distances)