"""Kernel functions and related methods."""

import warnings

import numpy as np

from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import euclidean_distances


def alpha_decaying_kernel(X, epsilon, alpha=10):
    r"""Calculate the $\alpha$-decaying kernel function.

    Calculate a simplified variant of the $\alpha$-decaying kernel as
    described by Moon et al. [1]. In contrast to the original
    description, this kernel only uses a *single* local density estimate
    instead of per-point estimates.

    Parameters
    ----------
    X : np.array of size (n, n)
        Input data set.

    epsilon : float
        Standard deviation or local scale parameter. This parameter is
        globally used and does *not* depend on the local neighbourhood
        of a point.

    alpha : float
        Value for the decay.

    Returns
    -------
    Kernel matrix.

    References
    -----
    [1]: Moon et al., Visualizing Structure and Transitions for
    Biological Data Exploration, Nature Biotechnology 37, pp. 1482–1492,
    2019. URL: https://www.nature.com/articles/s41587-019-0336-3
    """
    D = euclidean_distances(X)
    return np.exp(-(D / epsilon)**alpha)


def get_kernel_fn(kernel):
    """Return kernel function as callable."""
    if kernel == 'gaussian':
        # Default kernel; handled by the diffusion condensation functor,
        # so there's nothing to do for us.
        return None
    elif kernel == 'laplacian':
        def kernel_fn(X, epsilon):
            return laplacian_kernel(X, gamma=1.0 / epsilon)
        return kernel_fn
    elif kernel == 'constant':
        def kernel_fn(X, epsilon):
            n = X.shape[0]
            K = np.full((n, n), 1.0 / epsilon)
            return K
        return kernel_fn
    elif kernel == 'box':
        def kernel_fn(X, epsilon):
            K = euclidean_distances(X)

            # Use mask because we have to ensure that the same positions
            # are being matched when setting kernel values manually.
            mask = K <= epsilon

            K[mask] = 1.0
            K[~mask] = 0.0

            return K
        return kernel_fn
    elif kernel == 'alpha':
        def kernel_fn(X, epsilon):
            return alpha_decaying_kernel(X, epsilon)
        return kernel_fn

    warnings.warn(
        f'Falling back to default kernel instead of using kernel "{kernel}".'
    )

    # Fall back to default kernel.
    return None
