"""Kernel functions and related methods."""

import warnings

import numpy as np

from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.metrics.pairwise import euclidean_distances


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

    warnings.warn(
        f'Falling back to default kernel instead of using kernel "{kernel}".'
    )

    # Fall back to default kernel.
    return None
