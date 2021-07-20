"""Kernel functions and related methods."""

import numpy as np

from sklearn.metrics.pairwise import laplacian_kernel


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

    # Fall back to default kernel.
    return None

