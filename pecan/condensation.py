import numpy as np

from sklearn.datasets import make_moons
from sklearn.metrics.pairwise import rbf_kernel


def make_affinity_matrix(X, epsilon):
    """Calculate affinity matrix.

    This functions calculates an affinity matrix from an input matrix.
    The input matrix is required to be of shape $(n, d)$, with $n$ and
    $d$ representing the number of samples and dimensions.
    """
    return rbf_kernel(X, gamma=1.0 / epsilon)


def condensation(X, epsilon):
    """Run condensation process for a given data set."""
    n = X.shape[0]

    # Denotes the previous density measurement, which is initialised to
    # an identity matrix depending on the number of samples, as well as
    # the previously-observed difference, which has to be set to inf in
    # order to be suitable.
    Q_prev = np.identity(n)
    Q_diff = np.inf

    # We only work on a copy of the data set since we shift everything,
    # i.e. points start changing their positions.
    X = X.copy()

    i = 0
    j = -2

    while i - j > 1:

        j = i

        while Q_diff >= 1e-4:

            # This signals that we want to perform an additional
            # operation of diffusion here.
            i += 1

            A = make_affinity_matrix(X, epsilon)
            Q = np.sum(A, axis=1)
            K = np.diag(1.0 / Q) @ A @ np.diag(1.0 / Q)
            P = np.diag(1.0 / np.sum(K, axis=1)) @ K
            X = P @ X

            Q_diff = np.max(Q - Q_prev)
            Q_prev = Q

        epsilon *= 2
        Q_diff = np.inf


if __name__ == '__main__':
    X = make_moons(100, random_state=42)[0]

    condensation(X, epsilon=1.0)
