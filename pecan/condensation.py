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
    # an identity matrix depending on the number of samples.
    Q_prev = np.identity(n)

    X = X.copy()
    A = make_affinity_matrix(X, epsilon)
    Q = np.sum(A, axis=1)
    K = np.diag(1.0 / Q) @ A @ np.diag(1.0 / Q)
    P = np.diag(1.0 / np.sum(K, axis=1)) @ K
    X = P @ X

    Q_diff = np.max(Q - Q_prev)


if __name__ == '__main__':
    X = make_moons(100, random_state=42)[0]

    condensation(X, epsilon=1.0)
