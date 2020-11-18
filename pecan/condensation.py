"""Topology-based diffusion condensation scheme."""

import argparse
import sys

import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel

from utilities import UnionFind

from data import moons
from data import hyperuniform_circle
from data import hyperuniform_ellipse


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
    uf = UnionFind(n_vertices=n)

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

    # Will store the data set per iteration to check whether the
    # implementation works as expected.
    X_per_iteration = {
        0: X.copy(),
    }

    while i - j > 1:

        j = i

        while Q_diff >= 1e-4:

            # This signals that we want to perform an additional
            # operation of diffusion here.
            i += 1

            # Process new merges by checking whether their respective
            # label assignments changed.
            D = euclidean_distances(X)

            for i1, i2 in np.transpose(np.nonzero(D < 1e-3)):
                if i1 > i2 and uf.find(i1) != uf.find(i2):
                    uf.merge(i1, i2)

                    # TODO: assign a proper persistence for the points
                    # here. It's not clear whether we should use time,
                    # epsilon, or something else.

            A = make_affinity_matrix(X, epsilon)
            Q = np.sum(A, axis=1)
            K = np.diag(1.0 / Q) @ A @ np.diag(1.0 / Q)
            P = np.diag(1.0 / np.sum(K, axis=1)) @ K
            X = P @ X

            # Store new variant of the data set for the current
            # iteration at time $i$.
            X_per_iteration[i] = X.copy()

            Q_diff = np.max(Q - Q_prev)
            Q_prev = Q

        epsilon *= 2
        Q_diff = np.inf

    return X_per_iteration


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data',
        default='hyperuniform_ellipse',
        type=str,
    )

    parser.add_argument(
        '-n', '--num-samples',
        default=128,
        type=int
    )

    args = parser.parse_args()
    this = sys.modules[__name__]

    generator = getattr(this, args.data)
    X, C = generator(args.num_samples, random_state=42)
