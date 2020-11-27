"""Topology-based diffusion condensation scheme."""

import argparse
import sys

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel

from yaspin import yaspin
from yaspin.spinners import Spinners

from data import barbell
from data import double_annulus
from data import moons
from data import hyperuniform_circle
from data import hyperuniform_ellipse

from ripser import Ripser

from utilities import UnionFind


def analyse_persistence_diagram(data):
    """Analyse persistence diagram statistics."""
    pd = np.asarray(data['D'])
    if len(pd) == 0:
        return

    pd = pd / np.max(pd)

    persistence_values = np.sum(pd, axis=1)
    total_persistence = np.sum(persistence_values) / len(persistence_values)

    print(f'Total persistence: {total_persistence:.2f}')


def make_affinity_matrix(X, epsilon):
    """Calculate affinity matrix.

    This functions calculates an affinity matrix from an input matrix.
    The input matrix is required to be of shape $(n, d)$, with $n$ and
    $d$ representing the number of samples and dimensions.
    """
    return rbf_kernel(X, gamma=1.0 / epsilon)


class CalculateDiffusionHomology:
    """Calculate diffusion homology.

    This class keeps track of homology classes arising during the
    condensation process. This calculation is the natural analogy
    to persistent homology, but instead of filtrating over scales
    we filtrate over time steps.
    """

    def __init__(self):
        """Create new instance."""
        self.persistence_pairs = []
        self.uf = None

    def __call__(self, t, X, P, D):
        """Update function for this functor."""
        if not self.uf:
            self.uf = UnionFind(X.shape[0])

        for i1, i2 in np.transpose(np.nonzero(D < 1e-3)):
            if i1 > i2 and self.uf.find(i1) != self.uf.find(i2):
                self.uf.merge(i1, i2)

                # On the connected component level, the addition of
                # this pair is easy because *everything* is created
                # at t = 0.
                self.persistence_pairs.append((0, t))


class DiffusionCondensation:
    """Generic diffusion condensation functor.

    This class permits running a generic diffusion condensation process
    on a data set. It supports certain hooks by which additional steps,
    such as further processing operations, may be integrated.
    """

    def __init__(self, callbacks=[]):
        """Initialise new instance and register callbacks."""
        self.callbacks = []

    def __call__(self, X, epsilon):
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

        # Will store the data set per iteration to check whether the
        # implementation works as expected.
        data = {
            't_0': X.copy(),
        }

        # FIXME: will contain all return probabilities observed as the
        # algorithm runs. This is a way to detect general loops in the
        # data; not sure how to link that to TDA.
        R = []

        # Current time-inhomogeneous diffusion operator. Will be updated in
        # each step. This is used to keep track of return probabilities.
        P_t = np.identity(n)

        with yaspin(spinner=Spinners.dots) as sp:
            while i - j > 1:

                j = i

                while Q_diff >= 1e-4:

                    sp.text = f'Iteration {i}'

                    # This signals that we want to perform an additional
                    # operation of diffusion here.
                    i += 1

                    # Process new merges by checking whether their respective
                    # label assignments changed.
                    D = euclidean_distances(X)

                    A = make_affinity_matrix(X, epsilon)
                    Q = np.sum(A, axis=1)
                    K = np.diag(1.0 / Q) @ A @ np.diag(1.0 / Q)
                    P = np.diag(1.0 / np.sum(K, axis=1)) @ K

                    for callback in self.callbacks:
                        callback(i, X, P, D)

                    X = P @ X

                    # Store new variant of the data set for the current
                    # iteration at time $i$.
                    data[f't_{i}'] = X.copy()

                    Q_diff = np.max(Q - Q_prev)
                    Q_prev = Q

                epsilon *= 2
                Q_diff = np.inf

        return data


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
    data = {
        't_0': X.copy(),
    }

    # Will contain the persistence pairs generated during the algorithm.
    # For dimension 0, they will contain only the destruction times, but
    # I will store them as 2-tuples to ensure consistency.
    persistence_pairs = []

    # FIXME: will contain all return probabilities observed as the
    # algorithm runs. This is a way to detect general loops in the
    # data; not sure how to link that to TDA.
    R = []

    # Current time-inhomogeneous diffusion operator. Will be updated in
    # each step. This is used to keep track of return probabilities.
    P_t = np.identity(n)

    with yaspin(spinner=Spinners.dots) as sp:
        while i - j > 1:

            j = i

            while Q_diff >= 1e-4:

                sp.text = f'Iteration {i}'

                # This signals that we want to perform an additional
                # operation of diffusion here.
                i += 1

                # Process new merges by checking whether their respective
                # label assignments changed.
                D = euclidean_distances(X)

                for i1, i2 in np.transpose(np.nonzero(D < 1e-3)):
                    if i1 > i2 and uf.find(i1) != uf.find(i2):
                        uf.merge(i1, i2)

                        # On the connected component level, the addition of
                        # this pair is easy because *everything* is created
                        # at t = 0.
                        persistence_pairs.append((0, i))

                A = make_affinity_matrix(X, epsilon)
                Q = np.sum(A, axis=1)
                K = np.diag(1.0 / Q) @ A @ np.diag(1.0 / Q)
                P = np.diag(1.0 / np.sum(K, axis=1)) @ K
                X = P @ X

                # Calculate topological features alongside the diffusion
                # process. These are features generated by the intrinsic
                # geometry of the diffusion process.

                tuples, points = Ripser(dimension=1)(D)

                mask = [len(c) == 2 for c, _ in tuples]

                cycle_pairs = tuples[mask]
                cycle_points = points[mask]

                # TODO: make this configurable; I am not sure whether it
                # is the smartest choice to change the lookup of
                # topological features here.
                if False:
                    cycle_pairs_ = []

                    for sigma, tau in cycle_pairs:
                        sigma = [uf.find(v) for v in sigma]
                        tau = [uf.find(v) for v in tau]
                        cycle_pairs_.append((sigma, tau))

                    cycle_pairs = np.asarray(cycle_pairs_, dtype=object)

                data[f'pairs_{i}'] = cycle_pairs
                data[f'points_{i}'] = cycle_points

                eigenvalues, eigenvectors = np.linalg.eigh(P)

                eigenvalues = eigenvalues**16

                return_probabilities = np.multiply(eigenvectors, eigenvectors)
                return_probabilities = np.multiply(
                    return_probabilities,
                    eigenvalues
                )
                return_probabilities = np.sum(return_probabilities, axis=1)

                print(return_probabilities)

                R.append(return_probabilities)

                # Store new variant of the data set for the current
                # iteration at time $i$.
                data[f't_{i}'] = X.copy()

                Q_diff = np.max(Q - Q_prev)
                Q_prev = Q

            epsilon *= 2
            Q_diff = np.inf

    if True:
        # FIXME: this can be made smarter; just visualises the return
        # probabilities over the diffusion process.
        R = np.asarray(R)
        plt.matshow(R)
        plt.show()

    data['D'] = np.asarray(persistence_pairs)
    return data


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

    parser.add_argument(
        '-e', '--epsilon',
        # TODO: ensure that this makes sense and be adjusted more
        # easily, depending on the number of points etc.
        default=np.pi / 128,
        type=float,
    )

    parser.add_argument(
        '-o', '--output',
        default=None,
        type=str,
    )

    args = parser.parse_args()
    this = sys.modules[__name__]

    # Search for a generator routine, as requested by the client. This
    # does not fail gracefully.
    generator = getattr(this, args.data)

    X, C = generator(args.num_samples, random_state=42)
    data = condensation(X, args.epsilon)

    diffusion_homology = CalculateDiffusionHomology()

    diffusion_condensation = DiffusionCondensation([diffusion_homology])
    diffusion_condensation(X, args.epsilon)

    print(diffusion_homology.persistence_pairs)

    #analyse_persistence_diagram(data)

    #data['C'] = C

    #if args.output is None:
    #    args.output = args.data + '.npz'

    #np.savez(args.output, **data)
