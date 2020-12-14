"""Topology-based diffusion condensation scheme."""

import argparse
import logging
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

from utilities import generate_output_filename
from utilities import UnionFind


class CalculatePersistentHomology:
    """Persistent homology calculation callback.

    This class keeps track of homology classes arising during the
    condensation process. It uses a Vietoris--Rips filtration for
    each time step to obtain persistent homology information over
    scales and time steps.

    This considers the underlying point cloud as a dynamic metric
    space.
    """

    def __init__(self, dimension=1):
        """Build new instance of callback and set parameters.

        Parameters
        ----------
        dimension : int
            Maximum dimension for which to calculate topological
            features.
        """
        self.dimension = dimension
        self.persistence_pairs = dict()
        self.persistence_points = dict()

        # TODO: if set, this class will require an additional
        # Union--Find structure for rewriting generators. Not
        # sure whether this is smart.
        self.rewrite_generators = False

    def __call__(self, t, X, P, D):
        """Update function for this functor."""
        # FIXME: this limit is hard-coded for now because the analysis
        # of larger data sets is made more complicated.
        if len(X) > 256:
            return

        # Calculate topological features alongside the diffusion
        # process. These are features generated by the intrinsic
        # geometry of the diffusion process.
        #
        # The Ripser class will return persistence pairs *and* a
        # set of persistence points, i.e. coordinates/distances.
        tuples, points = Ripser(dimension=self.dimension)(D)

        # Add additional information about the dimension of each
        # topological feature.
        dimension = np.asarray([
            len(c) - 1 for c, _ in tuples
        ])

        # Adds the dimension as an additional column, turning the 2D
        # points of the diagram into 3D points.
        points = np.column_stack((points, dimension))

        self.persistence_pairs[t] = tuples
        self.persistence_points[t] = points

        # TODO: make this configurable; I am not sure whether it is
        # the smartest choice to change the lookup of topological
        # features here.
        #
        # HIC SVNT LEONES
        if self.rewrite_generators:

            tuples_ = []

            for sigma, tau in tuples:
                sigma = [uf.find(v) for v in sigma]
                tau = [uf.find(v) for v in tau]
                tuples_.append((sigma, tau))

            tuples = np.asarray(tuples, dtype=object)


class CalculateDiffusionHomology:
    """Diffusion homology calculation callback.

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


class CalculateReturnProbabilities:
    """Return probabilities calculation callback.

    This callback calculates the return probabilities for random walks
    up to a pre-defined length.
    """

    def __init__(self, K):
        """Create new instance of the callback.

        Parameters
        ----------
        K : int
            Maximum length of a random walk to use for the calculation
            of return probabilities.
        """
        self.K = K
        self.return_probabilities = dict()

    def __call__(self, t, X, P, D):
        """Update function for this functor."""
        # TODO: use different matrix for the decomposition; need to
        # account for diagonal terms here?
        eigenvalues, eigenvectors = np.linalg.eigh(P)
        eigenvalues = eigenvalues**16

        U = np.multiply(eigenvectors, eigenvectors)

        # Create a matrix that will store the return probabilities. The
        # matrix will be of shape (n, K).
        n = X.shape[0]
        R = np.empty((n, self.K))

        for k in range(self.K):

            # TODO: potentially use a smarter calculation here so that
            # we do not have to recompute them every time?
            V = eigenvalues**k

            return_probabilities = np.multiply(
                U,
                V
            )

            return_probabilities = np.sum(return_probabilities, axis=1)
            R[:, k] = return_probabilities

        # Store the return probabilities for the condensation time t.
        self.return_probabilities[t] = R


class DiffusionCondensation:
    """Generic diffusion condensation functor.

    This class permits running a generic diffusion condensation process
    on a data set. It supports certain hooks by which additional steps,
    such as further processing operations, may be integrated.
    """

    def __init__(
        self,
        callbacks=[],
        prefix='data_',
    ):
        """Initialise new instance and register callbacks.

        Parameters
        ----------
        callbacks : list of callable
            Function objects (functors) that will be called during each
            iteration. Every functor instance must satisfy a signature,
            as described below::

                callback(i, X, P, D)

            Where `i` is the current time step, `X` is the current data,
            `P` is the current diffusion operator, and `D` is a distance
            matrix between data points (using the Euclidean distance).

        prefix : str
            Indicates the prefix to be used for storing individual time
            steps. If set to `X`, the first key of the diffusion
            condensation process will be called `X_t_0`.
        """
        self.callbacks = callbacks
        self.prefix = prefix

        # TODO: this could be made configurable, but at present, I am
        # relying on a precise signature of this function.
        self.kernel_fn = self.make_affinity_matrix

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
            self.prefix + 't_0': X.copy(),
        }

        for callback in self.callbacks:
            callback(i, X, np.identity(n), euclidean_distances(X))

        logging.info('Started diffusion condensation process')

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

                    A = self.kernel_fn(X, epsilon)
                    Q = np.sum(A, axis=1)
                    K = np.diag(1.0 / Q) @ A @ np.diag(1.0 / Q)
                    P = np.diag(1.0 / np.sum(K, axis=1)) @ K

                    for callback in self.callbacks:
                        callback(i, X, P, D)

                    X = P @ X

                    # Store new variant of the data set for the current
                    # iteration at time $i$.
                    data[f'{self.prefix}t_{i}'] = X.copy()

                    Q_diff = np.max(Q - Q_prev)
                    Q_prev = Q

                epsilon *= 2
                Q_diff = np.inf

        logging.info('Finished diffusion condensation process')
        return data

    @staticmethod
    def make_affinity_matrix(X, epsilon):
        """Calculate affinity matrix.

        This functions calculates an affinity matrix from an input matrix.
        The input matrix is required to be of shape $(n, d)$, with $n$ and
        $d$ representing the number of samples and dimensions.

        Parameters
        ----------
        X : np.array of shape (n, m)
            Input matrix with `n` samples and `m` features.

        epsilon : float
            Smoothing parameter for the kernel calculation.
        """
        return rbf_kernel(X, gamma=1.0 / epsilon)


if __name__ == '__main__':

    # Set up logging to obtain some nice output information, runtime,
    # and much more.
    logging.basicConfig(
        format='%(asctime)s.%(msecs)03d  [%(levelname)-10s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.DEBUG
    )

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
        default=np.nan,
        type=float,
    )

    parser.add_argument(
        '-o', '--output',
        default=None,
        type=str,
    )

    args = parser.parse_args()
    this = sys.modules[__name__]

    if np.isnan(args.epsilon):
        args.epsilon = np.pi / args.num_samples

        logging.info(
            f'Epsilon parameter has not been set. Calculating '
            f'it based on {args.num_samples} points as '
            f'{args.epsilon:.4f}.'
        )

    logging.info(f'Data set: {args.data}')
    logging.info(f'Number of samples: {args.num_samples}')
    logging.info(f'Epsilon: {args.epsilon:.4f}')

    # Search for a generator routine, as requested by the client. This
    # does not fail gracefully.
    generator = getattr(this, args.data)

    logging.debug(f'Using generator routine {generator}')

    X, C = generator(args.num_samples, random_state=42)

    diffusion_homology = CalculateDiffusionHomology()
    return_probabilities = CalculateReturnProbabilities(K=8)
    persistent_homology = CalculatePersistentHomology()

    diffusion_condensation = DiffusionCondensation(
        [
            diffusion_homology,
            return_probabilities,
            persistent_homology
        ]
    )

    data = diffusion_condensation(X, args.epsilon)

    # Collect additional information provided by the functors and add it
    # to the output data frame.

    data.update({
        f'return_probabilities_t_{i}': prob for i, prob in
        return_probabilities.return_probabilities.items()
    })

    data.update({
        'diffusion_homology_persistence_pairs': np.asarray(
            diffusion_homology.persistence_pairs
        )
    })

    data.update({
        f'persistence_pairs_t_{i}': pairs for i, pairs in
        persistent_homology.persistence_pairs.items()
    })

    data.update({
        f'persistence_points_t_{i}': pairs for i, pairs in
        persistent_homology.persistence_points.items()
    })

    # Storing the full data set. This is either specified by the client
    # or we pick an output filename.

    if args.output is None:
        args.output = generate_output_filename(args)
    elif args.output == '-':
        args.output = sys.stdout.buffer

    logging.info(f'Storing results in {args.output}')

    np.savez(args.output, **data)
