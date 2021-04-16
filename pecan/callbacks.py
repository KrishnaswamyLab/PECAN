"""Callback functors to imbue condensation process with additional data."""

import numpy as np

from abc import ABC
from abc import abstractmethod 

from ripser import Ripser

from utilities import UnionFind


class Callback(ABC):
    """Generic callback class."""

    @abstractmethod
    def __call__(self, t, X, P, D):
        """Generic update function for a single step.

        This is the main update function that is called in each step of
        the diffusion condensation process.

        Parameters
        ----------
        t : int
            Current iteration step of the diffusion condensation
            process. The callback is allowed to use this for any
            form of bookkeeping.

        X : np.array
            Current set of data points, with samples being in the rows,
            and dimensions being in the columns. Each iteration step is
            allowed to update `X`, so the callback may want to cache it
            in case operations pertain to past versions.

        P : np.array
            Transition matrix arising from the diffusion process.

        D : np.array
            Matrix of pairwise Euclidean distances between samples from
            X. This is provided for convenience purposes.
        """
        pass

    def finalise(self, data):
        """Finalise callback processing and update data dictionary.

        Tells the callback to finalise data processing and provides
        a way to optionally update the data dictionary, thus making
        it possible to store data generated by the callback.

        The default implementation performs just a pass-through.

        Parameters
        ----------
        data : dict
            Data dictionary. Contains keys generated by the diffusion
            process and other callbacks.

        Returns
        -------
        Updated data dictionary.
        """
        return data


class CalculatePersistentHomology(Callback):
    """Persistent homology calculation callback.

    This class keeps track of homology classes arising during the
    condensation process. It uses a Vietoris--Rips filtration for
    each time step to obtain persistent homology information over
    scales and time steps.

    This considers the underlying point cloud as a dynamic metric
    space.
    """

    def __init__(self, dimension=1, max_cardinality=512):
        """Build new instance of callback and set parameters.

        Parameters
        ----------
        dimension : int
            Maximum dimension for which to calculate topological
            features.

        max_cardinality : int
            Maximum cardinality of point clouds for which topological
            features should be calculated. Since calculating features
            slows down other computations, larger point clouds should
            be handled differently.
        """
        self.dimension = dimension
        self.max_cardinality = max_cardinality
        self.persistence_pairs = dict()
        self.persistence_points = dict()

    def __call__(self, t, X, P, D):
        """Update function for this functor."""
        # Nothing to do here if the point cloud is too large.
        if len(X) > self.max_cardinality:
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

    def finalise(self, data):
        """Update data dictionary."""
        data.update({
            f'persistence_pairs_t_{i}': pairs for i, pairs in
            self.persistence_pairs.items()
        })

        data.update({
            f'persistence_points_t_{i}': pairs for i, pairs in
            self.persistence_points.items()
        })

        return data


class CalculateDiffusionHomology(Callback):
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

    def finalise(self, data):
        """Update data dictionary."""
        data.update({
            'diffusion_homology_persistence_pairs': np.asarray(
                self.persistence_pairs
            )
        })

        return data


class CalculateReturnProbabilities(Callback):
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

    def finalise(self, data):
        """Update data dictionary."""
        data.update({
            f'return_probabilities_t_{i}': prob for i, prob in
            self.return_probabilities.items()
        })

        return data
