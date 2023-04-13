"""Diffusion condensation functor."""

import logging

import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import rbf_kernel

from yaspin import yaspin
from yaspin.spinners import Spinners


class DiffusionCondensation:
    """Generic diffusion condensation functor.

    This class permits running a generic diffusion condensation process
    on a data set. It supports certain hooks by which additional steps,
    such as further processing operations, may be integrated.
    """

    def __init__(
        self,
        callbacks=[],
        prefix="data_",
        kernel_fn=None,
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

        kernel_fn : callable (optional)
            If set, overrides kernel functions to be used for the
            affinity matrix calculation. The function needs to be
            able to calculate pairwise affinities for a matrix. A
            keyword argument `epsilon` must be supported, but can
            also be ignored within the function.
        """
        self.callbacks = callbacks
        self.prefix = prefix
        self.kernel_fn = kernel_fn

        if self.kernel_fn is None:
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
            self.prefix + "t_0": X.copy(),
            "P_t_0": np.identity(n),
        }

        for callback in self.callbacks:
            callback(i, X, np.identity(n), euclidean_distances(X),A = 1-euclidean_distances(X))

        logging.info("Started diffusion condensation process")

        with yaspin(spinner=Spinners.dots) as sp:
            while i - j > 1:

                j = i

                while Q_diff >= 1e-4:

                    sp.text = f"Iteration {i}"

                    # This signals that we want to perform an additional
                    # operation of diffusion here.
                    i += 1

                    # Process new merges by checking whether their respective
                    # label assignments changed.
                    D = euclidean_distances(X)

                    # K: Affinity matrix defined by the kernel
                    # Q: Vector of degrees (row sums)
                    # P: D^-1 K
                    # A  = D^-1/2 K D^-1/2
                    K = self.kernel_fn(X, epsilon)
                    Q = np.sum(K, axis=1)
                    P = np.diag(1.0 / Q) @ K
                    A = np.diag(np.sqrt(1/Q)) @ K @ np.diag(np.sqrt(1/Q))
                    
                    # Store diffusion operator
                    data[f"P_t_{i}"] = P

                    for callback in self.callbacks:
                        callback(i, X, P, D, A)

                    X = P @ X

                    # Store new variant of the data set for the current
                    # iteration at time $i$.
                    data[f"{self.prefix}t_{i}"] = X.copy()

                    Q_diff = np.max(Q - Q_prev)
                    Q_prev = Q

                epsilon *= 2
                Q_diff = np.inf

        # Update data dictionary with all the data generated by the
        # callback.
        for callback in self.callbacks:
            data = callback.finalise(data)

        logging.info("Finished diffusion condensation process")
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
