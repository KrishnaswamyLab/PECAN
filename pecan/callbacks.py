"""Callback functors to imbue condensation process with additional data."""

import itertools
import warnings
from abc import ABC, abstractmethod

import numpy as np
import scipy
from ripser import Ripser
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from vineyards_oineus import get_oineus_pairs
import tqdm
from utilities import UnionFind


class Callback(ABC):
    """Generic callback class."""

    @abstractmethod
    def __call__(self, t, X, P, D):
        """Update function for a single diffusion condensation step.

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


class CalculateVineyards(Callback):
    """Vineyards Callback.

    This class keeps track of homology classes arising throughout
    the condensation process. It generates a Persistence Vineyard that
    tracks features from a Vietoris--Rips filtration at each
    time step.

    This considers the underlying point cloud as a dynamic metric
    space.
    """

    def __init__(self, dimension=1, homotopy_steps=1, max_cardinality=512):
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
        
        self.vineyard = {}

        self.vine_support = tuple() #(X,t) from previous step
        self.unique_simplexes = set()
        self.simplex_info = {}
        self.filtration_args =  {'max_radius': 5}
        self.num_homotopy_steps = homotopy_steps

    def __call__(self, t, X, P, D):
        """Update function for this functor."""
        # Nothing to do here if the point cloud is too large.
        if len(X) > self.max_cardinality:
            return
        
        #Setup at t=0
        if t == 0:
            simplices_0 = get_oineus_pairs(X,"pc",self.filtration_args)
            self.simplex_info[t] = (simplices_0)
            self.unique_simplexes.union(set(simplices_0))
            self.vineyard = {k:[] for k in self.unique_simplexes}

        else:
            # Unpacking Vine Support
            X_prev, t_prev = self.vine_support
            steps = np.linspace(1,0,self.num_homotopy_steps+1,endpoint=False)[::-1]
             # Spatial point-wise linear homotopy
             # perturbing previous space by `delta` towards current  
            for x_ in tqdm.tqdm(steps,disable=True):
               
                t_delta = t_prev * (1-x_) + t * x_
                X_delta = X_prev * (1-x_) + X * x_

                # Recording homology per timestep
                simplices_delta = get_oineus_pairs(X_delta, "pc",self.filtration_args)
                self.simplex_info[t_delta] = simplices_delta
                self.unique_simplexes.union(set(simplices_delta.keys()))

                #Updating Vines
                for simplex,feature in simplices_delta.items():
                    leaf = tuple([t_delta]) + feature
                    if simplex not in self.vineyard.keys():
                        self.vineyard[simplex] = []
                    self.vineyard[simplex].append(leaf)


            # Naive testing- can remove later for efficiency
            assert np.array_equal(X,X_delta), "Interpolation gone wrong."
        

        self.vine_support = (X.copy(),t)


    def __repr__(self):
        """Return name of callback."""
        return "CalculateVineyards"

    def finalise(self, data):
        """Update data dictionary."""

        data.update({"vineyard" : self.vineyard})
        data.update(self.filtration_args)
        data.update({"homotopy_summary": self.simplex_info})

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

        # Skip processing if we are unable to get some topological
        # features. This could be caused, for instance, by missing
        # `ripser` binaries.
        if tuples is None or points is None:
            return

        # Add additional information about the dimension of each
        # topological feature.
        dimension = np.asarray([len(c) - 1 for c, _ in tuples])

        # Adds the dimension as an additional column, turning the 2D
        # points of the diagram into 3D points.
        points = np.column_stack((points, dimension))

        self.persistence_pairs[t] = tuples
        self.persistence_points[t] = points

    def __repr__(self):
        """Return name of callback."""
        return "CalculatePersistentHomology"

    def finalise(self, data):
        """Update data dictionary."""
        data.update(
            {
                f"persistence_pairs_t_{i}": pairs
                for i, pairs in self.persistence_pairs.items()
            }
        )

        data.update(
            {
                f"persistence_points_t_{i}": pairs
                for i, pairs in self.persistence_points.items()
            }
        )

        return data


class CalculateDiffusionHomology(Callback):
    """Diffusion homology calculation callback.

    This class keeps track of homology classes arising during the
    condensation process. This calculation is the natural analogy
    to persistent homology, but instead of filtrating over scales
    we filtrate over time steps.
    """

    def __init__(self, threshold=1e-3):
        """Create new instance.

        Parameters
        ----------
        threshold : float
            Specifies a threshold for the merges. If a pair of points is
            closer than this threshold, it will be merged.
        """
        self.persistence_pairs = []
        self.edges = []
        self.uf = None
        self.threshold = threshold
        self.distances = None
        self.reset_distances = False

    def __call__(self, t, X, P, D):
        """Update function for this functor."""
        if self.uf is None:
            self.uf = UnionFind(X.shape[0])

        if self.distances is None:
            self.distances = np.full_like(D, np.inf)

        # Reset distances for all points that are *above* the specified
        # distance threshold again. Our distance shall reflect the time
        # at which the points *remain* within this distance.
        if self.reset_distances:
            mask = np.transpose(np.nonzero(D >= self.threshold))
            self.distances[mask] = np.inf

        for i1, i2 in np.transpose(np.nonzero(D < self.threshold)):

            # Update distances of the two pairs. This corresponds to
            # their diffusion merge distance, i.e. the first time at
            # which the points should be merged.
            if not np.isfinite(self.distances[i1, i2]):
                self.distances[i1, i2] = t
                self.distances[i2, i1] = t

            if i1 > i2 and self.uf.find(i1) != self.uf.find(i2):
                younger, older = self.uf.find(i1), self.uf.find(i2)

                # Store edge; the order does not really matter here
                # but it should be consistent.
                if younger < older:
                    younger, older = older, younger

                self.edges.append((t, younger, older))

                self.uf.merge(i1, i2)

                # On the connected component level, the addition of
                # this pair is easy because *everything* is created
                # at t = 0.
                self.persistence_pairs.append((0, t))

        # Ensure that all self distances are zero. We only do this once
        # and do not care about the entries during the iteration above.
        np.fill_diagonal(self.distances, 0.0)

    def __repr__(self):
        """Return name of callback."""
        return "CalculateDiffusionHomology"

    def finalise(self, data):
        """Update data dictionary."""
        T = np.max(np.ma.masked_invalid(self.distances))
        self.distances[np.isinf(self.distances)] = T

        data.update(
            {
                "diffusion_homology_persistence_pairs": np.asarray(
                    self.persistence_pairs
                ),
                "diffusion_homology_edges": np.asarray(self.edges),
                "diffusion_homology_distances": self.distances,
            }
        )

        # Calculate Betti curve over all diffusion iteration steps
        if len(self.persistence_pairs) > 0:
            betti = [
                (t, np.sum(np.asarray(self.persistence_pairs)[:, 1] >= t))
                for t in np.arange(0, T + 1)
            ]

            data.update(
                {
                    "diffusion_homology_betti": np.asarray(betti),
                }
            )

        return data


class CalculateBifiltration(Callback):
    """Stub for bifiltration calculations.

    The purpose of this callback is to be a fully-fledged callback for
    calculating persistent homology of a bifiltration. At present, the
    callback merely operates on the $1$-skeleton, employing sets of 2D
    values that are duplicates of the distance values.
    """

    def __call__(self, t, X, P, D):
        """Calculate bifiltration features.

        TODO: document me and be less terse :)
        """
        vertices = [[i] for i in range(len(X))]

        # Edges (without any duplicates: since the input range is already
        # sorted, `itertools` will report everything in *lexicographical*
        # order.
        edges = list(itertools.combinations(range(len(X)), 2))

        # Create function values for vertices first.
        function_values = [[(0, 0)] for _ in vertices]

        # Use duplicates of the distance values for each edge.
        function_values.extend([(D[i, j], D[i, j])] for i, j in edges)

        simplicial_complex = vertices + edges

        bifi = rivet.Bifiltration(
            x_label="distance",
            y_label="distance",
            simplices=simplicial_complex,
            appearances=function_values,
        )

        bifi_betti = rivet.betti(bifi, homology=1)
        rank_invariant = bifi_betti.graded_rank

        # TODO: do something with the rank invariant :)

    def __repr__(self):
        """Return name of callback."""
        return "CalculateBifiltration"


class CalculateBifiltrationDiffusionDistance_v_Distance(Callback):
    """Bifiltration calculations over diffusion time and distance."""

    def __call__(self, t, X, P, D):
        """Calculate bifiltration features.

        TODO: document me and be less terse :)
        """
        vertices = [[i] for i in range(len(X))]

        # Edges (without any duplicates: since the input range is already
        # sorted, `itertools` will report everything in *lexicographical*
        # order.
        edges = list(itertools.combinations(range(len(X)), 2))

        # Create function values for vertices first.
        function_values = [[(0, 0)] for _ in vertices]

        # Compute the diffusion distances from
        # each point to a (randomly chosen) point
        e, V = np.linalg.eig(P)
        DiffusionCoords = V.T
        Pdists = scipy.spatial.distance_matrix(DiffusionCoords)

        # Use duplicates of the distance values for each edge.
        function_values.extend([(D[i, j], D[i, j])] for i, j in edges)

        simplicial_complex = vertices + edges

        bifi = rivet.Bifiltration(
            x_label="diffusion",
            y_label="distance",
            simplices=simplicial_complex,
            appearances=function_values,
        )

        bifi_betti = rivet.betti(bifi, homology=1)
        rank_invariant = bifi_betti.graded_rank

        # TODO: do something with the rank invariant :)
        # TODO: Read rivet paper -- understand how to visualize this bifiltration, and what comes out of it + how to summarize it!


class CalculateReturnProbabilities(Callback):
    """Return probabilities calculation callback.

    This callback calculates the return probabilities for random walks
    up to a pre-defined length.

    Notes
    -----
    This callback is not fully tested. Its output should not be relied
    on in practice.
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

            return_probabilities = np.multiply(U, V)

            return_probabilities = np.sum(return_probabilities, axis=1)
            R[:, k] = return_probabilities

        # Store the return probabilities for the condensation time t.
        self.return_probabilities[t] = R

    def finalise(self, data):
        """Update data dictionary."""
        data.update(
            {
                f"return_probabilities_t_{i}": prob
                for i, prob in self.return_probabilities.items()
            }
        )

        return data


class CalculateTangentSpace(Callback):
    """Tangent space calculation callback.

    This callback calculates a tangent space and performs some
    calculations on it.
    """

    def __init__(self, n_neighbours=8):
        """Create new instance of the callback.

        Parameters
        ----------
        n_neighbours : int
            Number of neighbours to use for estimating the kernel space.
        """
        self.k = n_neighbours
        self.knn = NearestNeighbors(n_neighbors=self.k, metric="euclidean")
        self.curvature = {}

    def __call__(self, t, X, P, D):
        """Update function for this functor."""
        self.knn.fit(X)
        all_neighbours = self.knn.kneighbors(X, return_distance=False)

        curvature = []

        for i, neighbours in enumerate(all_neighbours):
            local_curvature = self._estimate_tangent_space(X, i, neighbours)
            curvature.append(local_curvature)

        # Store all curvature values for the current time step.
        self.curvature[t] = curvature

    def __repr__(self):
        """Return name of callback."""
        return "CalculateTangentSpace"

    def finalise(self, data):
        """Update data dictionary."""
        data.update(
            {f"curvature_t_{i}": curv for i, curv in self.curvature.items()}
        )

        return data

    def _estimate_tangent_space(self, X, index, neighbour_indices):
        # Create local space with `X[index]` being the base point. We
        # aim to estimate the *tangent* space around this point, so a
        # local coordinate system is required.
        Y = X[neighbour_indices, :] - X[index]

        pca = PCA()

        # Ignore some issues with the fitting process. As condensation
        # continues, the fit might be degenerate because everything is
        # converging towards a single point.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pca.fit(Y)

        components = pca.components_

        # Project all points into the respective space spanned by each of
        # the components.
        Z = [np.dot(Y, c) for c in components]
        dimension = len(Z) - 1

        x0 = np.zeros((dimension, dimension)).ravel()
        result = scipy.optimize.minimize(
            self._hypersurface_loss, x0, args=(Z, Y), method="Nelder-Mead"
        )

        x0 = result.x.reshape((dimension, dimension))
        curvature = np.linalg.det(x0)
        return curvature

    def _hypersurface_loss(self, A, *args):
        """Loss entailed by a quadratic hypersurface fit.

        Parameters
        ----------
        A : np.array of shape (d**2, )
            The coefficients for fitting the hypersurface, with `d`
            referring to the local dimension.

        *args : tuple
            Tuple containing fit parameters, viz. `Z`, the projections
            of points onto the respective basis vectors, and `Y`, the
            local tangent space.

        Returns
        -------
        Error for the fit with current parameters `A`.
        """
        Z = args[0]  # projections
        Y = args[1]  # local tangent space
        D = len(Z) - 1  # dimension of fit
        n = len(Y)  # number of points
        A = A.reshape((D, D))

        loss = 0.0

        for i in range(n):
            loss_per_point = 0.0

            # TODO: this is relatively inefficient, but at least it's
            # readable!
            for d1 in range(D):
                for d2 in range(D):
                    loss_per_point += Z[d1][i] * Z[d2][i] * A[d1, d2]

            loss_per_point -= Z[D][i]
            loss_per_point = 0.5 * loss_per_point**2

            loss += loss_per_point

        # Just to be nice, we turn it back into the format desired by
        # the optimisation function.
        A = A.ravel()
        return loss
