"""
Set of graph-based curvature measures for benchmarking
"""
import numpy as np
from DiffusionEMD import DiffusionCheb
from DiffusionEMD.estimate_utils import l1_distance_matrix
from pecan import data
from DiffusionEMD.estimate_utils import l1_distance_matrix
import graphtools
from scipy.spatial.distance import pdist, squareform
import ot


class Ollivier_Ricci_Curvature_Regular_OT_with_Diffusion_Distances:
    """
    As input, takes a graphtools graph.
    Calculates the Ollivier-Ricci curvature using diffusion distance as the ground distance and Diffusion EMD as the Wasserstein distance.
    """

    def __init__(self, graph, idleness_parameter):  # TODO: Implement idlenss parameter
        self.name = "Curvature from Regular OT with L2 diffusion distance"
        # Compute symmetric diffusion operator
        self.Ms = graph.diff_aff.toarray()  # TODO: can we keep things sparse for longer
        # TODO: DEMD already does eigendecomposition with fast algorithms. Can we reuse that?
        self.M = graph.P.toarray()
        self.A = graph.K.toarray() - np.eye(len(self.Ms))
        # Create diffusion map
        self.E, self.V = np.linalg.eigh(self.Ms)
        self.diffusion_coordinates = self.V * self.E
        # create distance matrix of diffusion distances
        diffusion_distance_matrix_condensed = pdist(self.diffusion_coordinates)
        self.diffusion_distance_matrix = np.ascontiguousarray(
            squareform(diffusion_distance_matrix_condensed)
        )

    def diffusion_distances(self, i, j):
        return np.linalg.norm(
            self.diffusion_coordinates[i] - self.diffusion_coordinates[j], ord=2
        )  # use L2 norm

    def EMD(self, distributions):
        # takes two distributions as input, returns the DEMD between them.
        d = ot.emd2(
            np.ascontiguousarray(distributions[:, 0]),
            np.ascontiguousarray(distributions[:, 1]),
            self.diffusion_distance_matrix,
        )
        return d

    def curvature_between(self, i, j):
        """
        Returns the Ollivier-Ricci curvature between nodes i and j
        """
        # create a pair of distributions by diffusing two diracs
        distributions = np.zeros((len(self.A), 2))
        distributions[:, 0][i] = 1
        distributions[:, 1][j] = 1
        distributions = (
            distributions.T @ self.M
        )  # TODO: This is inefficient and could be combined with DEMD's diffusions.
        # find demd between distributions
        Wij = self.EMD(distributions.T)
        # find diffusion distance between i and j
        dij = self.diffusion_distances(i, j)
        # convert to Ollivier Ricci Curvature
        Kij = 1 - Wij / dij  # TODO: warn when dij is zero
        # debugging
        print(f"EMD is {Wij}")
        print(f"Diffusion distance is {dij}")
        return Kij


class Ollivier_Ricci_Curvature_DEMD:
    """
    As input, takes a graphtools graph.
    Calculates the Ollivier-Ricci curvature using diffusion distance as the ground distance and Diffusion EMD as the Wasserstein distance.
    """

    def __init__(
        self, graph, idleness_parameter=0.5, alpha=1
    ):  # TODO: Implement idleness parameter
        self.alpha = alpha
        self.name = "curvature from DEMD over l1 diffusion distance"
        # Compute symmetric diffusion operator
        self.Ms = graph.diff_aff.toarray()  # TODO: can we keep things sparse for longer
        # TODO: DEMD already does eigendecomposition with fast algorithms. Can we reuse that?
        self.M = graph.P.toarray()
        self.A = graph.K.toarray() - np.eye(len(self.Ms))
        # Create diffusion map
        self.E, self.V = np.linalg.eigh(self.Ms)
        self.diffusion_coordinates = self.V * self.E
        # initialize diffusion emd operator and fit to graph
        self.demd = DiffusionCheb()
        self.demd.fit(self.A)

    def diffusion_distances(self, i, j):
        # uses the diffusion ground distance defined in Tong et al.
        summed_scales = 0
        for k in range(0, 4):
            scale = 2 ** (-k)
            scaled_M = self.M ** scale
            summed_scales += scale ** self.alpha * np.sum(
                np.abs(scaled_M[i] - scaled_M[j])
            )
        return summed_scales

    def EMD(self, distributions):
        # takes two distributions as input, returns the DEMD between them.
        embeddings = self.demd.transform(distributions)
        d = np.linalg.norm(embeddings[0] - embeddings[1], ord=1)
        return d

    def curvature_between(self, i, j):
        """
        Returns the Ollivier-Ricci-DEMD curvature between nodes i and j
        """
        # create a pair of distributions by diffusing two diracs
        distributions = np.zeros((len(self.A), 2))
        distributions[:, 0][i] = 1
        distributions[:, 1][j] = 1
        distributions = (
            distributions.T @ self.M
        )  # TODO: This is inefficient and could be combined with DEMD's diffusions.
        # find demd between distributions
        Wij = self.EMD(distributions.T)
        # find diffusion distance between i and j
        dij = self.diffusion_distances(i, j)
        # convert to Ollivier Ricci Curvature
        Kij = 1 - Wij / dij  # TODO: warn when dij is zero
        # debugging
        print(f"DEMD is {Wij}")
        print(f"Diffusion distance is {dij}")
        return Kij


class Ollivier_Ricci_Curvature_DEMD_Total:
    """
    As input, takes a graphtools graph.
    Calculates the Ollivier-Ricci curvature using diffusion distance as the ground distance and Diffusion EMD as the Wasserstein distance.
    """

    def __init__(
        self, graph, idleness_parameter=0.5, lp=1
    ):  # TODO: Implement idleness parameter
        self.lp = lp
        self.name = "DEMD_Total: curvature from DEMD over DEMD between diracs"
        # Compute symmetric diffusion operator
        self.Ms = (
            graph.diff_aff
        )  # .toarray() #TODO: can we keep things sparse for longer
        # TODO: DEMD already does eigendecomposition with fast algorithms. Can we reuse that?
        self.M = graph.P  # .toarray()
        self.A = graph.K - np.eye(len(self.Ms))
        # Create diffusion map
        # 		self.E, self.V = np.linalg.eigh(self.Ms)
        # 		self.diffusion_coordinates = self.V * self.E
        # initialize diffusion emd operator and fit to graph
        self.demd = DiffusionCheb()
        self.demd.fit(self.A)

    # def diffusion_distances(self, i, j):
    # 	# uses the diffusion ground distance defined in Tong et al.
    # 	summed_scales = 0
    # 	for k in range(0,4):
    # 		scale = 2**(-k)
    # 		scaled_M = (self.M**scale)
    # 		summed_scales += scale**self.alpha * np.sum(np.abs(scaled_M[i] - scaled_M[j]))
    # 	return summed_scales

    def EMD(self, distributions):
        # takes two distributions as input, returns the DEMD between them.
        embeddings = self.demd.transform(distributions)
        d = np.linalg.norm(embeddings[0] - embeddings[1], ord=1)
        return d

    def curvature_between(self, i, j):
        """
        Returns the Ollivier-Ricci-DEMD curvature between nodes i and j
        """
        # create a pair of distributions by diffusing two diracs
        distributions = np.zeros((len(self.A), 2))
        distributions[:, 0][i] = 1
        distributions[:, 1][j] = 1
        distributions_diffused = (
            distributions.T @ self.M
        )  # TODO: This is inefficient and could be combined with DEMD's diffusions.
        # find demd between distributions
        Wij = self.EMD(distributions_diffused.T)
        # find diffusion distance between i and j
        dij = self.EMD(distributions)
        # convert to Ollivier Ricci Curvature
        Kij = 1 - Wij / dij  # TODO: warn when dij is zero
        # debugging
        print(f"EMD between diffused diracs is {Wij}")
        print(f"EMD between diracs is {dij}")
        return Kij
